# -*- coding: utf-8 -*-
"""
Flask VIP Tài/Xỉu - No NumPy, No scikit-learn
- Thu thập dữ liệu định kỳ từ API (poll thread)
- Bộ suy luận "AI-lite" thuần Python:
  + Markov n-gram (k = 1..10) có trọng số theo độ dài pattern & độ phủ mẫu
  + Phân tích cầu: bệt, 1-1, 2-1, 2-2, 2-3, hỗn hợp
  + Xu hướng gần (last 10 / last 20)
  + Tần suất toàn cục (global bias)
  + Giảm nhiễu theo entropy (nếu xác suất mẫu kém rõ ràng)
- Kết hợp bỏ phiếu có trọng số → dự đoán & độ tin cậy không random
"""

import requests
import time
import math
import threading
from collections import Counter, deque, defaultdict
from flask import Flask, jsonify

# =========================
# Cấu hình
# =========================
POLL_URL = "https://toilavinhmaycays23.onrender.com/vinhmaycay"
POLL_INTERVAL_SEC = 30
MAX_HISTORY = 500  # lưu tối đa 500 phiên gần nhất

# Trọng số tổ hợp (có thể tinh chỉnh)
W_MARKOV = 0.46
W_PATTERN = 0.28
W_LOCAL_TREND = 0.14
W_GLOBAL_FREQ = 0.12

# Giới hạn độ tin cậy hiển thị (chống overfit)
CONF_MIN = 52.0
CONF_MAX = 97.5

app = Flask(__name__)

# =========================
# Bộ nhớ
# =========================
history = deque(maxlen=MAX_HISTORY)   # list dict: {'phien': int, 'ket_qua': 'Tài'|'Xỉu'}
last_phien = None
lock = threading.Lock()

# Markov store: dict[k][pattern_str]['Tài'|'Xỉu'] = count
# pattern_str là chuỗi 'T'/'X' của k kết quả gần nhất
markov_counts = {k: defaultdict(lambda: {'Tài': 0, 'Xỉu': 0}) for k in range(1, 11)}

# =========================
# Utils
# =========================
def as_TX(result_str):
    """Chuyển đổi kết quả 'Tài'/'Xỉu' sang 'T'/'X'."""
    return 'T' if result_str == 'Tài' else 'X'

def from_TX(ch):
    """Chuyển đổi ký tự 'T'/'X' sang chuỗi 'Tài'/'Xỉu'."""
    return 'Tài' if ch == 'T' else 'Xỉu'

def current_streak(results):
    """Trả về (độ dài chuỗi bệt hiện tại, 'Tài'/'Xỉu' hoặc None)."""
    if not results:
        return 0, None
    side = results[-1]
    streak = 1
    for r in reversed(results[:-1]):
        if r == side:
            streak += 1
        else:
            break
    return streak, side

def entropy(p):
    """Entropy nhị phân (0..1). p là xác suất Tài."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    # log base 2 để chuẩn hoá [0..1]
    return - (p*math.log(p, 2) + (1-p)*math.log(1-p, 2))

# =========================
# Cập nhật Markov theo lịch sử
# =========================
def rebuild_markov(results):
    """Xây lại thống kê Markov n-gram từ đầu (khi khởi động hoặc reset lớn)."""
    for k in range(1, 11):
        markov_counts[k].clear()
    if len(results) < 2:
        return
    # Duyệt qua lịch sử để đếm pattern → kết quả tiếp theo
    tx = [as_TX(r) for r in results]
    for k in range(1, 11):
        if len(tx) <= k:
            continue
        for i in range(len(tx) - k):
            pattern = ''.join(tx[i:i+k])
            nxt = tx[i+k]
            out = from_TX(nxt)
            markov_counts[k][pattern][out] += 1

def update_markov_with_new(results):
    """Cập nhật Markov incremental khi có kết quả mới."""
    if len(results) < 2:
        return
    tx = [as_TX(r) for r in results]
    # thêm cặp cuối cùng cho mọi k
    for k in range(1, 11):
        if len(tx) > k:
            pattern = ''.join(tx[-(k+1):-1])
            nxt = tx[-1]
            out = from_TX(nxt)
            markov_counts[k][pattern][out] += 1

# =========================
# Phân tích cầu (pattern analysis)
# =========================
def smart_pattern_analysis(results):
    """
    Phát hiện & chấm điểm cầu.
    Trả về: (bridge_labels, pattern_vote) trong đó:
      - bridge_labels: list mô tả ngắn
      - pattern_vote: dict {'Tài': score_float, 'Xỉu': score_float}
    """
    labels = []
    vote = {'Tài': 0.0, 'Xỉu': 0.0}
    n = len(results)
    if n == 0:
        return labels, vote

    # Cầu bệt
    streak, side = current_streak(results)
    if streak >= 3:
        labels.append(f"Cầu bệt {side} ({streak})")
        # ưu tiên mạnh theo bệt, tăng theo streak (giảm dần biên độ)
        s = min(12.0 + (streak-3)*2.5, 28.0)
        vote[side] += s

    # Cầu 1-1 (T-X-T-X hoặc X-T-X-T) phát hiện dựa trên 4 cuối
    if n >= 4:
        last4 = results[-4:]
        if last4 in (['Tài','Xỉu','Tài','Xỉu'], ['Xỉu','Tài','Xỉu','Tài']):
            labels.append("Cầu 1-1")
            # dự đoán đảo chiều so với phiên gần nhất
            next_side = 'Tài' if results[-1] == 'Xỉu' else 'Xỉu'
            vote[next_side] += 18.0

    # Cầu 2-2 (T,T,X,X) hoặc (X,X,T,T)
    if n >= 4:
        last4 = results[-4:]
        if last4 in (['Tài','Tài','Xỉu','Xỉu'], ['Xỉu','Xỉu','Tài','Tài']):
            labels.append("Cầu 2-2")
            # xu hướng lặp tiếp nhóm đối ứng
            # ví dụ T,T,X,X → ưu tiên Tài (đảo nhóm), nhưng nhẹ hơn 1-1
            prefer = 'Tài' if last4 == ['Tài','Tài','Xỉu','Xỉu'] else 'Xỉu'
            vote[prefer] += 12.0

    # Cầu 2-1 (T,T,X) hoặc (X,X,T)
    if n >= 3:
        last3 = results[-3:]
        if last3 in (['Tài','Tài','Xỉu'], ['Xỉu','Xỉu','Tài']):
            labels.append("Cầu 2-1")
            # theo nhánh dài
            prefer = 'Tài' if last3[:2] == ['Tài','Tài'] else 'Xỉu'
            vote[prefer] += 10.0

    # Cầu 2-3 (T,T,X,X,X) hoặc (X,X,T,T,T)
    if n >= 5:
        last5 = results[-5:]
        if last5 in (['Tài','Tài','Xỉu','Xỉu','Xỉu'], ['Xỉu','Xỉu','Tài','Tài','Tài']):
            labels.append("Cầu 2-3")
            # theo nhóm 3 dài
            prefer = 'Xỉu' if last5[-3:] == ['Xỉu','Xỉu','Xỉu'] else 'Tài'
            vote[prefer] += 14.0

    # Nếu không có cầu rõ ràng, ghi chú
    if not labels:
        labels.append("Không có cầu rõ ràng")

    return labels, vote

# =========================
# Dự đoán Markov (k=1..10)
# =========================
def markov_predict(results):
    """
    Trả về (prob_Tai, info_str, cover_samples)
      - prob_Tai: xác suất Tài theo tổng hợp Markov nhiều bậc
      - info_str: mô tả ngắn gọn
      - cover_samples: tổng số mẫu theo tất cả k
    """
    if len(results) < 2:
        return 0.5, "Markov: thiếu dữ liệu", 0

    tx = [as_TX(r) for r in results]
    agg_weight = 0.0
    agg_prob_t = 0.0
    total_followers = 0
    details = []

    for k in range(1, 11):
        if len(tx) <= k:
            continue
        prefix = ''.join(tx[-k:])
        counts = markov_counts[k].get(prefix, None)
        if not counts:
            continue
        cT = counts['Tài']
        cX = counts['Xỉu']
        total = cT + cX
        if total == 0:
            continue
        # Xác suất theo k
        pT = cT / total
        # Trọng số: k * log(1+total) (độ dài pattern và độ phủ)
        w = k * math.log(1 + total, 2)
        agg_prob_t += pT * w
        agg_weight += w
        total_followers += total
        details.append(f"k={k}:{cT}/{total}T")

    if agg_weight == 0.0:
        return 0.5, "Markov: chưa khớp pattern", 0

    prob_t = agg_prob_t / agg_weight
    info = "Markov[" + ", ".join(details[:4]) + (", ..." if len(details) > 4 else "") + "]"
    return prob_t, info, total_followers

# =========================
# Xu hướng & Tần suất
# =========================
def local_trend(results, lookback=10):
    """Tỉ lệ Tài trong lookback gần nhất (mặc định 10)."""
    if not results:
        return 0.5, 0
    m = min(len(results), lookback)
    seg = results[-m:]
    cT = sum(1 for r in seg if r == 'Tài')
    return cT / m, m

def global_freq(results):
    """Tỉ lệ Tài toàn cục."""
    if not results:
        return 0.5, 0
    cT = sum(1 for r in results if r == 'Tài')
    return cT / len(results), len(results)

# =========================
# Hợp nhất phiếu & tính độ tin cậy
# =========================
def combine_votes(prob_markov, pattern_vote, prob_local, prob_global, cover_markov, n_local, n_global, bridges):
    """
    Hợp nhất xác suất từ các mô-đun vào xác suất cuối cùng (Tài).
    - pattern_vote là điểm 'Tài'/'Xỉu' → chuyển thành xác suất pattern
    """
    # Pattern → prob bằng softmax tay (không numpy)
    sT = pattern_vote.get('Tài', 0.0)
    sX = pattern_vote.get('Xỉu', 0.0)
    if sT == 0.0 and sX == 0.0:
        prob_pattern = 0.5
    else:
        # softmax 2 lớp
        eT = math.exp(sT / 12.0)
        eX = math.exp(sX / 12.0)
        prob_pattern = eT / (eT + eX)

    # Điều chỉnh trọng số theo độ phủ mẫu (nhiều mẫu → tin cậy hơn)
    # scale vào [0.5..1.0]
    wM = 0.5 + min(0.5, math.log(1 + cover_markov, 3) / 5.0)
    wL = 0.5 + min(0.5, math.log(1 + n_local, 3) / 5.0)
    wG = 0.5 + min(0.5, math.log(1 + n_global, 3) / 5.0)

    # Trọng số nền
    WM = W_MARKOV * wM
    WP = W_PATTERN
    WL = W_LOCAL_TREND * wL
    WG = W_GLOBAL_FREQ * wG

    # Xác suất tổng hợp
    p = (prob_markov * WM +
         prob_pattern * WP +
         prob_local * WL +
         prob_global * WG) / (WM + WP + WL + WG)

    # Giảm tin cậy nếu entropy cao (gần 50/50)
    H = entropy(p)  # 0..1
    conf = (1.0 - H) * 100.0  # 0..100

    # Tăng/giảm nhẹ theo số lượng cầu rõ ràng
    clear_patterns = [b for b in bridges if b != "Không có cầu rõ ràng"]
    if clear_patterns:
        conf *= min(1.15, 1.03 + 0.03 * len(clear_patterns))  # tối đa +15%
    else:
        conf *= 0.96  # không có cầu rõ → giảm nhẹ

    # Chặn min/max để UI không “ảo”
    conf = max(CONF_MIN, min(CONF_MAX, conf))

    predict = 'Tài' if p >= 0.5 else 'Xỉu'
    return predict, conf, prob_pattern

# =========================
# Hàm dự đoán chính
# =========================
def predict_vip(results):
    """
    Dự đoán Tài/Xỉu dựa trên:
      - Markov k=1..10
      - Phân tích cầu
      - Xu hướng gần (10) & tần suất toàn cục
      - Kết hợp xác suất + điều chỉnh entropy → do_tin_cay
    """
    n = len(results)
    if n == 0:
        return 'Tài', 50.0, "Chưa có dữ liệu."

    bridges, pattern_vote = smart_pattern_analysis(results)
    prob_markov, markov_info, cover = markov_predict(results)
    prob_local10, n_local = local_trend(results, lookback=10)
    prob_global, n_global = global_freq(results)

    predict, conf, prob_pattern = combine_votes(
        prob_markov, pattern_vote, prob_local10, prob_global,
        cover, n_local, n_global, bridges
    )

    # Tạo mô tả ngắn gọn, dễ hiểu
    brief_pattern = ", ".join(bridges)
    p_markov_pct = f"{prob_markov*100:.1f}%"
    p_pattern_pct = f"{prob_pattern*100:.1f}%"
    p_local_pct = f"{prob_local10*100:.1f}%"
    p_global_pct = f"{prob_global*100:.1f}%"

    explain = (
        f"{brief_pattern}. "
        f"Markov: {p_markov_pct} Tài ({markov_info}). "
        f"Pattern: {p_pattern_pct} Tài. "
        f"Gần 10: {p_local_pct} Tài, Toàn cục: {p_global_pct} Tài. "
        f"Chốt: {predict}."
    )

    return predict, conf, explain

# =========================
# Polling API
# =========================
def poll_api():
    global last_phien
    while True:
        try:
            resp = requests.get(POLL_URL, timeout=10)
            resp.raise_for_status() # Ném lỗi nếu HTTP request thất bại
            data = resp.json()

            phien = data.get('Phien')
            ket_qua = data.get('Ket_qua')  # 'Tài' or 'Xỉu'

            if phien is None or ket_qua not in ('Tài', 'Xỉu'):
                time.sleep(POLL_INTERVAL_SEC)
                continue

            with lock:
                global history
                global markov_counts
                # khởi tạo lại nếu rỗng
                if not history:
                    history.append({'phien': phien, 'ket_qua': ket_qua})
                    last_phien = phien
                    rebuild_markov([h['ket_qua'] for h in history])
                else:
                    if phien > last_phien:
                        history.append({'phien': phien, 'ket_qua': ket_qua})
                        last_phien = phien
                        update_markov_with_new([h['ket_qua'] for h in history])

        except Exception as e:
            # In ra lỗi nhưng không dừng thread
            print(f"[poll_api] Error: {e}")

        time.sleep(POLL_INTERVAL_SEC)

# Khởi động polling thread (daemon)
threading.Thread(target=poll_api, daemon=True).start()

# =========================
# API Endpoints
# =========================
@app.route('/predict', methods=['GET'])
def get_predict():
    with lock:
        if not history:
            return jsonify({'error': 'No data available yet. Waiting for API poll.'}), 503

        results = [h['ket_qua'] for h in history]
        pred, confidence, explain = predict_vip(results)

        # Lấy snapshot data hiện tại để trả về cho tiện UI
        try:
            resp = requests.get(POLL_URL, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # Dùng dữ liệu cũ nếu API bị lỗi
            data = {'Phien': history[-1]['phien'], 'Ket_qua': history[-1]['ket_qua']}

        session = data.get('Phien')
        dice = f"{data.get('Xuc_xac_1', '')} - {data.get('Xuc_xac_2', '')} - {data.get('Xuc_xac_3', '')}"
        total = data.get('Tong', '')
        result = data.get('Ket_qua')
        next_session = session + 1 if isinstance(session, int) else None
        pattern_str = ''.join(['T' if r == 'Tài' else 'X' for r in results[-20:]])

    return jsonify({
        'session': session,
        'dice': dice,
        'total': total,
        'result': result,
        'next_session': next_session,
        'predict': pred,
        'do_tin_cay': f"{confidence:.1f}%",
        'giai_thich': explain,
        'pattern': pattern_str,
        'id': 'Tele@idol_vannhat'
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    with lock:
        results = [h['ket_qua'] for h in history]
        n = len(results)
        cT = sum(1 for r in results if r == 'Tài')
        cX = n - cT
        streak, side = current_streak(results)
        recent10 = results[-10:] if n >= 10 else results
        cT10 = sum(1 for r in recent10 if r == 'Tài')
    return jsonify({
        'total_samples': n,
        'tai_count': cT,
        'xiu_count': cX,
        'current_streak': streak,
        'streak_side': side,
        'recent10_tai': cT10,
        'recent10_xiu': len(recent10) - cT10
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'message': 'App is running.'})

