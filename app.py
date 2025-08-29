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
# Địa chỉ API để lấy dữ liệu. Lưu ý, Render có thể tắt API nếu không có request
# Thường xuyên.
POLL_URL = "https://toilavinhmaycays23.onrender.com/vinhmaycay"
POLL_INTERVAL_SEC = 30  # Chu kỳ poll
MAX_HISTORY = 500  # Lưu tối đa 500 phiên gần nhất

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
# Bộ nhớ dùng chung giữa các luồng
# =========================
# Sử dụng deque để quản lý lịch sử hiệu quả hơn
history = deque(maxlen=MAX_HISTORY)   # list dict: {'phien': int, 'ket_qua': 'Tài'|'Xỉu'}
last_phien = None
# Khóa để đồng bộ truy cập vào các biến dùng chung
lock = threading.Lock()

# Markov store: dict[k][pattern_str]['Tài'|'Xỉu'] = count
markov_counts = {k: defaultdict(lambda: {'Tài': 0, 'Xỉu': 0}) for k in range(1, 11)}

# =========================
# Các hàm tiện ích
# =========================
def as_TX(result_str):
    """Chuyển 'Tài'/'Xỉu' thành 'T'/'X'."""
    return 'T' if result_str == 'Tài' else 'X'

def from_TX(ch):
    """Chuyển 'T'/'X' thành 'Tài'/'Xỉu'."""
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

# =========================
# Cập nhật Markov
# =========================
def rebuild_markov():
    """Xây lại thống kê Markov n-gram từ đầu."""
    for k in range(1, 11):
        markov_counts[k].clear()
    if len(history) < 2:
        return
    
    tx = [as_TX(h['ket_qua']) for h in history]
    for k in range(1, 11):
        if len(tx) <= k:
            continue
        for i in range(len(tx) - k):
            pattern = ''.join(tx[i:i+k])
            nxt = tx[i+k]
            out = from_TX(nxt)
            markov_counts[k][pattern][out] += 1

def update_markov_with_new():
    """Cập nhật Markov incremental khi có kết quả mới."""
    if len(history) < 2:
        return
    
    tx = [as_TX(h['ket_qua']) for h in history]
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
    Trả về: (bridge_labels, pattern_vote)
    """
    labels = []
    vote = {'Tài': 0.0, 'Xỉu': 0.0}
    n = len(results)
    if n < 3:
        return ["Không đủ dữ liệu"], vote

    # Cầu bệt
    streak, side = current_streak(results)
    if streak >= 3:
        labels.append(f"Cầu bệt {side} ({streak})")
        s = min(12.0 + (streak-3)*2.5, 28.0)
        vote[side] += s

    # Các cầu khác (1-1, 2-2, v.v.)
    def check_pattern(name, pattern, score, next_predict_side):
        if tuple(results[-len(pattern):]) == pattern:
            labels.append(f"Cầu {name}")
            vote[next_predict_side] += score
    
    check_pattern("1-1", ('Tài', 'Xỉu', 'Tài', 'Xỉu'), 18.0, 'Tài')
    check_pattern("1-1", ('Xỉu', 'Tài', 'Xỉu', 'Tài'), 18.0, 'Xỉu')
    check_pattern("2-2", ('Tài', 'Tài', 'Xỉu', 'Xỉu'), 12.0, 'Tài')
    check_pattern("2-2", ('Xỉu', 'Xỉu', 'Tài', 'Tài'), 12.0, 'Xỉu')
    check_pattern("2-1", ('Tài', 'Tài', 'Xỉu'), 10.0, 'Tài')
    check_pattern("2-1", ('Xỉu', 'Xỉu', 'Tài'), 10.0, 'Xỉu')
    check_pattern("2-3", ('Tài', 'Tài', 'Xỉu', 'Xỉu', 'Xỉu'), 14.0, 'Tài')
    check_pattern("2-3", ('Xỉu', 'Xỉu', 'Tài', 'Tài', 'Tài'), 14.0, 'Xỉu')
    
    if not labels:
        labels.append("Không có cầu rõ ràng")

    return labels, vote

# =========================
# Dự đoán Markov & các yếu tố khác
# =========================
def markov_predict(results):
    """
    Tổng hợp dự đoán từ các bậc Markov.
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
        
        cT, cX = counts['Tài'], counts['Xỉu']
        total = cT + cX
        if total == 0:
            continue
        
        pT = cT / total
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

def local_trend(results, lookback=10):
    """Tỉ lệ Tài trong lookback gần nhất."""
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

def entropy(p):
    """Tính entropy nhị phân."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return - (p*math.log(p, 2) + (1-p)*math.log(1-p, 2))

# =========================
# Hợp nhất phiếu & tính độ tin cậy
# =========================
def combine_votes(prob_markov, pattern_vote, prob_local, prob_global, cover_markov, bridges):
    """
    Hợp nhất xác suất từ các mô-đun vào xác suất cuối cùng (Tài).
    """
    sT = pattern_vote.get('Tài', 0.0)
    sX = pattern_vote.get('Xỉu', 0.0)
    if sT == 0.0 and sX == 0.0:
        prob_pattern = 0.5
    else:
        eT = math.exp(sT / 12.0)
        eX = math.exp(sX / 12.0)
        prob_pattern = eT / (eT + eX)

    # Điều chỉnh trọng số
    wM = W_MARKOV * (0.5 + min(0.5, math.log(1 + cover_markov, 3) / 5.0))
    wL = W_LOCAL_TREND * (0.5 + min(0.5, math.log(1 + len(history), 3) / 5.0))
    wG = W_GLOBAL_FREQ * (0.5 + min(0.5, math.log(1 + len(history), 3) / 5.0))
    
    total_weight = wM + W_PATTERN + wL + wG
    if total_weight == 0:
        return 'Tài', CONF_MIN, 0.5

    p = (prob_markov * wM +
         prob_pattern * W_PATTERN +
         prob_local * wL +
         prob_global * wG) / total_weight

    # Tính độ tin cậy dựa trên entropy
    H = entropy(p)  # 0..1
    conf = (1.0 - H) * 100.0  # 0..100
    
    clear_patterns = [b for b in bridges if b != "Không có cầu rõ ràng"]
    if clear_patterns:
        conf *= min(1.15, 1.03 + 0.03 * len(clear_patterns))
    else:
        conf *= 0.96

    conf = max(CONF_MIN, min(CONF_MAX, conf))

    predict = 'Tài' if p >= 0.5 else 'Xỉu'
    return predict, conf, prob_pattern

# =========================
# Hàm dự đoán chính
# =========================
def predict_vip(results):
    """
    Dự đoán Tài/Xỉu dựa trên nhiều yếu tố.
    """
    n = len(results)
    if n == 0:
        return 'Tài', 50.0, "Chưa có dữ liệu."

    bridges, pattern_vote = smart_pattern_analysis(results)
    prob_markov, markov_info, cover = markov_predict(results)
    prob_local10, _ = local_trend(results, lookback=10)
    prob_global, _ = global_freq(results)

    predict, conf, prob_pattern = combine_votes(
        prob_markov, pattern_vote, prob_local10, prob_global,
        cover, bridges
    )

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
            resp = requests.get(POLL_URL, timeout=15)
            data = resp.json()
            phien = data.get('Phien')
            ket_qua = data.get('Ket_qua')

            if phien is None or ket_qua not in ('Tài', 'Xỉu'):
                print("[poll_api] Invalid data received.")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            with lock:
                global history
                if last_phien is None:
                    # Lần chạy đầu tiên hoặc reset
                    history.append({'phien': phien, 'ket_qua': ket_qua})
                    last_phien = phien
                    rebuild_markov()
                else:
                    if phien > last_phien:
                        # Thêm kết quả mới nếu có phiên mới
                        history.append({'phien': phien, 'ket_qua': ket_qua})
                        last_phien = phien
                        update_markov_with_new()

        except requests.exceptions.Timeout:
            print("[poll_api] Request timed out.")
        except requests.exceptions.RequestException as e:
            print(f"[poll_api] Request error: {e}")
        except Exception as e:
            print(f"[poll_api] Unexpected error: {e}")
            
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

        session_data = {}
        try:
            resp = requests.get(POLL_URL, timeout=8)
            session_data = resp.json()
        except Exception:
            # Fallback to local history if API fails
            pass

        session = session_data.get('Phien', history[-1]['phien'])
        dice = f"{session_data.get('Xuc_xac_1', '')} - {session_data.get('Xuc_xac_2', '')} - {session_data.get('Xuc_xac_3', '')}"
        total = session_data.get('Tong', '')
        result = session_data.get('Ket_qua', history[-1]['ket_qua'])
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
    return jsonify({'ok': True, 'history_size': len(history), 'last_phien': last_phien})

# =========================
# Chạy app
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
