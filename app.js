// -*- coding: utf-8 -*-
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

import express from 'express';
import fetch from 'node-fetch';
import {
    Counter,
    deque,
    defaultdict
} from 'collections';
import {
    EventEmitter
} from 'events';

// =========================
// Cấu hình
// =========================
const POLL_URL = "https://toilavinhmaycays23.onrender.com/vinhmaycay";
const POLL_INTERVAL_SEC = 30;
const MAX_HISTORY = 500; // lưu tối đa 500 phiên gần nhất

// Trọng số tổ hợp (có thể tinh chỉnh)
const W_MARKOV = 0.46;
const W_PATTERN = 0.28;
const W_LOCAL_TREND = 0.14;
const W_GLOBAL_FREQ = 0.12;

// Giới hạn độ tin cậy hiển thị (chống overfit)
const CONF_MIN = 52.0;
const CONF_MAX = 97.5;

const app = express();
const port = process.env.PORT || 5000;

// =========================
// Bộ nhớ
// =========================
const history = []; // [{'phien': int, 'ket_qua': 'Tài'|'Xỉu'}]
let last_phien = null;
let isPolling = false; // Biến cờ để tránh chạy poll đồng thời

// Markov store: Map<k, Map<pattern_str, {Tai: count, Xiu: count}>>
const markovCounts = new Map();
for (let k = 1; k <= 10; k++) {
    markovCounts.set(k, new Map());
}

// =========================
// Utils
// =========================
const asTX = (resultStr) => resultStr === 'Tài' ? 'T' : 'X';
const fromTX = (ch) => ch === 'T' ? 'Tài' : 'Xỉu';

const currentStreak = (results) => {
    if (results.length === 0) {
        return {
            length: 0,
            side: null
        };
    }
    const side = results[results.length - 1];
    let length = 1;
    for (let i = results.length - 2; i >= 0; i--) {
        if (results[i] === side) {
            length++;
        } else {
            break;
        }
    }
    return {
        length,
        side
    };
};

const entropy = (p) => {
    if (p <= 0.0 || p >= 1.0) {
        return 0.0;
    }
    return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
};

// =========================
// Cập nhật Markov theo lịch sử
// =========================
const rebuildMarkov = (results) => {
    for (let k = 1; k <= 10; k++) {
        markovCounts.get(k).clear();
    }
    if (results.length < 2) {
        return;
    }
    const tx = results.map(asTX);
    for (let k = 1; k <= 10; k++) {
        if (tx.length <= k) {
            continue;
        }
        const kMap = markovCounts.get(k);
        for (let i = 0; i < tx.length - k; i++) {
            const pattern = tx.slice(i, i + k).join('');
            const nxt = tx[i + k];
            const out = fromTX(nxt);
            if (!kMap.has(pattern)) {
                kMap.set(pattern, {
                    'Tài': 0,
                    'Xỉu': 0
                });
            }
            kMap.get(pattern)[out]++;
        }
    }
};

const updateMarkovWithNew = (results) => {
    if (results.length < 2) {
        return;
    }
    const tx = results.map(asTX);
    for (let k = 1; k <= 10; k++) {
        if (tx.length > k) {
            const pattern = tx.slice(-(k + 1), -1).join('');
            const nxt = tx[tx.length - 1];
            const out = fromTX(nxt);
            const kMap = markovCounts.get(k);
            if (!kMap.has(pattern)) {
                kMap.set(pattern, {
                    'Tài': 0,
                    'Xỉu': 0
                });
            }
            kMap.get(pattern)[out]++;
        }
    }
};

// =========================
// Phân tích cầu (pattern analysis)
// =========================
const smartPatternAnalysis = (results) => {
    const labels = [];
    const vote = {
        'Tài': 0.0,
        'Xỉu': 0.0
    };
    const n = results.length;
    if (n === 0) {
        return {
            labels,
            vote
        };
    }

    // Cầu bệt
    const {
        length: streak,
        side
    } = currentStreak(results);
    if (streak >= 3) {
        labels.push(`Cầu bệt ${side} (${streak})`);
        const s = Math.min(12.0 + (streak - 3) * 2.5, 28.0);
        vote[side] += s;
    }

    // Cầu 1-1
    if (n >= 4) {
        const last4 = results.slice(-4);
        if (last4[0] !== last4[1] && last4[1] !== last4[2] && last4[2] !== last4[3]) {
            labels.push("Cầu 1-1");
            const nextSide = results[n - 1] === 'Xỉu' ? 'Tài' : 'Xỉu';
            vote[nextSide] += 18.0;
        }
    }

    // Cầu 2-2
    if (n >= 4) {
        const last4 = results.slice(-4);
        if (last4[0] === last4[1] && last4[2] === last4[3] && last4[0] !== last4[2]) {
            labels.push("Cầu 2-2");
            const prefer = last4[0] === 'Tài' ? 'Tài' : 'Xỉu';
            vote[prefer] += 12.0;
        }
    }

    // Cầu 2-1
    if (n >= 3) {
        const last3 = results.slice(-3);
        if (last3[0] === last3[1] && last3[1] !== last3[2]) {
            labels.push("Cầu 2-1");
            const prefer = last3[0];
            vote[prefer] += 10.0;
        }
    }

    // Cầu 2-3
    if (n >= 5) {
        const last5 = results.slice(-5);
        if (last5[0] === last5[1] && last5[1] !== last5[2] && last5[2] === last5[3] && last5[3] === last5[4] && last5[0] !== last5[2]) {
            labels.push("Cầu 2-3");
            const prefer = last5[2];
            vote[prefer] += 14.0;
        }
    }

    if (labels.length === 0) {
        labels.push("Không có cầu rõ ràng");
    }

    return {
        labels,
        vote
    };
};

// =========================
// Dự đoán Markov (k=1..10)
// =========================
const markovPredict = (results) => {
    if (results.length < 2) {
        return {
            probTai: 0.5,
            infoStr: "Markov: thiếu dữ liệu",
            coverSamples: 0
        };
    }

    const tx = results.map(asTX);
    let aggWeight = 0.0;
    let aggProbT = 0.0;
    let totalFollowers = 0;
    const details = [];

    for (let k = 1; k <= 10; k++) {
        if (tx.length <= k) {
            continue;
        }
        const prefix = tx.slice(-k).join('');
        const counts = markovCounts.get(k).get(prefix);
        if (!counts) {
            continue;
        }
        const cT = counts['Tài'];
        const cX = counts['Xỉu'];
        const total = cT + cX;
        if (total === 0) {
            continue;
        }
        const pT = cT / total;
        const w = k * Math.log2(1 + total);
        aggProbT += pT * w;
        aggWeight += w;
        totalFollowers += total;
        details.push(`k=${k}:${cT}/${total}T`);
    }

    if (aggWeight === 0.0) {
        return {
            probTai: 0.5,
            infoStr: "Markov: chưa khớp pattern",
            coverSamples: 0
        };
    }

    const probTai = aggProbT / aggWeight;
    const infoStr = `Markov[${details.slice(0, 4).join(', ')}${details.length > 4 ? ', ...' : ''}]`;
    return {
        probTai,
        infoStr,
        coverSamples: totalFollowers
    };
};

// =========================
// Xu hướng & Tần suất
// =========================
const localTrend = (results, lookback = 10) => {
    if (results.length === 0) {
        return {
            probTai: 0.5,
            n: 0
        };
    }
    const m = Math.min(results.length, lookback);
    const seg = results.slice(-m);
    const cT = seg.filter(r => r === 'Tài').length;
    return {
        probTai: cT / m,
        n: m
    };
};

const globalFreq = (results) => {
    if (results.length === 0) {
        return {
            probTai: 0.5,
            n: 0
        };
    }
    const cT = results.filter(r => r === 'Tài').length;
    return {
        probTai: cT / results.length,
        n: results.length
    };
};

// =========================
// Hợp nhất phiếu & tính độ tin cậy
// =========================
const combineVotes = (probMarkov, patternVote, probLocal, probGlobal, coverMarkov, nLocal, nGlobal, bridges) => {
    const sT = patternVote['Tài'] || 0.0;
    const sX = patternVote['Xỉu'] || 0.0;
    let probPattern;
    if (sT === 0.0 && sX === 0.0) {
        probPattern = 0.5;
    } else {
        const eT = Math.exp(sT / 12.0);
        const eX = Math.exp(sX / 12.0);
        probPattern = eT / (eT + eX);
    }

    const wM = 0.5 + Math.min(0.5, Math.log2(1 + coverMarkov) / 5.0);
    const wL = 0.5 + Math.min(0.5, Math.log2(1 + nLocal) / 5.0);
    const wG = 0.5 + Math.min(0.5, Math.log2(1 + nGlobal) / 5.0);

    const WM = W_MARKOV * wM;
    const WP = W_PATTERN;
    const WL = W_LOCAL_TREND * wL;
    const WG = W_GLOBAL_FREQ * wG;

    const p = (probMarkov * WM +
        probPattern * WP +
        probLocal * WL +
        probGlobal * WG) / (WM + WP + WL + WG);

    const H = entropy(p);
    let conf = (1.0 - H) * 100.0;

    const clearPatterns = bridges.filter(b => b !== "Không có cầu rõ ràng");
    if (clearPatterns.length > 0) {
        conf *= Math.min(1.15, 1.03 + 0.03 * clearPatterns.length);
    } else {
        conf *= 0.96;
    }

    conf = Math.max(CONF_MIN, Math.min(CONF_MAX, conf));
    const predict = p >= 0.5 ? 'Tài' : 'Xỉu';

    return {
        predict,
        conf,
        probPattern
    };
};

// =========================
// Hàm dự đoán chính
// =========================
const predictVIP = (results) => {
    if (results.length === 0) {
        return {
            predict: 'Tài',
            confidence: 50.0,
            explain: "Chưa có dữ liệu."
        };
    }

    const {
        labels: bridges,
        vote: patternVote
    } = smartPatternAnalysis(results);
    const {
        probTai: probMarkov,
        infoStr: markovInfo,
        coverSamples: cover
    } = markovPredict(results);
    const {
        probTai: probLocal10,
        n: nLocal
    } = localTrend(results, 10);
    const {
        probTai: probGlobal,
        n: nGlobal
    } = globalFreq(results);

    const {
        predict,
        conf,
        probPattern
    } = combineVotes(
        probMarkov, patternVote, probLocal10, probGlobal,
        cover, nLocal, nGlobal, bridges
    );

    const briefPattern = bridges.join(", ");
    const pMarkovPct = `${(probMarkov * 100).toFixed(1)}%`;
    const pPatternPct = `${(probPattern * 100).toFixed(1)}%`;
    const pLocalPct = `${(probLocal10 * 100).toFixed(1)}%`;
    const pGlobalPct = `${(probGlobal * 100).toFixed(1)}%`;

    const explain = `${briefPattern}. Markov: ${pMarkovPct} Tài (${markovInfo}). Pattern: ${pPatternPct} Tài. Gần 10: ${pLocalPct} Tài, Toàn cục: ${pGlobalPct} Tài. Chốt: ${predict}.`;

    return {
        predict,
        confidence: conf,
        explain
    };
};

// =========================
// Polling API
// =========================
const pollApi = async () => {
    if (isPolling) return;
    isPolling = true;

    try {
        const resp = await fetch(POLL_URL);
        const data = await resp.json();

        const {
            Phien: phien,
            Ket_qua: ket_qua
        } = data;

        if (phien === null || !['Tài', 'Xỉu'].includes(ket_qua)) {
            return;
        }

        if (last_phien === null) {
            history.push({
                phien,
                ket_qua
            });
            last_phien = phien;
            rebuildMarkov(history.map(h => h.ket_qua));
        } else {
            if (phien > last_phien) {
                if (history.length >= MAX_HISTORY) {
                    history.shift();
                }
                history.push({
                    phien,
                    ket_qua
                });
                last_phien = phien;
                updateMarkovWithNew(history.map(h => h.ket_qua));
            }
        }

    } catch (e) {
        console.error(`[poll_api] Error: ${e.message}`);
    } finally {
        isPolling = false;
    }
};

// Khởi động polling
setInterval(pollApi, POLL_INTERVAL_SEC * 1000);

// =========================
// API Endpoints
// =========================
app.get('/predict', async (req, res) => {
    if (history.length === 0) {
        return res.status(503).json({
            error: 'No data available yet. Waiting for API poll.'
        });
    }

    const {
        predict,
        confidence,
        explain
    } = predictVIP(history.map(h => h.ket_qua));

    let data = {};
    try {
        const resp = await fetch(POLL_URL);
        data = await resp.json();
    } catch (e) {
        // Bỏ qua lỗi, dùng dữ liệu cũ
    }

    const session = data.Phien || history[history.length - 1].phien;
    const dice = `${data.Xuc_xac_1 || ''} - ${data.Xuc_xac_2 || ''} - ${data.Xuc_xac_3 || ''}`;
    const total = data.Tong || '';
    const result = data.Ket_qua || history[history.length - 1].ket_qua;
    const nextSession = typeof session === 'number' ? session + 1 : null;
    const patternStr = history.slice(-20).map(h => asTX(h.ket_qua)).join('');

    res.json({
        session,
        dice,
        total,
        result,
        next_session: nextSession,
        predict,
        do_tin_cay: `${confidence.toFixed(1)}%`,
        giai_thich: explain,
        pattern: patternStr,
        id: 'Tele@idol_vannhat'
    });
});

app.get('/stats', (req, res) => {
    const results = history.map(h => h.ket_qua);
    const n = results.length;
    const cT = results.filter(r => r === 'Tài').length;
    const cX = n - cT;
    const {
        length: streak,
        side
    } = currentStreak(results);
    const recent10 = n >= 10 ? results.slice(-10) : results;
    const cT10 = recent10.filter(r => r === 'Tài').length;

    res.json({
        total_samples: n,
        tai_count: cT,
        xiu_count: cX,
        current_streak: streak,
        streak_side: side,
        recent10_tai: cT10,
        recent10_xiu: recent10.length - cT10
    });
});

app.get('/health', (req, res) => {
    res.json({
        ok: true
    });
});

// =========================
// Chạy app
// =========================
app.listen(port, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${port}`);
});
}

// =========================
// Utils
// =========================
const asTX = (resultStr) => resultStr === 'Tài' ? 'T' : 'X';
const fromTX = (ch) => ch === 'T' ? 'Tài' : 'Xỉu';

const currentStreak = (results) => {
    if (results.length === 0) {
        return { length: 0, side: null };
    }
    const side = results[results.length - 1];
    let length = 1;
    for (let i = results.length - 2; i >= 0; i--) {
        if (results[i] === side) {
            length++;
        } else {
            break;
        }
    }
    return { length, side };
};

const entropy = (p) => {
    if (p <= 0.0 || p >= 1.0) {
        return 0.0;
    }
    return - (p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
};

// =========================
// Cập nhật Markov theo lịch sử
// =========================
const rebuildMarkov = (results) => {
    for (let k = 1; k <= 10; k++) {
        markovCounts.get(k).clear();
    }
    if (results.length < 2) {
        return;
    }
    const tx = results.map(asTX);
    for (let k = 1; k <= 10; k++) {
        if (tx.length <= k) {
            continue;
        }
        const kMap = markovCounts.get(k);
        for (let i = 0; i < tx.length - k; i++) {
            const pattern = tx.slice(i, i + k).join('');
            const nxt = tx[i + k];
            const out = fromTX(nxt);
            if (!kMap.has(pattern)) {
                kMap.set(pattern, { 'Tài': 0, 'Xỉu': 0 });
            }
            kMap.get(pattern)[out]++;
        }
    }
};

const updateMarkovWithNew = (results) => {
    if (results.length < 2) {
        return;
    }
    const tx = results.map(asTX);
    for (let k = 1; k <= 10; k++) {
        if (tx.length > k) {
            const pattern = tx.slice(-(k + 1), -1).join('');
            const nxt = tx[tx.length - 1];
            const out = fromTX(nxt);
            const kMap = markovCounts.get(k);
            if (!kMap.has(pattern)) {
                kMap.set(pattern, { 'Tài': 0, 'Xỉu': 0 });
            }
            kMap.get(pattern)[out]++;
        }
    }
};

// =========================
// Phân tích cầu (pattern analysis)
// =========================
const smartPatternAnalysis = (results) => {
    const labels = [];
    const vote = { 'Tài': 0.0, 'Xỉu': 0.0 };
    const n = results.length;
    if (n === 0) {
        return { labels, vote };
    }

    // Cầu bệt
    const { length: streak, side } = currentStreak(results);
    if (streak >= 3) {
        labels.push(`Cầu bệt ${side} (${streak})`);
        const s = Math.min(12.0 + (streak - 3) * 2.5, 28.0);
        vote[side] += s;
    }

    // Cầu 1-1
    if (n >= 4) {
        const last4 = results.slice(-4);
        if ((last4[0] !== last4[1] && last4[1] !== last4[2] && last4[2] !== last4[3])) {
             if (last4[0] === 'Tài') { // T-X-T-X
                labels.push("Cầu 1-1");
                const nextSide = results[n - 1] === 'Xỉu' ? 'Tài' : 'Xỉu';
                vote[nextSide] += 18.0;
            }
        }
    }

    // Cầu 2-2
    if (n >= 4) {
        const last4 = results.slice(-4);
        if ((last4[0] === last4[1] && last4[2] === last4[3] && last4[0] !== last4[2])) {
            labels.push("Cầu 2-2");
            const prefer = last4[0] === 'Tài' ? 'Tài' : 'Xỉu';
            vote[prefer] += 12.0;
        }
    }

    // Cầu 2-1
    if (n >= 3) {
        const last3 = results.slice(-3);
        if ((last3[0] === last3[1] && last3[1] !== last3[2])) {
            labels.push("Cầu 2-1");
            const prefer = last3[0];
            vote[prefer] += 10.0;
        }
    }

    // Cầu 2-3
    if (n >= 5) {
        const last5 = results.slice(-5);
        if (last5[0] === last5[1] && last5[1] !== last5[2] && last5[2] === last5[3] && last5[3] === last5[4] && last5[0] !== last5[2]) {
            labels.push("Cầu 2-3");
            const prefer = last5[2];
            vote[prefer] += 14.0;
        }
    }

    if (labels.length === 0) {
        labels.push("Không có cầu rõ ràng");
    }

    return { labels, vote };
};

// =========================
// Dự đoán Markov (k=1..10)
// =========================
const markovPredict = (results) => {
    if (results.length < 2) {
        return { probTai: 0.5, infoStr: "Markov: thiếu dữ liệu", coverSamples: 0 };
    }

    const tx = results.map(asTX);
    let aggWeight = 0.0;
    let aggProbT = 0.0;
    let totalFollowers = 0;
    const details = [];

    for (let k = 1; k <= 10; k++) {
        if (tx.length <= k) {
            continue;
        }
        const prefix = tx.slice(-k).join('');
        const counts = markovCounts.get(k).get(prefix);
        if (!counts) {
            continue;
        }
        const cT = counts['Tài'];
        const cX = counts['Xỉu'];
        const total = cT + cX;
        if (total === 0) {
            continue;
        }
        const pT = cT / total;
        const w = k * Math.log2(1 + total);
        aggProbT += pT * w;
        aggWeight += w;
        totalFollowers += total;
        details.push(`k=${k}:${cT}/${total}T`);
    }

    if (aggWeight === 0.0) {
        return { probTai: 0.5, infoStr: "Markov: chưa khớp pattern", coverSamples: 0 };
    }

    const probTai = aggProbT / aggWeight;
    const infoStr = `Markov[${details.slice(0, 4).join(', ')}${details.length > 4 ? ', ...' : ''}]`;
    return { probTai, infoStr, coverSamples: totalFollowers };
};

// =========================
// Xu hướng & Tần suất
// =========================
const localTrend = (results, lookback = 10) => {
    if (results.length === 0) {
        return { probTai: 0.5, n: 0 };
    }
    const m = Math.min(results.length, lookback);
    const seg = results.slice(-m);
    const cT = seg.filter(r => r === 'Tài').length;
    return { probTai: cT / m, n: m };
};

const globalFreq = (results) => {
    if (results.length === 0) {
        return { probTai: 0.5, n: 0 };
    }
    const cT = results.filter(r => r === 'Tài').length;
    return { probTai: cT / results.length, n: results.length };
};

// =========================
// Hợp nhất phiếu & tính độ tin cậy
// =========================
const combineVotes = (probMarkov, patternVote, probLocal, probGlobal, coverMarkov, nLocal, nGlobal, bridges) => {
    const sT = patternVote['Tài'] || 0.0;
    const sX = patternVote['Xỉu'] || 0.0;
    let probPattern;
    if (sT === 0.0 && sX === 0.0) {
        probPattern = 0.5;
    } else {
        const eT = Math.exp(sT / 12.0);
        const eX = Math.exp(sX / 12.0);
        probPattern = eT / (eT + eX);
    }

    const wM = 0.5 + Math.min(0.5, Math.log(1 + coverMarkov) / Math.log(3) / 5.0);
    const wL = 0.5 + Math.min(0.5, Math.log(1 + nLocal) / Math.log(3) / 5.0);
    const wG = 0.5 + Math.min(0.5, Math.log(1 + nGlobal) / Math.log(3) / 5.0);

    const WM = W_MARKOV * wM;
    const WP = W_PATTERN;
    const WL = W_LOCAL_TREND * wL;
    const WG = W_GLOBAL_FREQ * wG;

    const p = (probMarkov * WM +
               probPattern * WP +
               probLocal * WL +
               probGlobal * WG) / (WM + WP + WL + WG);

    const H = entropy(p);
    let conf = (1.0 - H) * 100.0;

    const clearPatterns = bridges.filter(b => b !== "Không có cầu rõ ràng");
    if (clearPatterns.length > 0) {
        conf *= Math.min(1.15, 1.03 + 0.03 * clearPatterns.length);
    } else {
        conf *= 0.96;
    }

    conf = Math.max(CONF_MIN, Math.min(CONF_MAX, conf));
    const predict = p >= 0.5 ? 'Tài' : 'Xỉu';

    return { predict, conf, probPattern };
};

// =========================
// Hàm dự đoán chính
// =========================
const predictVIP = (results) => {
    if (results.length === 0) {
        return { predict: 'Tài', confidence: 50.0, explain: "Chưa có dữ liệu." };
    }

    const { labels: bridges, vote: patternVote } = smartPatternAnalysis(results);
    const { probTai: probMarkov, infoStr: markovInfo, coverSamples: cover } = markovPredict(results);
    const { probTai: probLocal10, n: nLocal } = localTrend(results, 10);
    const { probTai: probGlobal, n: nGlobal } = globalFreq(results);

    const { predict, conf, probPattern } = combineVotes(
        probMarkov, patternVote, probLocal10, probGlobal,
        cover, nLocal, nGlobal, bridges
    );

    const briefPattern = bridges.join(", ");
    const pMarkovPct = `${(probMarkov * 100).toFixed(1)}%`;
    const pPatternPct = `${(probPattern * 100).toFixed(1)}%`;
    const pLocalPct = `${(probLocal10 * 100).toFixed(1)}%`;
    const pGlobalPct = `${(probGlobal * 100).toFixed(1)}%`;

    const explain = `${briefPattern}. Markov: ${pMarkovPct} Tài (${markovInfo}). Pattern: ${pPatternPct} Tài. Gần 10: ${pLocalPct} Tài, Toàn cục: ${pGlobalPct} Tài. Chốt: ${predict}.`;

    return { predict, confidence: conf, explain };
};

// =========================
// Polling API
// =========================
const pollApi = async () => {
    try {
        const resp = await fetch(POLL_URL, { timeout: 10000 });
        const data = await resp.json();

        const { Phien: phien, Ket_qua: ket_qua } = data;

        if (phien === null || !['Tài', 'Xỉu'].includes(ket_qua)) {
            return;
        }

        if (lock.release()) {
            await lock.acquire();
        } else {
            lock.acquire();
        }
        
        if (last_phien === null) {
            history.push({ phien, ket_qua });
            last_phien = phien;
            rebuildMarkov(history.map(h => h.ket_qua));
        } else {
            if (phien > last_phien) {
                if (history.length >= MAX_HISTORY) {
                    history.shift();
                }
                history.push({ phien, ket_qua });
                last_phien = phien;
                updateMarkovWithNew(history.map(h => h.ket_qua));
            }
        }
        lock.release();

    } catch (e) {
        console.error(`[poll_api] Error: ${e.message}`);
    }
};

// Khởi động polling
setInterval(pollApi, POLL_INTERVAL_SEC * 1000);

// =========================
// API Endpoints
// =========================
app.get('/predict', async (req, res) => {
    if (history.length === 0) {
        return res.status(503).json({ error: 'No data available yet. Waiting for API poll.' });
    }

    const { predict, confidence, explain } = predictVIP(history.map(h => h.ket_qua));

    let data = {};
    try {
        const resp = await fetch(POLL_URL, { timeout: 8000 });
        data = await resp.json();
    } catch (e) {
        // Bỏ qua lỗi, dùng dữ liệu cũ
    }

    const session = data.Phien || history[history.length - 1].phien;
    const dice = `${data.Xuc_xac_1 || ''} - ${data.Xuc_xac_2 || ''} - ${data.Xuc_xac_3 || ''}`;
    const total = data.Tong || '';
    const result = data.Ket_qua || history[history.length - 1].ket_qua;
    const nextSession = typeof session === 'number' ? session + 1 : null;
    const patternStr = history.slice(-20).map(h => asTX(h.ket_qua)).join('');

    res.json({
        session,
        dice,
        total,
        result,
        next_session: nextSession,
        predict,
        do_tin_cay: `${confidence.toFixed(1)}%`,
        giai_thich: explain,
        pattern: patternStr,
        id: 'Tele@idol_vannhat'
    });
});

app.get('/stats', (req, res) => {
    const results = history.map(h => h.ket_qua);
    const n = results.length;
    const cT = results.filter(r => r === 'Tài').length;
    const cX = n - cT;
    const { length: streak, side } = currentStreak(results);
    const recent10 = n >= 10 ? results.slice(-10) : results;
    const cT10 = recent10.filter(r => r === 'Tài').length;

    res.json({
        total_samples: n,
        tai_count: cT,
        xiu_count: cX,
        current_streak: streak,
        streak_side: side,
        recent10_tai: cT10,
        recent10_xiu: recent10.length - cT10
    });
});

app.get('/health', (req, res) => {
    res.json({ ok: true });
});

// =========================
// Chạy app
// =========================
app.listen(port, '0.0.0.0', () => {
    console.log(`Server is running at http://0.0.0.0:${port}`);
});
