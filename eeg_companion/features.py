"""從一個視窗算出「想睡指數」。

腦波：想睡時 theta(4-8Hz)、alpha(8-12Hz) 上升，beta(15-30Hz) 下降。
      指數 D = (相對theta + 相對alpha) / 相對beta，越大越想睡。
EOG ：想睡初期會出現緩慢眼球轉動(SEM, 0.1-0.5Hz)，低頻能量上升是入睡前兆。
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (15, 30)}


def _bandpowers(x, sfreq):
    x = x - np.mean(x)
    nper = min(len(x), int(2 * sfreq))
    freqs, psd = welch(x, sfreq, nperseg=nper)
    out = {}
    for name, (lo, hi) in BANDS.items():
        out[name] = float(np.trapezoid(psd[(freqs >= lo) & (freqs < hi)],
                                       freqs[(freqs >= lo) & (freqs < hi)]))
    total = sum(out.values()) + 1e-12
    return {k: v / total for k, v in out.items()}


def _eog_slow_ratio(x, sfreq):
    x = x - np.mean(x)
    nper = min(len(x), int(2 * sfreq))
    freqs, psd = welch(x, sfreq, nperseg=nper)
    slow = np.trapezoid(psd[(freqs >= 0.1) & (freqs < 0.5)], freqs[(freqs >= 0.1) & (freqs < 0.5)])
    total = np.trapezoid(psd[(freqs >= 0.1) & (freqs < 15)], freqs[(freqs >= 0.1) & (freqs < 15)]) + 1e-12
    return float(slow / total)


def compute(win):
    """回傳 dict：想睡指數 D、各頻帶相對能量、EOG 慢波比例。"""
    bp = _bandpowers(win["eeg"], win["sfreq"])
    D = (bp["theta"] + bp["alpha"]) / (bp["beta"] + 1e-6)
    sem = _eog_slow_ratio(win["eog"], win["sfreq"]) if win.get("eog") is not None else None
    return dict(D=float(D), sem=sem, t=win["t"], **{f"rel_{k}": v for k, v in bp.items()})


# ---- 多模態：心率(ECG) 與三軸姿勢 ----
from scipy.signal import butter, filtfilt, find_peaks


def heart_rate(ecg, sf):
    """從 ECG 抓 R 峰算心率(BPM)。訊號太短或抓不到峰回傳 None。"""
    if ecg is None or len(ecg) < int(3 * sf):
        return None
    x = np.asarray(ecg, float); x = x - np.mean(x)
    if np.std(x) < 1e-6:
        return None
    b, a = butter(2, [5/(sf/2), 15/(sf/2)], btype="band")
    y = filtfilt(b, a, x) ** 2
    thr = np.median(y) + 4 * np.median(np.abs(y - np.median(y))) / 0.6745
    pk, _ = find_peaks(y, height=max(thr, 1e-9), distance=int(0.3 * sf))
    if len(pk) < 3:
        return None
    rr = np.diff(pk) / sf
    rr = rr[(rr > 0.3) & (rr < 1.5)]
    return float(60.0 / np.median(rr)) if len(rr) else None


def accel_features(x, y, z):
    """三軸(單位 g)→ 重力向量、活動量(AC RMS)、瞬時抖動(jerk)。"""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    gx, gy, gz = float(np.mean(x)), float(np.mean(y)), float(np.mean(z))
    activity = float(np.sqrt(np.mean((x-gx)**2 + (y-gy)**2 + (z-gz)**2)))
    jerk = float(np.max(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))) if len(x) > 1 else 0.0
    return dict(g=(gx, gy, gz), activity=activity, jerk=jerk)


def compute_mm(win):
    """在 compute() 之上補心率與三軸。回傳含 D/hr/g/activity/jerk 的 dict。"""
    out = compute(win)
    ecg = win.get("ecg"); sf = win.get("ecg_sf", win["sfreq"])
    out["hr"] = heart_rate(ecg, sf) if ecg is not None else None
    acc = win.get("acc")
    if acc is not None:
        af = accel_features(acc["x"], acc["y"], acc["z"])
        out["g"] = af["g"]; out["activity"] = af["activity"]; out["jerk"] = af["jerk"]
    else:
        out["g"] = out["activity"] = out["jerk"] = None
    return out
