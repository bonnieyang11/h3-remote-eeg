"""RBD (REM 睡眠行為障礙) 偵測演算法。

核心邏輯：
  1. 簡易睡眠分期 — 用 H3 四通道 EEG + EOG 判斷 Wake/NREM/REM
  2. 動作偵測     — 加速度 + 影像雙通道融合
  3. RBD 事件標記  — REM 期間出現顯著動作 = 疑似 RBD 發作

H3 通道對應：
  EEG1=Fp1−右耳前, EEG2=Fp2−左耳前, EEG3=ECG, EEG4=ROC−A1(EOG)
  只用 EEG1(或 EEG2) + EOG 做分期，ECG 做心率輔助。
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch, butter, filtfilt
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class SleepStage(str, Enum):
    WAKE = "WAKE"
    NREM = "NREM"
    REM = "REM"
    UNKNOWN = "UNKNOWN"


@dataclass
class RBDEvent:
    t: float
    stage: SleepStage
    accel_activity: float | None
    video_motion: float | None
    combined_motion: float
    confidence: float
    detail: str = ""


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30),
}


def _relative_bandpower(x: np.ndarray, sfreq: float) -> dict[str, float]:
    x = x - np.mean(x)
    nper = min(len(x), int(2 * sfreq))
    freqs, psd = welch(x, sfreq, nperseg=nper)
    bp = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp[name] = float(np.trapezoid(psd[mask], freqs[mask]))
    total = sum(bp.values()) + 1e-12
    return {k: v / total for k, v in bp.items()}


def _muscle_tone(eog: np.ndarray, eeg: np.ndarray,
                  sfreq: float) -> float:
    """從 EOG + 額葉 EEG 高頻成分估算肌張力（chin EMG 代理指標）。

    EOG (ROC-A1) 參考點在乳突，靠近顳肌/咬肌，>10 Hz 成分主要是肌電。
    額葉 EEG (Fp1/Fp2) 的 >20 Hz 成分也包含面部肌電。
    REM 正常失張力 → 此值低；RSWA (REM without atonia) → 此值高。
    """
    tone = 0.0
    n = 0

    # EOG 高頻 (>10 Hz): 肌電主要貢獻
    if len(eog) > int(sfreq * 0.2):
        try:
            b, a = butter(4, 10.0 / (sfreq / 2), btype='high')
            eog_hf = filtfilt(b, a, eog - np.mean(eog))
            tone += float(np.sqrt(np.mean(eog_hf ** 2)))
            n += 1
        except Exception:
            pass

    # EEG 高頻 (>20 Hz): 額肌活動
    if len(eeg) > int(sfreq * 0.2):
        try:
            b, a = butter(4, 20.0 / (sfreq / 2), btype='high')
            eeg_hf = filtfilt(b, a, eeg - np.mean(eeg))
            tone += float(np.sqrt(np.mean(eeg_hf ** 2))) * 0.5
            n += 1
        except Exception:
            pass

    return tone / max(n, 1)


def _eog_rem_index(eog: np.ndarray, sfreq: float) -> float:
    """EOG 快速眼動指標：高頻(0.5-2Hz)能量 / 總能量。
    REM 時有快速眼動 burst → 此指標高；SEM(慢眼動)集中在 <0.5Hz → 此指標低。
    """
    eog = eog - np.mean(eog)
    nper = min(len(eog), int(2 * sfreq))
    freqs, psd = welch(eog, sfreq, nperseg=nper)
    rapid = np.trapezoid(psd[(freqs >= 0.5) & (freqs < 2.0)],
                         freqs[(freqs >= 0.5) & (freqs < 2.0)])
    total = np.trapezoid(psd[(freqs >= 0.1) & (freqs < 10)],
                         freqs[(freqs >= 0.1) & (freqs < 10)]) + 1e-12
    return float(rapid / total)


class SleepStager:
    """簡易即時睡眠分期器（30 秒 epoch 或更短的滑動窗口）。

    規則（針對額葉 EEG + 單通道 EOG）：
      - Wake: 高 alpha(閉眼) 或高 beta(開眼)，低 delta
      - NREM: 高 delta（N3）或有 sigma 活動（N2 紡錘波），低 REM 指標
      - REM: 低 delta、低 alpha、theta 為主、EOG 快速眼動指標高
    """
    def __init__(self, baseline_epochs: int = 10):
        self.baseline_epochs = baseline_epochs
        self._history: list[dict] = []
        self._baselines: dict | None = None

    def _compute_baselines(self):
        if len(self._history) < self.baseline_epochs:
            return
        keys = self._history[0].keys()
        self._baselines = {}
        for k in keys:
            vals = [h[k] for h in self._history[:self.baseline_epochs] if h[k] is not None]
            if vals:
                self._baselines[k] = {"mu": np.mean(vals), "sd": max(np.std(vals), 1e-6)}

    def classify(self, eeg: np.ndarray, eog: np.ndarray | None,
                 sfreq: float) -> tuple[SleepStage, dict]:
        bp = _relative_bandpower(eeg, sfreq)
        rem_idx = _eog_rem_index(eog, sfreq) if eog is not None else None
        tone = _muscle_tone(eog if eog is not None else eeg, eeg, sfreq)

        feat = {**{f"rel_{k}": v for k, v in bp.items()},
                "rem_idx": rem_idx, "muscle_tone": tone}
        self._history.append(feat)
        if self._baselines is None:
            self._compute_baselines()
            return SleepStage.UNKNOWN, feat

        delta, theta, alpha = bp["delta"], bp["theta"], bp["alpha"]
        sigma, beta = bp["sigma"], bp["beta"]

        # Wake: alpha + beta dominant
        if alpha + beta > 0.45:
            return SleepStage.WAKE, feat

        # REM: low delta, theta dominant, high REM eye movement index
        rem_likely = (delta < 0.35 and theta > 0.20
                      and (rem_idx is not None and rem_idx > 0.35))
        if rem_likely:
            return SleepStage.REM, feat

        # NREM: high delta (N3) or sigma bump (N2)
        if delta > 0.40 or sigma > 0.10:
            return SleepStage.NREM, feat

        # Fallback: light sleep → NREM
        if delta + theta > alpha + beta:
            return SleepStage.NREM, feat

        return SleepStage.WAKE, feat


class MotionDetector:
    """融合加速度 + 影像動作偵測。"""
    def __init__(self, accel_baseline_sec: float = 60.0,
                 accel_threshold: float = 3.0,
                 video_threshold: float = 0.02):
        self.accel_threshold = accel_threshold
        self.video_threshold = video_threshold
        self._accel_history: deque[float] = deque(maxlen=300)
        self._accel_mu: float | None = None
        self._accel_sd: float | None = None
        self._baseline_n = int(accel_baseline_sec)

    def update_accel(self, activity: float | None, jerk: float | None):
        if activity is None:
            return 0.0
        self._accel_history.append(activity)
        if len(self._accel_history) >= self._baseline_n and self._accel_mu is None:
            vals = list(self._accel_history)
            self._accel_mu = np.mean(vals)
            self._accel_sd = max(np.std(vals), 1e-6)
        if self._accel_mu is not None:
            z = (activity - self._accel_mu) / self._accel_sd
            return max(0.0, float(z))
        return 0.0

    def score(self, accel_activity: float | None = None,
              accel_jerk: float | None = None,
              video_motion: float | None = None) -> tuple[float, dict]:
        """回傳 (combined_score, details)。combined_score > 1.0 = 顯著動作。"""
        az = self.update_accel(accel_activity, accel_jerk)

        vz = 0.0
        if video_motion is not None and video_motion > self.video_threshold:
            vz = video_motion / self.video_threshold

        # 加速度與影像取較大者，有兩個來源一致時加成
        if az > 0 and vz > 0:
            combined = max(az, vz) * 1.3
        else:
            combined = max(az, vz)

        return combined, {"accel_z": az, "video_z": vz,
                          "accel_raw": accel_activity, "video_raw": video_motion}


class RBDDetector:
    """RBD 偵測主類別：睡眠分期 + 動作偵測 → RBD 事件。

    用法:
        det = RBDDetector()
        for win in source.windows():
            event = det.process(win, video_motion=cam.motion_score())
            if event:
                print(f"[RBD] t={event.t:.0f}s conf={event.confidence:.2f}")
    """
    def __init__(self, motion_threshold: float = 1.5,
                 min_rem_epochs: int = 3,
                 cooldown_sec: float = 30.0):
        self.stager = SleepStager()
        self.motion = MotionDetector()
        self.motion_threshold = motion_threshold
        self.min_rem_epochs = min_rem_epochs
        self.cooldown_sec = cooldown_sec
        self._rem_count = 0
        self._last_event_t = -1e9
        self.events: list[RBDEvent] = []
        self.stage_log: list[tuple[float, SleepStage]] = []
        self._tone_nrem: deque[float] = deque(maxlen=300)
        self._tone_rem: deque[float] = deque(maxlen=300)

    def process(self, win: dict, video_motion: float | None = None) -> RBDEvent | None:
        eeg = win["eeg"]
        eog = win.get("eog")
        sfreq = win["sfreq"]
        t = win["t"]

        stage, feat = self.stager.classify(eeg, eog, sfreq)
        self.stage_log.append((t, stage))

        # 收集各期肌張力，用於 RSWA 判定
        tone = feat.get("muscle_tone", 0)
        if stage == SleepStage.NREM:
            self._tone_nrem.append(tone)
        if stage == SleepStage.REM:
            self._rem_count += 1
            self._tone_rem.append(tone)
        else:
            self._rem_count = 0

        # RSWA: REM 期肌張力相對於 NREM 異常升高
        rswa = False
        rswa_ratio = None
        if (len(self._tone_nrem) >= 30 and len(self._tone_rem) >= 5
                and stage == SleepStage.REM):
            nrem_mean = np.mean(self._tone_nrem)
            if nrem_mean > 1e-6:
                rswa_ratio = tone / nrem_mean
                rswa = rswa_ratio > 1.5

        accel_activity = win.get("activity")
        accel_jerk = win.get("jerk")
        combined, motion_detail = self.motion.score(
            accel_activity, accel_jerk, video_motion)

        # RSWA 作為額外的 RBD 證據，提升信心度
        if rswa:
            combined *= 1.3

        in_rem = (stage == SleepStage.REM and
                  self._rem_count >= self.min_rem_epochs)
        significant_motion = combined > self.motion_threshold
        not_cooling = (t - self._last_event_t) > self.cooldown_sec

        if in_rem and significant_motion and not_cooling:
            confidence = min(1.0, combined / (self.motion_threshold * 2))
            if rswa:
                confidence = min(1.0, confidence * 1.2)
            sources = []
            if motion_detail["accel_z"] > 1.0:
                sources.append("加速度")
            if motion_detail["video_z"] > 1.0:
                sources.append("影像")
            if rswa:
                sources.append(f"RSWA×{rswa_ratio:.1f}")
            event = RBDEvent(
                t=t,
                stage=stage,
                accel_activity=accel_activity,
                video_motion=video_motion,
                combined_motion=combined,
                confidence=confidence,
                detail=f"REM中偵測到動作 ({'+'.join(sources) or '?'})",
            )
            self.events.append(event)
            self._last_event_t = t
            return event

        return None

    def summary(self) -> dict:
        """回傳本次監測的摘要。"""
        total_epochs = len(self.stage_log)
        stage_counts = {}
        for _, s in self.stage_log:
            stage_counts[s.value] = stage_counts.get(s.value, 0) + 1

        tone_info = {}
        if self._tone_nrem:
            tone_info["nrem_mean"] = float(np.mean(self._tone_nrem))
        if self._tone_rem:
            tone_info["rem_mean"] = float(np.mean(self._tone_rem))
        if tone_info.get("nrem_mean") and tone_info.get("rem_mean"):
            tone_info["rem_nrem_ratio"] = (
                tone_info["rem_mean"] / max(tone_info["nrem_mean"], 1e-6))

        return {
            "total_epochs": total_epochs,
            "stage_distribution": {
                k: f"{v/total_epochs*100:.1f}%" for k, v in stage_counts.items()
            } if total_epochs > 0 else {},
            "rbd_events": len(self.events),
            "events": [
                {"t": e.t, "confidence": e.confidence, "detail": e.detail}
                for e in self.events
            ],
            "muscle_tone": tone_info,
        }
