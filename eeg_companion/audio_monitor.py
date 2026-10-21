"""睡眠音訊監測 — 偵測打呼、夢囈、RBD 大叫。

用麥克風持續錄音，分析短時窗 (1 秒) 的聲學特徵，分類為:
  SILENCE / SNORE / TALK / SCREAM

特徵:
  - RMS 能量 (dB)
  - 頻譜質心 (spectral centroid)
  - 週期性 (autocorrelation，偵測打呼的規律呼吸模式)
  - 起始強度 (onset strength，偵測突發大叫)
"""
from __future__ import annotations
import time
import threading
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from collections import deque


class AudioEvent(str, Enum):
    SILENCE = "SILENCE"
    SNORE = "SNORE"
    TALK = "TALK"
    SCREAM = "SCREAM"


@dataclass
class AudioFrame:
    timestamp: float
    event: AudioEvent
    rms_db: float
    spectral_centroid: float
    periodicity: float
    onset: float
    duration: float = 0.0


class AudioMonitor:
    """持續監聽音訊，分類睡眠事件。支援本機麥克風或 RTSP 串流（Tapo 攝影機）。"""

    def __init__(self, samplerate: int = 16000, block_sec: float = 1.0,
                 device=None, rtsp_url: str | None = None,
                 quiet_db: float = -45.0,
                 snore_db: float = -35.0,
                 talk_db: float = -30.0,
                 scream_db: float = -20.0):
        self.sr = samplerate
        self.block_size = int(samplerate * block_sec)
        self.device = device
        self.rtsp_url = rtsp_url

        self.quiet_db = quiet_db
        self.snore_db = snore_db
        self.talk_db = talk_db
        self.scream_db = scream_db

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._current: AudioFrame = AudioFrame(
            timestamp=time.time(), event=AudioEvent.SILENCE,
            rms_db=-60, spectral_centroid=0, periodicity=0, onset=0,
        )
        self._history: deque[AudioFrame] = deque(maxlen=3600)
        self._prev_rms = 0.0

        self._events: list[dict] = []
        self._active_event: dict | None = None

        self._callbacks: list = []
        self._calibrating = True
        self._cal_samples: list[float] = []
        self._cal_duration = 30  # 前 30 秒校準環境底噪

    def on_event(self, cb):
        self._callbacks.append(cb)

    @property
    def current(self) -> AudioFrame:
        with self._lock:
            return self._current

    @property
    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def summary(self) -> dict:
        with self._lock:
            counts = {e.value: 0 for e in AudioEvent}
            total_dur = {e.value: 0.0 for e in AudioEvent}
            for f in self._history:
                counts[f.event.value] += 1
                total_dur[f.event.value] += 1.0

            evt_summary = []
            for e in self._events:
                evt_summary.append({
                    "type": e["type"],
                    "start": e["start_time"],
                    "duration": e["duration"],
                })
            return {
                "counts": counts,
                "durations_sec": total_dur,
                "events": evt_summary,
                "snore_episodes": sum(1 for e in self._events if e["type"] == "SNORE"),
                "talk_episodes": sum(1 for e in self._events if e["type"] == "TALK"),
                "scream_episodes": sum(1 for e in self._events if e["type"] == "SCREAM"),
            }

    def start(self):
        if self._running:
            return
        self._running = True
        if self.rtsp_url:
            # 攝影機遠場麥克風：降低閾值，補償距離衰減
            self.quiet_db -= 10
            self.snore_db -= 10
            self.talk_db -= 10
            self.scream_db -= 10
            self._thread = threading.Thread(target=self._rtsp_loop, daemon=True)
            self._thread.start()
            print(f"[AudioMonitor] 開始監聽 RTSP 音訊 (sr={self.sr}, 遠場模式)")
        else:
            try:
                import sounddevice  # noqa: F401
            except ImportError:
                raise ImportError("pip install sounddevice")
            self._thread = threading.Thread(target=self._mic_loop, daemon=True)
            self._thread.start()
            print(f"[AudioMonitor] 開始監聽麥克風 (sr={self.sr}, device={self.device})")

    def stop(self):
        self._running = False
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.terminate()
            except Exception:
                pass
            self._ffmpeg_proc = None
        if self._thread:
            self._thread.join(timeout=5)
        self._close_active_event()

    _ffmpeg_proc = None

    def _rtsp_loop(self):
        """從 RTSP 串流用 ffmpeg 擷取音訊。"""
        import subprocess, struct
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sr),
            "-ac", "1",
            "-f", "s16le",
            "-",
        ]
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("[AudioMonitor] ffmpeg 未安裝，無法擷取 RTSP 音訊")
            return

        bytes_per_block = self.block_size * 2  # 16-bit = 2 bytes
        while self._running:
            raw = self._ffmpeg_proc.stdout.read(bytes_per_block)
            if not raw or len(raw) < bytes_per_block:
                if not self._running:
                    break
                time.sleep(0.5)
                continue
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
            self._process(samples)

    def _mic_loop(self):
        import sounddevice as sd
        with sd.InputStream(samplerate=self.sr, channels=1,
                            blocksize=self.block_size,
                            device=self.device) as stream:
            while self._running:
                data, overflowed = stream.read(self.block_size)
                if overflowed:
                    continue
                audio = data[:, 0].astype(np.float64)
                self._process(audio)

    def _process(self, audio: np.ndarray):
        now = time.time()

        rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
        rms_db = 20 * np.log10(rms)

        # 自動校準：前 N 秒學習環境底噪
        if self._calibrating:
            self._cal_samples.append(rms_db)
            if len(self._cal_samples) >= self._cal_duration:
                floor = np.mean(self._cal_samples) + 2 * np.std(self._cal_samples)
                self.quiet_db = max(self.quiet_db, floor + 3)
                self.snore_db = max(self.snore_db, floor + 8)
                self.talk_db = max(self.talk_db, floor + 12)
                self.scream_db = max(self.scream_db, floor + 22)
                self._calibrating = False
                print(f"[AudioMonitor] 校準完成: 底噪={np.mean(self._cal_samples):.1f}dB, "
                      f"quiet>{self.quiet_db:.0f} snore>{self.snore_db:.0f} "
                      f"talk>{self.talk_db:.0f} scream>{self.scream_db:.0f}")
            return

        centroid = self._spectral_centroid(audio)
        periodicity = self._periodicity(audio)

        onset = max(0, rms_db - self._prev_rms)
        self._prev_rms = rms_db

        event = self._classify(rms_db, centroid, periodicity, onset)

        frame = AudioFrame(
            timestamp=now, event=event,
            rms_db=round(rms_db, 1),
            spectral_centroid=round(centroid, 0),
            periodicity=round(periodicity, 3),
            onset=round(onset, 1),
        )

        with self._lock:
            self._current = frame
            self._history.append(frame)
            self._update_events(frame)

        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception:
                pass

    def _spectral_centroid(self, audio: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sr)
        total = spectrum.sum()
        if total < 1e-10:
            return 0.0
        return float(np.sum(freqs * spectrum) / total)

    def _periodicity(self, audio: np.ndarray) -> float:
        """自相關週期性，高值表示規律重複模式（如打呼）。"""
        n = len(audio)
        if n < 100:
            return 0.0
        audio = audio - np.mean(audio)
        norm = np.sum(audio ** 2)
        if norm < 1e-10:
            return 0.0
        corr = np.correlate(audio, audio, mode='full')[n-1:]
        corr = corr / norm
        # 搜尋 0.2~1.5 秒的週期（打呼頻率 ~0.7-5 Hz）
        lo = int(0.2 * self.sr)
        hi = min(int(1.5 * self.sr), len(corr) - 1)
        if lo >= hi:
            return 0.0
        peak = float(np.max(corr[lo:hi]))
        return max(0.0, peak)

    def _classify(self, rms_db: float, centroid: float,
                  periodicity: float, onset: float) -> AudioEvent:
        if rms_db < self.quiet_db:
            return AudioEvent.SILENCE

        if rms_db > self.scream_db and onset > 15 and centroid > 800:
            return AudioEvent.SCREAM

        if rms_db > self.snore_db and periodicity > 0.3 and centroid < 600:
            return AudioEvent.SNORE

        if rms_db > self.talk_db and 200 < centroid < 3500 and periodicity < 0.25:
            return AudioEvent.TALK

        if rms_db > self.snore_db and centroid < 400:
            return AudioEvent.SNORE

        if rms_db > self.talk_db:
            return AudioEvent.TALK

        return AudioEvent.SILENCE

    def _update_events(self, frame: AudioFrame):
        if frame.event == AudioEvent.SILENCE:
            self._close_active_event()
            return

        if self._active_event is None:
            self._active_event = {
                "type": frame.event.value,
                "start_time": time.strftime("%H:%M:%S"),
                "start_ts": frame.timestamp,
                "duration": 1.0,
                "peak_db": frame.rms_db,
            }
        elif self._active_event["type"] == frame.event.value:
            self._active_event["duration"] = frame.timestamp - self._active_event["start_ts"]
            self._active_event["peak_db"] = max(
                self._active_event["peak_db"], frame.rms_db)
        else:
            self._close_active_event()
            self._active_event = {
                "type": frame.event.value,
                "start_time": time.strftime("%H:%M:%S"),
                "start_ts": frame.timestamp,
                "duration": 1.0,
                "peak_db": frame.rms_db,
            }

    def _close_active_event(self):
        if self._active_event is not None:
            if self._active_event["duration"] >= 2.0:
                evt = dict(self._active_event)
                evt.pop("start_ts", None)
                self._events.append(evt)
            self._active_event = None
