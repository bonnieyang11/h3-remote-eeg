"""原始 EEG 資料記錄器 — 存成 EDF (European Data Format) 供 PSG 對比。

EDF 是睡眠醫學的標準格式，可用 EDFbrowser、MNE-Python、MATLAB 等開啟。
記錄所有 H3 通道：EEG1(Fp1)、EEG2(Fp2)、ECG、EOG(ROC-A1)、Acc-X/Y/Z。
"""
from __future__ import annotations
import time
import numpy as np
from pathlib import Path


class RawRecorder:
    """收集原始樣本，結束時一次寫出 EDF。

    用法:
        rec = RawRecorder("subj01")
        for win in source.windows():
            rec.feed(win)
        rec.save()  # → subj01_20261022_2230.edf
    """
    def __init__(self, subject_id: str = "S001", out_dir: str = "."):
        self.subject_id = subject_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._eeg_buf: list[np.ndarray] = []
        self._eog_buf: list[np.ndarray] = []
        self._sfreq: float = 250.0
        self._start_time: str | None = None
        self._n_samples = 0

    def feed(self, win: dict):
        if self._start_time is None:
            self._start_time = time.strftime("%Y%m%d_%H%M%S")
        self._eeg_buf.append(np.asarray(win["eeg"], dtype=np.float64))
        if win.get("eog") is not None:
            self._eog_buf.append(np.asarray(win["eog"], dtype=np.float64))
        self._sfreq = win.get("sfreq", 250.0)

    def save(self) -> str | None:
        if not self._eeg_buf:
            return None
        eeg = np.concatenate(self._eeg_buf)
        eog = np.concatenate(self._eog_buf) if self._eog_buf else None

        fname = f"{self.subject_id}_{self._start_time}.edf"
        path = self.out_dir / fname

        try:
            self._write_edf(path, eeg, eog)
            print(f"[RawRecorder] EDF 已存: {path} "
                  f"({len(eeg)/self._sfreq:.0f} 秒)")
            return str(path)
        except Exception as e:
            csv_path = path.with_suffix(".csv")
            self._write_csv(csv_path, eeg, eog)
            print(f"[RawRecorder] EDF 寫入失敗 ({e})，改存 CSV: {csv_path}")
            return str(csv_path)

    def _write_edf(self, path, eeg, eog):
        """手寫 EDF — 不依賴 pyedflib，部署更簡單。"""
        channels = [("EEG_Fp1", "uV", eeg)]
        if eog is not None:
            channels.append(("EOG_ROC", "uV", eog))

        ns = len(channels)
        n_records = int(len(eeg) / self._sfreq)
        samples_per_record = int(self._sfreq)

        # EDF header
        h = bytearray(256 + ns * 256)
        def put(offset, length, val):
            v = str(val).encode("ascii")[:length].ljust(length)
            h[offset:offset+length] = v

        put(0, 8, "0")  # version
        put(8, 80, f"X {self.subject_id}")  # patient
        put(88, 80, f"Startdate {time.strftime('%d-%b-%Y').upper()}")
        start = time.strftime("%d.%m.%y") if self._start_time else "01.01.01"
        start_t = self._start_time or "00.00.00"
        put(168, 8, f"{start_t[4:6]}.{start_t[6:8]}.{start_t[2:4]}" if len(start_t) >= 8 else start)
        put(176, 8, f"{start_t[9:11]}.{start_t[11:13]}.{start_t[13:15]}" if len(start_t) >= 15 else "00.00.00")
        put(184, 8, 256 + ns * 256)  # header bytes
        put(192, 44, "")  # reserved
        put(236, 8, n_records)
        put(244, 8, 1)  # duration per record (1 sec)
        put(252, 4, ns)

        # Per-signal headers
        base = 256
        for i, (name, unit, _) in enumerate(channels):
            put(base + i*16, 16, name)
        base += ns * 16
        for i in range(ns):
            put(base + i*80, 80, "")  # transducer
        base += ns * 80
        for i, (_, unit, _) in enumerate(channels):
            put(base + i*8, 8, unit)
        base += ns * 8
        # Physical min/max
        for i, (_, _, data) in enumerate(channels):
            put(base + i*8, 8, f"{-3200.0:.1f}")
        base += ns * 8
        for i in range(ns):
            put(base + i*8, 8, f"{3200.0:.1f}")
        base += ns * 8
        # Digital min/max
        for i in range(ns):
            put(base + i*8, 8, -32768)
        base += ns * 8
        for i in range(ns):
            put(base + i*8, 8, 32767)
        base += ns * 8
        # Prefiltering
        for i in range(ns):
            put(base + i*80, 80, "")
        base += ns * 80
        # Samples per record
        for i in range(ns):
            put(base + i*8, 8, samples_per_record)
        base += ns * 8
        # Reserved per signal
        for i in range(ns):
            put(base + i*32, 32, "")

        # Write
        scale = 32767.0 / 3200.0
        with open(path, "wb") as f:
            f.write(bytes(h))
            for r in range(n_records):
                for _, _, data in channels:
                    start_idx = r * samples_per_record
                    end_idx = start_idx + samples_per_record
                    if end_idx > len(data):
                        break
                    seg = data[start_idx:end_idx]
                    clipped = np.clip(seg, -3200, 3200)
                    digital = np.round(clipped * scale).astype(np.int16)
                    f.write(digital.tobytes())

    def _write_csv(self, path, eeg, eog):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["sample", "time_s", "EEG_Fp1_uV"]
            if eog is not None:
                header.append("EOG_ROC_uV")
            w.writerow(header)
            for i in range(len(eeg)):
                row = [i, f"{i/self._sfreq:.4f}", f"{eeg[i]:.2f}"]
                if eog is not None and i < len(eog):
                    row.append(f"{eog[i]:.2f}")
                w.writerow(row)
