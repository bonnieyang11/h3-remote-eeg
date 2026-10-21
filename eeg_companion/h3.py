"""H3 腦波儀即時來源。

移植自 sleepwellH3-rec 的 Rust 解析器 (app/src-tauri/src/live/parser.rs)。
串流(剝掉 BLE 6-byte 標頭後)是 CR 結尾的 ASCII 狀態行(RT=/BAT=/XYZ=...)
與二進位波形區塊交錯。區塊由 `x=CCCCSS\\r` 宣告，接著:
  [N-1][M-1][每通道(div-1)的nibble,每byte2個] + total_words 個 16-bit big-endian
  offset-binary 樣本(通道交錯,通道c在 s%div[c]==0 時才有樣本) + 16-bit 資料位元組和。
H3: N=30, M=7, div=[1,1,1,1,5,5,5] → EEG1-4@250Hz, X/Y/Z@50Hz。
EEG 縮放: uV = (w-32768) * 25208.3/32768。通道對應你的蒙太奇:
  EEG1=Fp1−右耳前, EEG2=Fp2−左耳前, EEG3=ECG(雙鎖骨下), EEG4=ROC−A1(EOG)。
"""
from __future__ import annotations
import numpy as np

EEG_RATE = 250
UV_PER_LSB = 25208.3 / 32768.0
# BLE 常數(給 live 用)
XB_NAME = "xb51"
MANUF_ID = 0x9D78
SVC_FFF0 = "0000fff0-0000-1000-8000-00805f9b34fb"
CHAR_NOTIFY_FFF5 = "0000fff5-0000-1000-8000-00805f9b34fb"
CHAR_WRITE_FFF6 = "0000fff6-0000-1000-8000-00805f9b34fb"
CMD_MAGIC = b"KYXB"
CMD_STREAM_ON = CMD_MAGIC + bytes([0x10, 0x80, 0x00])
CMD_STREAM_OFF = CMD_MAGIC + bytes([0x10, 0x00, 0x80])


def _parse_xline(line: bytes):
    if len(line) != 8 or line[:2] != b"x=" or not line[2:].isalnum():
        return None
    try:
        cnt = int(line[2:6], 16) & 0xFFFF
        cs = int(line[6:8], 16) & 0xFF
    except ValueError:
        return None
    calc = ((cnt >> 8) + (cnt & 0xFF)) & 0xFF
    return cnt if calc == cs else None


class H3Parser:
    """增量式:餵位元組,吐出 ('line', str) 或 ('block', channels_list) 事件。"""
    def __init__(self):
        self.mode = "text"
        self.line = bytearray()
        self.hdr = bytearray()
        self.hdr_len = 0
        self.n = self.m = self.total_words = 0
        self.div = []
        self.data = bytearray()
        self.trail = bytearray()
        self.blocks = 0
        self.checksum_errors = 0

    def feed(self, buf: bytes):
        out = []
        i, L = 0, len(buf)
        while i < L:
            if self.mode == "text":
                b = buf[i]; i += 1
                if b == 0x0D:  # \r
                    line = bytes(self.line); self.line.clear()
                    if line:
                        cnt = _parse_xline(line)
                        if cnt is not None:
                            self._start(cnt)
                        else:
                            out.append(("line", line.decode("latin1", "replace")))
                elif b != 0x0A and len(self.line) < 512:
                    self.line.append(b)
            elif self.mode == "header":
                self.hdr.append(buf[i]); i += 1
                if len(self.hdr) == 2:
                    self.n = self.hdr[0] + 1
                    self.m = self.hdr[1] + 1
                    if self.m > 8 or self.n > 64:
                        self._abort(); continue
                    self.hdr_len = ((self.m + 1) >> 1) + 2
                if len(self.hdr) >= 2 and len(self.hdr) == self.hdr_len:
                    nib = []
                    for b in self.hdr[2:]:
                        nib.append((b >> 4) + 1); nib.append((b & 0x0F) + 1)
                    self.div = nib[:self.m]
                    self.total_words = sum(self.n // d for d in self.div)
                    if not (0 < self.total_words <= 2048):
                        self._abort(); continue
                    self.data.clear(); self.mode = "data"
            elif self.mode == "data":
                need = self.total_words * 2 - len(self.data)
                take = min(need, L - i)
                self.data.extend(buf[i:i + take]); i += take
                if len(self.data) == self.total_words * 2:
                    self.trail.clear(); self.mode = "trail"
            elif self.mode == "trail":
                self.trail.append(buf[i]); i += 1
                if len(self.trail) == 2:
                    out.append(("block", self._finish()))
        return out

    def _start(self, cnt):
        self.cur = cnt; self.hdr.clear(); self.hdr_len = 0; self.mode = "header"

    def _abort(self):
        self.mode = "text"; self.line.clear()

    def _finish(self):
        s = sum(self.data) & 0xFFFF
        trail = (self.trail[0] << 8) | self.trail[1]
        if s != trail:
            self.checksum_errors += 1
        chans = [[] for _ in range(self.m)]
        k = 0
        for si in range(self.n):
            for c in range(self.m):
                if si % self.div[c] == 0:
                    w = (self.data[2 * k] << 8) | self.data[2 * k + 1]
                    chans[c].append(w); k += 1
        self.blocks += 1
        self.mode = "text"; self.line.clear()
        return chans


def _blocks_to_channels(events):
    """把多個 block 的同通道樣本串起來 → dict{ch_index: np.array(words)}。"""
    acc = None
    for kind, payload in events:
        if kind != "block":
            continue
        if acc is None:
            acc = [[] for _ in payload]
        for c, ws in enumerate(payload):
            acc[c].extend(ws)
    return [np.array(c, dtype=np.float64) for c in (acc or [])]


def words_to_uv(words):
    return (words - 32768.0) * UV_PER_LSB


class H3ReplaySource:
    """重播一段側錄的 H3 payload(.bin)。驗證解析+接陪伴機器人用，不需硬體。"""
    def __init__(self, path, win_sec=4.0, step_sec=1.0, loops=1, speed=0.0):
        raw = open(path, "rb").read()
        p = H3Parser()
        ev = []
        for _ in range(loops):
            ev += p.feed(raw)
        self.stats = (p.blocks, p.checksum_errors)
        chans = _blocks_to_channels(ev)
        self.eeg = words_to_uv(chans[0])           # EEG1 = Fp1
        self.eog = words_to_uv(chans[3]) if len(chans) > 3 else None  # EEG4 = ROC(EOG)
        self.sfreq = float(EEG_RATE)
        self.win = int(win_sec * self.sfreq)
        self.step = int(step_sec * self.sfreq)
        self.step_sec, self.speed = step_sec, speed

    def windows(self):
        import time
        pos = 0
        n = len(self.eeg)
        while pos + self.win <= n:
            eog = self.eog[pos:pos + self.win] if self.eog is not None else None
            yield dict(eeg=self.eeg[pos:pos + self.win], eog=eog,
                       sfreq=self.sfreq, t=pos / self.sfreq)
            pos += self.step
            if self.speed > 0:
                time.sleep(self.step_sec / self.speed)

    def close(self):
        pass


class H3LiveSource:
    """真.即時:用 Mac 藍牙連 H3。需要 `pip install bleak` 與一台 H3。
    尚未在硬體上跑過(bleak 未安裝),接線邏輯已依 Rust 版 mod.rs 對齊。"""
    def __init__(self, serial_hex, win_sec=4.0, step_sec=1.0, buffer_sec=8.0):
        self.serial_hex = serial_hex.upper()
        self.sfreq = float(EEG_RATE)
        self.win = int(win_sec * self.sfreq)
        self.step = int(step_sec * self.sfreq)
        self.buf_eeg = np.zeros(0)
        self.buf_eog = np.zeros(0)
        self._t0 = 0

    def windows(self):
        import asyncio
        from bleak import BleakScanner, BleakClient
        parser = H3Parser()
        queue: list = []

        def on_notify(_char, value: bytearray):
            for kind, payload in parser.feed(bytes(value[6:])):  # 剝 6-byte 標頭
                if kind == "block":
                    ch = _blocks_to_channels([(kind, payload)])
                    queue.append((words_to_uv(ch[0]),
                                  words_to_uv(ch[3]) if len(ch) > 3 else None))

        async def stream():
            dev = await BleakScanner.find_device_by_filter(
                lambda d, ad: (ad.local_name == XB_NAME) and MANUF_ID in (ad.manufacturer_data or {}),
                timeout=15.0)
            if dev is None:
                raise RuntimeError("掃不到 H3(xb51)。確認手錶已開、未被別台 dongle 佔用")
            async with BleakClient(dev) as client:
                await client.start_notify(CHAR_NOTIFY_FFF5, on_notify)
                await client.write_gatt_char(CHAR_WRITE_FFF6, CMD_STREAM_ON, response=True)
                try:
                    while True:
                        await asyncio.sleep(0.05)
                        while queue:
                            eeg, eog = queue.pop(0)
                            self.buf_eeg = np.concatenate([self.buf_eeg, eeg])
                            if eog is not None:
                                self.buf_eog = np.concatenate([self.buf_eog, eog])
                        while len(self.buf_eeg) >= self.win:
                            eog = self.buf_eog[:self.win] if len(self.buf_eog) >= self.win else None
                            yield_item = dict(eeg=self.buf_eeg[:self.win].copy(), eog=(eog.copy() if eog is not None else None),
                                              sfreq=self.sfreq, t=self._t0 / self.sfreq)
                            self.buf_eeg = self.buf_eeg[self.step:]
                            self.buf_eog = self.buf_eog[self.step:] if len(self.buf_eog) >= self.step else self.buf_eog
                            self._t0 += self.step
                            self._pending = yield_item
                            return  # (簡化:實作時改成把 yield_item 丟回同步 generator)
                finally:
                    await client.write_gatt_char(CHAR_WRITE_FFF6, CMD_STREAM_OFF, response=True)

        raise NotImplementedError(
            "H3LiveSource 需要 bleak 與實機。async→sync 橋接請見 README 的『接 H3 即時』。")

    def close(self):
        pass
