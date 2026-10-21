"""狀態機：把每秒的想睡指數變成穩定的 ALERT / DROWSY / MICROSLEEP。

做法：先用前 baseline_sec 秒(假設此時清醒)算基準線的平均與標準差，
之後把每個 D 轉成 z 分數並做指數平滑，用遲滯門檻避免因單一雜訊亂報。
自動校準 = 換人戴、換一天都不用重調參數。
"""
from __future__ import annotations
from collections import deque
import numpy as np

ALERT, DROWSY, MICROSLEEP = "ALERT", "DROWSY", "MICROSLEEP"


class StateMachine:
    def __init__(self, baseline_sec=30, ema=0.3,
                 z_drowsy=1.5, z_micro=3.0, z_recover=0.8, need=4):
        self.baseline_sec = baseline_sec
        self.ema = ema
        self.z_drowsy, self.z_micro, self.z_recover = z_drowsy, z_micro, z_recover
        self.need = need
        self._base = []
        self.mu = self.sd = None
        self.z = 0.0
        self.state = ALERT
        self._hits = 0
        self._recov = 0

    def update(self, feat):
        """吃一筆 features.compute 的結果，回傳 (state, z, transitioned)。"""
        D = feat["D"]
        if self.mu is None:
            self._base.append(D)
            if feat["t"] >= self.baseline_sec and len(self._base) > 5:
                self.mu = float(np.mean(self._base))
                self.sd = max(float(np.std(self._base)), 0.05*abs(self.mu), 1e-3)
            return self.state, 0.0, False

        raw_z = (D - self.mu) / self.sd
        self.z = (1 - self.ema) * self.z + self.ema * raw_z
        prev = self.state

        if self.z >= self.z_micro:
            self._hits += 1
            if self._hits >= max(2, self.need // 2):
                self.state = MICROSLEEP
        elif self.z >= self.z_drowsy:
            self._hits += 1
            if self._hits >= self.need and self.state == ALERT:
                self.state = DROWSY
        else:
            self._hits = 0

        if self.z < self.z_recover:
            self._recov += 1
            if self._recov >= self.need:
                self.state = ALERT
        else:
            self._recov = 0

        return self.state, self.z, self.state != prev


import math


class FusionState:
    """多證據融合:腦波為主,心率下降與姿勢前傾/靜止為佐證。
    規則:腦波 z 達門檻「且」有另一路佐證 → DROWSY;腦波極強(z≥z_micro)單獨 → MICROSLEEP。
    這樣腦波是必要條件,但要有第二個訊號附和才報想睡,可大幅降低誤報。
    心率、重力向量各自用前 baseline_sec 秒自動校準。
    """
    def __init__(self, baseline_sec=30, ema=0.3, z_drowsy=1.5, z_micro=3.0,
                 hr_drop=6.0, tilt_deg=15.0, still=0.015, need=4):
        self.eeg = StateMachine(baseline_sec=baseline_sec, ema=ema,
                                z_drowsy=z_drowsy, z_micro=z_micro, need=need)
        self.baseline_sec = baseline_sec
        self.hr_drop, self.tilt_deg, self.still = hr_drop, tilt_deg, still
        self.hr_mu = None; self._hr_base = []; self.hr_ema = None
        self.g0 = None; self._g_base = []
        self.state = ALERT

    @staticmethod
    def _angle(a, b):
        na = math.sqrt(sum(v*v for v in a)); nb = math.sqrt(sum(v*v for v in b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        d = sum(x*y for x, y in zip(a, b)) / (na*nb)
        return math.degrees(math.acos(max(-1.0, min(1.0, d))))

    def update(self, feat):
        st_eeg, z, _ = self.eeg.update(feat)
        t = feat["t"]

        hr = feat.get("hr")
        if hr is not None:
            self.hr_ema = hr if self.hr_ema is None else 0.7*self.hr_ema + 0.3*hr
            if self.hr_mu is None and t < self.baseline_sec:
                self._hr_base.append(hr)
            elif self.hr_mu is None and self._hr_base:
                self.hr_mu = sum(self._hr_base)/len(self._hr_base)

        g = feat.get("g"); tilt = None
        if g is not None:
            if self.g0 is None and t < self.baseline_sec:
                self._g_base.append(g)
            elif self.g0 is None and self._g_base:
                n = len(self._g_base)
                self.g0 = tuple(sum(v[i] for v in self._g_base)/n for i in range(3))
            if self.g0 is not None:
                tilt = self._angle(g, self.g0)

        ev = []
        if z >= self.eeg.z_drowsy:
            ev.append("腦波")
        hr_vote = self.hr_mu is not None and self.hr_ema is not None and self.hr_ema < self.hr_mu - self.hr_drop
        if hr_vote:
            ev.append("心率下降")
        posture_vote = tilt is not None and tilt > self.tilt_deg
        if posture_vote:
            ev.append("姿勢前傾")
        still_vote = feat.get("activity") is not None and feat["activity"] < self.still
        startle = feat.get("jerk") is not None and feat["jerk"] > 0.6

        if z >= self.eeg.z_micro:
            state = MICROSLEEP
        elif ("腦波" in ev) and (hr_vote or posture_vote or still_vote):
            state = DROWSY
        else:
            state = ALERT
        self.state = state
        return state, dict(z=z, hr=self.hr_ema, tilt=tilt,
                           activity=feat.get("activity"), evidences=ev, startle=startle)
