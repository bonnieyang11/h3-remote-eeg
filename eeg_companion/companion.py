"""主迴圈：來源 -> 特徵 -> 狀態機 -> 該講話時就講話。
把三塊接起來，並管理「不要一直碎念」的冷卻時間。
"""
from __future__ import annotations
import random, time
from . import features
from .state import StateMachine, ALERT, DROWSY, MICROSLEEP

LINES = {
    DROWSY: ["你好像開始恍神了，要不要起來動一動？",
             "偵測到想睡的腦波，喝口水醒醒腦吧。",
             "注意力在下降囉，深呼吸一下。"],
    MICROSLEEP: ["你快睡著了，先停下來休息比較安全。",
                 "偵測到微睡眠，現在請不要繼續開車或操作。"],
    ALERT: ["精神回來了，狀態不錯，繼續加油。"],
}


class Companion:
    def __init__(self, source, voice, cooldown=20.0, on_status=None):
        self.source, self.voice = source, voice
        self.sm = StateMachine()
        self.cooldown = cooldown
        self.on_status = on_status
        self._last_spoke = -1e9

    def _maybe_speak(self, state, now):
        if state == ALERT and self._last_spoke < 0:
            return
        if now - self._last_spoke < self.cooldown:
            return
        line = random.choice(LINES[state])
        self.voice.speak(line)
        self._last_spoke = now

    def run(self):
        for win in self.source.windows():
            feat = features.compute(win)
            state, z, changed = self.sm.update(feat)
            if self.on_status:
                self.on_status(feat, state, z, changed)
            if changed and state in (DROWSY, MICROSLEEP):
                self._maybe_speak(state, feat["t"])
        self.source.close()
