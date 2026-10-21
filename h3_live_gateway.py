#!/usr/bin/env python3
"""真 H3 即時閘道:藍牙連 H3 → 開串流 → 解析 → 從 ECG 算心率 → 發進中央監測站。
把你本人的 H3 資料發成裝置 U01,監測站那格就是你即時的心率與心電。

在 gaze 環境、用 ! 自己跑(需藍牙權限):
  python h3_live_gateway.py [裝置位址] [uid=U01]
預設連 0DDF0060(位址 B9B3E110-F93A-F5FE-98B2-23AE288E051D)。
心率需要 ECG 電極(雙鎖骨下)貼好;沒貼會顯示無心跳但仍連線串流。
"""
import sys, asyncio, json, time, collections
import numpy as np
from scipy.signal import butter, filtfilt
from bleak import BleakClient
import paho.mqtt.client as mqtt
import mqtt_conf as mc
from eeg_companion.h3 import (H3Parser, CHAR_NOTIFY_FFF5, CHAR_WRITE_FFF6,
                              CMD_STREAM_ON, CMD_STREAM_OFF, words_to_uv)
from eeg_companion import features

ADDR = sys.argv[1] if len(sys.argv) > 1 else "B9B3E110-F93A-F5FE-98B2-23AE288E051D"
UID = sys.argv[2] if len(sys.argv) > 2 else "U01"
BROKER, PORT = mc.BROKER, mc.PORT
ECG_CH = 2                       # EEG3 = ECG
FS = 250
ecg_buf = collections.deque(maxlen=FS * 6)
parser = H3Parser()


def mkc():
    try: return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"h3-{UID}")
    except Exception: return mqtt.Client(client_id=f"h3-{UID}")


def on_notify(_char, data):
    for kind, payload in parser.feed(bytes(data)[6:]):   # 剝 6-byte BLE 標頭
        if kind == "block" and len(payload) > ECG_CH:
            for w in payload[ECG_CH]:
                ecg_buf.append(w)


async def main():
    m = mkc(); mc.configure(m); m.will_set(mc.topic(UID, "status"), "offline", qos=1, retain=True)
    m.connect(BROKER, PORT, 20); m.loop_start()
    m.publish(mc.topic(UID, "status"), "online", qos=1, retain=True)
    print(f"連線 H3 {ADDR} …(發成裝置 {UID})")
    async with BleakClient(ADDR) as client:
        print("已連上,開串流。")
        await client.start_notify(CHAR_NOTIFY_FFF5, on_notify)
        await client.write_gatt_char(CHAR_WRITE_FFF6, CMD_STREAM_ON, response=True)
        try:
            while True:
                await asyncio.sleep(1.0)
                if len(ecg_buf) >= FS * 3:
                    ecg = words_to_uv(np.array(ecg_buf))
                    hr = features.heart_rate(ecg, FS)
                    if hr and 30 < hr < 220:
                        m.publish(mc.topic(UID, "metrics"), json.dumps({"t": time.time(), "hr": round(hr, 1)}), qos=1)
                        # 帶通 1–40Hz 殺基線飄移,取最後 1 秒、置中、以 R 波高度標準化到約 ±1
                        win = ecg[-FS * 4:]
                        b, a = butter(2, [1.0 / (FS / 2), 40 / (FS / 2)], btype="band")
                        filt = filtfilt(b, a, win)
                        seg = filt[-FS:][::2]
                        seg = seg - np.median(seg)
                        scale = np.percentile(np.abs(seg), 98) + 1e-6
                        seg = np.clip(seg / scale, -1.2, 1.2)
                        m.publish(mc.topic(UID, "ecg"), json.dumps([round(float(v), 3) for v in seg]), qos=0)
                        print(f"HR = {hr:.0f} bpm  (已發進監測站 {UID})", flush=True)
                    else:
                        print("ECG 無明顯心跳,電極(雙鎖骨下)貼好了嗎?", flush=True)
        finally:
            try:
                await client.write_gatt_char(CHAR_WRITE_FFF6, CMD_STREAM_OFF, response=True)
            except Exception:
                pass
    m.publish(mc.topic(UID, "status"), "offline", qos=1, retain=True); m.loop_stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("停止。")
