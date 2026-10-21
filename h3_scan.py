#!/usr/bin/env python3
"""掃描藍牙,找 H3 腦波儀(廣播名 xb51、廠商碼 0x9D78)。
第一次跑,macOS 會要求「藍牙」權限:系統設定 → 隱私權與安全性 → 藍牙,勾選你的終端機。

在 gaze 環境跑:  python h3_scan.py
"""
import asyncio
from bleak import BleakScanner

MANUF = 0x9D78


def sid(md):
    """從 manufacturer data 還原 H3 序號:ID[lo16] STATUS[16] ID[hi16]。"""
    b = md.get(MANUF)
    if not b or len(b) < 6:
        return None
    lo = b[0] | (b[1] << 8); hi = b[4] | (b[5] << 8)
    return f"{hi:04X}{lo:04X}"


async def main():
    print("掃描中(12 秒)… 確認 H3 已開機、在附近、沒被別台連著。")
    found = await BleakScanner.discover(timeout=12.0, return_adv=True)
    hits = []
    for addr, (d, adv) in found.items():
        name = adv.local_name or (d.name or "")
        md = adv.manufacturer_data or {}
        is_h3 = (name == "xb51") or (MANUF in md)
        line = f"{'★H3' if is_h3 else '  '}  {addr}  name={name!r}  rssi={adv.rssi}"
        if MANUF in md:
            line += f"  序號={sid(md)}"
        print(line)
        if is_h3:
            hits.append((addr, sid(md)))
    print("\n找到 H3:", hits if hits else "沒有(檢查:H3 開機?藍牙權限?是否被 dongle 佔用?)")


if __name__ == "__main__":
    asyncio.run(main())
