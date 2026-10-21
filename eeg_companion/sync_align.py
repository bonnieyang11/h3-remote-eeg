"""H3 與攝影機時間對齊工具。

用法:
    1. 現場：H3 和攝影機都開始錄後，在鏡頭前拍 H3 三下
    2. 回來：
       - 從 H3 .bin 找到三次加速度 spike 的時間
       - 從攝影機影片找到拍擊畫面的時間戳
       - 算出偏移量，對齊所有事件

    offset = find_tap_sync(h3_bin_path)
    # → 回傳 H3 資料中三次拍擊的時間 (秒)
    # 對照攝影機影片中的時間戳，算出:
    #   cam_time = h3_time + offset
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


def find_tap_spikes(accel_data: np.ndarray, sfreq: float = 250.0,
                    threshold_g: float = 1.5,
                    min_gap_sec: float = 0.3) -> list[float]:
    """在加速度資料中找到拍擊 spike。

    Args:
        accel_data: 加速度向量長度 (已計算 sqrt(x²+y²+z²))
        sfreq: 取樣率
        threshold_g: spike 閾值 (g)
        min_gap_sec: 兩次 spike 最小間隔 (秒)

    Returns:
        spike 時間列表 (秒)
    """
    baseline = np.median(accel_data)
    peaks = np.where(accel_data > baseline + threshold_g)[0]

    if len(peaks) == 0:
        return []

    # 合併相鄰的 spike
    min_gap = int(min_gap_sec * sfreq)
    spikes = [peaks[0]]
    for p in peaks[1:]:
        if p - spikes[-1] > min_gap:
            spikes.append(p)

    return [float(s / sfreq) for s in spikes]


def find_tap_sync(bin_path: str, sfreq: float = 250.0,
                  search_sec: float = 120.0) -> dict:
    """從 H3 .bin 檔的前 N 秒找三次拍擊同步標記。

    Returns:
        {
            "taps": [t1, t2, t3],      # H3 時間 (秒)
            "intervals": [d1, d2],      # 拍擊間隔 (秒)
            "confidence": float,        # 0~1，三次等間隔越高越好
        }
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from eeg_companion.h3 import H3ReplaySource

    src = H3ReplaySource(bin_path, win_sec=1.0, step_sec=0.5)

    accel_all = []
    t_all = []
    for win in src.windows():
        if win["t"] > search_sec:
            break
        activity = win.get("activity", 0)
        if activity is not None:
            accel_all.append(activity)
            t_all.append(win["t"])

    src.close()

    if not accel_all:
        return {"taps": [], "error": "沒有加速度資料"}

    accel = np.array(accel_all)
    taps_t = find_tap_spikes(accel, sfreq=1.0 / 0.5,
                             threshold_g=0.5, min_gap_sec=0.5)

    # 從 spike 中找最像「三次等間隔拍擊」的組合
    if len(taps_t) >= 3:
        best = None
        best_score = 1e9
        for i in range(len(taps_t) - 2):
            for j in range(i + 1, len(taps_t) - 1):
                for k in range(j + 1, len(taps_t)):
                    t1, t2, t3 = taps_t[i], taps_t[j], taps_t[k]
                    d1, d2 = t2 - t1, t3 - t2
                    # 三次拍擊通常間隔 0.5~2 秒
                    if 0.3 < d1 < 3.0 and 0.3 < d2 < 3.0:
                        score = abs(d1 - d2)  # 等間隔越好
                        if score < best_score:
                            best_score = score
                            best = (t1, t2, t3, d1, d2)

        if best:
            t1, t2, t3, d1, d2 = best
            regularity = 1.0 - min(best_score / 1.0, 1.0)
            return {
                "taps": [t1, t2, t3],
                "intervals": [round(d1, 2), round(d2, 2)],
                "confidence": round(regularity, 2),
            }

    return {
        "taps": taps_t[:3] if taps_t else [],
        "note": f"找到 {len(taps_t)} 個 spike，不確定哪些是同步拍擊",
    }


def compute_offset(h3_tap_sec: float, cam_timestamp: str) -> float:
    """計算 H3→攝影機的時間偏移。

    Args:
        h3_tap_sec: H3 資料中拍擊的時間 (秒，從錄製開始算)
        cam_timestamp: 攝影機影片中同一拍擊的時間戳 "HH:MM:SS"

    Returns:
        offset (秒): cam_time = h3_time + offset
    """
    parts = cam_timestamp.strip().split(":")
    cam_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return cam_sec - h3_tap_sec


def align_events(events_jsonl: str, offset: float,
                 cam_video_dir: str | None = None) -> list[dict]:
    """把 RBD 事件的 H3 時間轉成攝影機時間，方便查找影片。

    Args:
        events_jsonl: rbd_log_*.jsonl 路徑
        offset: compute_offset 的結果
        cam_video_dir: Tapo SD 卡的影片目錄 (選填)

    Returns:
        帶攝影機時間的事件列表
    """
    import json

    aligned = []
    with open(events_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if "rbd_event" not in rec:
                continue
            h3_t = rec["t"]
            cam_t = h3_t + offset
            h = int(cam_t // 3600) % 24
            m = int((cam_t % 3600) // 60)
            s = int(cam_t % 60)
            cam_time_str = f"{h:02d}:{m:02d}:{s:02d}"

            evt = {
                "h3_time_sec": h3_t,
                "cam_time": cam_time_str,
                "stage": rec.get("stage"),
                **rec["rbd_event"],
            }

            # Tapo SD 卡的檔案結構通常是 YYYY-MM-DD/HH/MM.mp4
            if cam_video_dir:
                video_file = f"{h:02d}/{m:02d}.mp4"
                evt["video_file"] = video_file

            aligned.append(evt)

    return aligned


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  python sync_align.py <H3.bin>              # 找同步拍擊")
        print("  python sync_align.py <log.jsonl> <offset>  # 對齊事件")
        sys.exit(1)

    path = sys.argv[1]
    if path.endswith(".bin"):
        result = find_tap_sync(path)
        print(f"同步拍擊: {result}")
        if result.get("taps"):
            print(f"\n找到拍擊時間 (H3 秒): {result['taps']}")
            print(f"對照攝影機影片中同一拍擊的時間戳，執行:")
            print(f"  python sync_align.py <log.jsonl> <offset>")
            print(f"  offset = 攝影機秒數 - {result['taps'][0]:.1f}")
    elif path.endswith(".jsonl") and len(sys.argv) >= 3:
        offset = float(sys.argv[2])
        events = align_events(path, offset)
        print(f"RBD 事件 (攝影機時間):")
        for e in events:
            print(f"  {e['cam_time']}  {e.get('detail', '')}")
