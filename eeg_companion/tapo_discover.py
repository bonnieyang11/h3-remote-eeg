"""自動搜尋區網內的 Tapo 攝影機。

到不同的網路環境（受試者家裡等）不用手動查 IP，
掃描本機所在子網的 443 port + 嘗試 pytapo 連線來找 Tapo 攝影機。
"""
from __future__ import annotations
import socket
import subprocess
import sys
import concurrent.futures


def get_local_subnet() -> str | None:
    """取得本機的區網網段，例如 '192.168.1'。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        parts = ip.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3])
    except Exception:
        pass
    return None


def _check_port(ip: str, port: int = 443, timeout: float = 1.0) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((ip, port))
        s.close()
        return result == 0
    except Exception:
        return False


def scan_subnet(subnet: str | None = None, port: int = 443,
                timeout: float = 1.0) -> list[str]:
    """掃描子網中開了指定 port 的 IP。"""
    if subnet is None:
        subnet = get_local_subnet()
    if subnet is None:
        print("[discover] 無法偵測本機網段")
        return []

    print(f"[discover] 掃描 {subnet}.1-254 port {port} ...")
    candidates = []
    ips = [f"{subnet}.{i}" for i in range(1, 255)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
        futures = {pool.submit(_check_port, ip, port, timeout): ip for ip in ips}
        for f in concurrent.futures.as_completed(futures):
            if f.result():
                candidates.append(futures[f])

    candidates.sort(key=lambda x: int(x.split(".")[-1]))
    print(f"[discover] 找到 {len(candidates)} 個開了 port {port} 的 IP")
    return candidates


def find_tapo(user: str, password: str, subnet: str | None = None,
              cloud_password: str = "") -> dict | None:
    """掃描子網並嘗試 RTSP 連線，回傳第一台成功的攝影機資訊。"""
    import cv2

    # 同時掃 443 和 554 (RTSP)
    candidates_443 = scan_subnet(subnet, port=443)
    candidates_554 = scan_subnet(subnet, port=554)
    candidates = list(dict.fromkeys(candidates_554 + candidates_443))

    if not candidates:
        return None

    for ip in candidates:
        url = f"rtsp://{user}:{password}@{ip}:554/stream1"
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"[discover] ★ 找到 Tapo 攝影機: {ip} ({w}x{h})")
                    return {"host": ip, "model": "Tapo", "resolution": f"{w}x{h}"}
        except Exception:
            continue

    print("[discover] 子網中沒有找到能用這組帳密連上的 Tapo 攝影機")
    return None


if __name__ == "__main__":
    print(f"本機網段: {get_local_subnet()}")
    candidates = scan_subnet()
    for ip in candidates:
        print(f"  {ip}")
