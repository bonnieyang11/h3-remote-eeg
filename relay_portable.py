#!/usr/bin/env python3
"""可攜式 RTSP → WebSocket 即時影像轉播。

只需要 Python 3 + FFmpeg + websockets，不依賴其他模組。
AirDrop 這一個檔案到 MacBook 就能用。

用法：
    pip install websockets
    python relay_portable.py

    # 或指定攝影機 IP：
    python relay_portable.py 192.168.1.100
"""
import asyncio
import base64
import json
import socket
import subprocess
import sys

# ─── Tapo 攝影機設定 ───
import os
TAPO_USER = os.environ.get("TAPO_USER", "")
TAPO_PASS = os.environ.get("TAPO_PASS", "")
RELAY_PORT = int(os.environ.get("RELAY_PORT", "9876"))
FPS = 10

# 也可以從 .env 讀
_env = Path(__file__).parent / ".env"
if _env.exists() and not TAPO_USER:
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k == "TAPO_USER": TAPO_USER = v
            elif k == "TAPO_PASS": TAPO_PASS = v

if not TAPO_USER or not TAPO_PASS:
    print("❌ 需要設定攝影機帳密：")
    print("   建立 .env 檔案：")
    print('   TAPO_USER="your_user"')
    print('   TAPO_PASS="your_pass"')
    print("   或用環境變數：TAPO_USER=xxx TAPO_PASS=xxx python relay_portable.py")
    sys.exit(1)

try:
    from websockets.server import serve
except ImportError:
    print("❌ 需要安裝 websockets：")
    print("   pip install websockets")
    sys.exit(1)

# 檢查 FFmpeg
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
except FileNotFoundError:
    print("❌ 需要安裝 FFmpeg：")
    print("   brew install ffmpeg")
    sys.exit(1)

clients: set = set()
frame_n = 0


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_camera():
    """掃描區網找 Tapo 攝影機（port 554）。"""
    local_ip = get_local_ip()
    prefix = ".".join(local_ip.split(".")[:3]) + "."
    print(f"掃描 {prefix}0/24 找攝影機...")
    import concurrent.futures

    def check(ip):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((ip, 554))
            s.close()
            return ip
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
        futures = {pool.submit(check, prefix + str(i)): i for i in range(1, 255)}
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if result and result != local_ip:
                print(f"  找到: {result}")
                return result
    return None


async def broadcast(frame_bytes: bytes):
    global frame_n
    frame_n += 1
    if not clients:
        return
    data = base64.b64encode(frame_bytes).decode("ascii")
    msg = json.dumps({"type": "frame", "data": data, "n": frame_n})
    dead = set()
    for ws in clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    clients -= dead


async def ffmpeg_reader(rtsp_url: str):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-f", "image2pipe", "-vcodec", "mjpeg",
        "-q:v", "5", "-r", str(FPS),
        "-vf", "scale=640:-1",
        "pipe:1",
    ]
    print(f"FFmpeg 啟動中...")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    buf = bytearray()
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    try:
        while True:
            chunk = await proc.stdout.read(32768)
            if not chunk:
                err = await proc.stderr.read()
                if err:
                    print(f"FFmpeg 錯誤: {err.decode('utf-8','replace')[:200]}")
                break
            buf.extend(chunk)
            while True:
                soi = buf.find(SOI)
                if soi < 0:
                    buf.clear()
                    break
                eoi = buf.find(EOI, soi + 2)
                if eoi < 0:
                    if soi > 0:
                        del buf[:soi]
                    break
                frame = bytes(buf[soi : eoi + 2])
                del buf[: eoi + 2]
                await broadcast(frame)
    finally:
        proc.kill()
        await proc.wait()


async def handler(ws):
    clients.add(ws)
    addr = ws.remote_address
    print(f"  [+] {addr[0]} 已連線 (共 {len(clients)} 人)")
    try:
        async for _ in ws:
            pass
    finally:
        clients.discard(ws)
        print(f"  [-] {addr[0]} 已斷線 (共 {len(clients)} 人)")


async def status_loop():
    while True:
        await asyncio.sleep(15)
        if clients:
            print(f"  [狀態] {len(clients)} 連線, {frame_n} frames")


async def main(cam_ip: str):
    rtsp = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{cam_ip}:554/stream1"
    local_ip = get_local_ip()

    print()
    print("╔═══════════════════════════════════════╗")
    print("║   RBD 即時影像轉播伺服器               ║")
    print("╠═══════════════════════════════════════╣")
    print(f"║  攝影機: {cam_ip:<29}║")
    print(f"║  WebSocket: ws://{local_ip}:{RELAY_PORT:<13}║")
    print("╚═══════════════════════════════════════╝")
    print()
    print(f"📱 手機 RBD Monitor → 攝影機分頁 → 輸入: {local_ip}")
    print()

    async with serve(handler, "0.0.0.0", RELAY_PORT):
        reader = asyncio.create_task(ffmpeg_reader(rtsp))
        status = asyncio.create_task(status_loop())
        try:
            await asyncio.gather(reader, status)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cam_ip = sys.argv[1]
    else:
        print("🔍 自動搜尋攝影機...")
        cam_ip = find_camera()
        if not cam_ip:
            print("❌ 找不到攝影機，請手動指定 IP：")
            print(f"   python {sys.argv[0]} 192.168.1.xxx")
            sys.exit(1)

    try:
        asyncio.run(main(cam_ip))
    except KeyboardInterrupt:
        print("\n停止轉播")
