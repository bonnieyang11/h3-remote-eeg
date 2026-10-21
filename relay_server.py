#!/usr/bin/env python3
"""RTSP → WebSocket 即時影像轉播伺服器。

在 Mac mini 上執行：
    python relay_server.py

手機瀏覽器連到 ws://MAC_MINI_IP:9876 就能即時看攝影機畫面。
用 FFmpeg 讀取 RTSP 串流，轉成 MJPEG，透過 WebSocket 廣播。

也支援遠端存取（ngrok/Tailscale）：
    # 安裝 ngrok 後：
    ngrok tcp 9876
    # 或用 Tailscale 直接連 Mac mini IP
"""
from __future__ import annotations
import asyncio
import json
import signal
import subprocess
import sys
import struct
from pathlib import Path

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("需要 websockets 套件：pip install websockets")
    sys.exit(1)

RELAY_PORT = 9876
FRAME_RATE = 10
JPEG_QUALITY = 5  # 2=best, 31=worst

clients: set = set()
frame_count = 0


def load_tapo_config() -> dict:
    conf_path = Path(__file__).parent / "tapo_conf.json"
    if conf_path.exists():
        return json.loads(conf_path.read_text())
    return {}


def build_rtsp_url(conf: dict, ip_override: str | None = None) -> str:
    host = ip_override or conf.get("host", "")
    user = conf.get("user", "")
    password = conf.get("password", "")
    if host == "auto" and not ip_override:
        from eeg_companion.tapo_discover import find_tapo
        result = find_tapo()
        if result:
            host = result["host"]
            print(f"自動發現攝影機: {host}")
        else:
            print("找不到攝影機，請指定 IP")
            sys.exit(1)
    return f"rtsp://{user}:{password}@{host}:554/stream1"


async def broadcast(frame_bytes: bytes):
    global frame_count
    frame_count += 1
    if not clients:
        return
    import base64
    data = base64.b64encode(frame_bytes).decode('ascii')
    msg = json.dumps({"type": "frame", "data": data, "n": frame_count})
    dead = set()
    for ws in clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    clients -= dead


async def ffmpeg_reader(rtsp_url: str):
    """FFmpeg 讀 RTSP，輸出 MJPEG，解析 JPEG frame boundaries。"""
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", str(JPEG_QUALITY),
        "-r", str(FRAME_RATE),
        "-vf", "scale=640:-1",
        "pipe:1",
    ]
    print(f"啟動 FFmpeg: {' '.join(cmd[:6])}...")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    buf = bytearray()
    SOI = b'\xff\xd8'
    EOI = b'\xff\xd9'

    try:
        while True:
            chunk = await proc.stdout.read(32768)
            if not chunk:
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
                frame = bytes(buf[soi:eoi + 2])
                del buf[:eoi + 2]
                await broadcast(frame)
    finally:
        proc.kill()
        await proc.wait()


async def handler(ws):
    clients.add(ws)
    remote = ws.remote_address
    print(f"[+] 連線: {remote[0]}:{remote[1]} (共 {len(clients)} 人)")
    try:
        async for msg in ws:
            pass
    finally:
        clients.discard(ws)
        print(f"[-] 斷線: {remote[0]}:{remote[1]} (共 {len(clients)} 人)")


async def status_printer():
    while True:
        await asyncio.sleep(10)
        if clients:
            print(f"[狀態] {len(clients)} 個連線, 已送 {frame_count} 張 frame")


async def main(rtsp_url: str):
    print(f"╔══════════════════════════════════════╗")
    print(f"║   RTSP → WebSocket 即時影像轉播      ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  WebSocket: ws://0.0.0.0:{RELAY_PORT}       ║")
    print(f"║  RTSP: {rtsp_url[:36]:<36} ║")
    print(f"║  FPS: {FRAME_RATE}, 解析度: 640p            ║")
    print(f"╚══════════════════════════════════════╝")
    print()

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"📱 手機連線網址: ws://{local_ip}:{RELAY_PORT}")
        print(f"   在 RBD Monitor App 的攝影機頁面輸入: {local_ip}")
    except Exception:
        print(f"📱 在 RBD Monitor App 的攝影機頁面輸入 Mac mini 的 IP")
    print()

    async with serve(handler, "0.0.0.0", RELAY_PORT):
        reader_task = asyncio.create_task(ffmpeg_reader(rtsp_url))
        status_task = asyncio.create_task(status_printer())
        try:
            await asyncio.gather(reader_task, status_task)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    ip_override = sys.argv[1] if len(sys.argv) > 1 else None
    conf = load_tapo_config()
    rtsp_url = build_rtsp_url(conf, ip_override)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main(rtsp_url))
    except KeyboardInterrupt:
        print("\n停止轉播")
