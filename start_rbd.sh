#!/bin/bash
# ═══════════════════════════════════════════
#  RBD 監測一鍵啟動 (MacBook 用)
#
#  AirDrop 這三個檔案到 MacBook：
#    1. start_rbd.sh      (這個腳本)
#    2. relay_portable.py  (影像轉播)
#    3. rbd_app.html       (手機 App)
#
#  然後在 MacBook 終端機：
#    chmod +x start_rbd.sh
#    ./start_rbd.sh
# ═══════════════════════════════════════════

set -e

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║     RBD 睡眠監測系統 - 現場部署        ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# ─── 檢查相依套件 ───
echo "🔍 檢查環境..."

if ! command -v python3 &>/dev/null; then
    echo "❌ 需要 Python 3"
    exit 1
fi
echo "  ✓ Python3"

if ! command -v ffmpeg &>/dev/null; then
    echo "❌ 需要 FFmpeg，安裝中..."
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "  請先安裝 Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "  然後: brew install ffmpeg"
        exit 1
    fi
fi
echo "  ✓ FFmpeg"

python3 -c "import websockets" 2>/dev/null || {
    echo "  安裝 websockets..."
    pip3 install websockets
}
echo "  ✓ websockets"
echo ""

# ─── 取得本機 IP ───
LOCAL_IP=$(python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('8.8.8.8',80));print(s.getsockname()[0]);s.close()" 2>/dev/null || echo "unknown")

echo "═══════════════════════════════════════"
echo ""
echo "📡 MacBook IP: $LOCAL_IP"
echo ""
echo "═══════════════════════════════════════"
echo ""
echo "🔧 接下來需要設定 WiFi 熱點 (一次就好):"
echo ""
echo "   ┌─────────────────────────────────────┐"
echo "   │  系統設定 → 一般 → 共享              │"
echo "   │  → 網際網路共享                       │"
echo "   │                                      │"
echo "   │  共享來源: iPhone USB / Thunderbolt   │"
echo "   │  透過: Wi-Fi                         │"
echo "   │                                      │"
echo "   │  點「Wi-Fi 選項」:                    │"
echo "   │    名稱 = 你家 WiFi SSID               │"
echo "   │    密碼 = 你家 WiFi 密碼              │"
echo "   │    安全性 = WPA2/WPA3 Personal       │"
echo "   │                                      │"
echo "   │  打開「網際網路共享」開關              │"
echo "   └─────────────────────────────────────┘"
echo ""
echo "   💡 這樣 Tapo 攝影機開機就會自動連上!"
echo "      不需要重新設定攝影機"
echo ""
echo "   設定好了嗎？按 Enter 繼續..."
read -r

echo ""
echo "🔍 搜尋 Tapo 攝影機..."
echo ""

# ─── 啟動轉播 ───
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/relay_portable.py" "$@"
