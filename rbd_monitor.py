"""RBD 整夜監測腳本 — 帶去受試者家跑的主程式。

用法:
    python rbd_monitor.py --replay demo.bin                    # 重播測試
    python rbd_monitor.py --serial AABBCCDD --cam              # H3 + 攝影機
    python rbd_monitor.py --serial AABBCCDD --cam --dashboard  # 加上即時面板

面板:
    加 --dashboard 會在 http://localhost:8080 開啟即時監測面板，
    用瀏覽器打開就能看到即時數據。

需要:
    pip install bleak paho-mqtt numpy scipy pytapo opencv-python-headless websockets
"""
from __future__ import annotations
import argparse, json, time, sys, os, threading, subprocess, platform

sys.path.insert(0, os.path.dirname(__file__))
from eeg_companion.rbd import RBDDetector, SleepStage
from eeg_companion.features import compute_mm, accel_features

# ---------------------------------------------------------------------------
# Audio alert
# ---------------------------------------------------------------------------
def _alert_sound(level: str = "warning"):
    """播放系統警報音。level: 'warning' 或 'critical'。"""
    try:
        if platform.system() == "Darwin":
            sound = "/System/Library/Sounds/Sosumi.aiff" if level == "warning" \
                else "/System/Library/Sounds/Funk.aiff"
            subprocess.Popen(["afplay", sound],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def _alert_speak(text: str):
    """macOS 語音播報。"""
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["say", "-v", "Mei-Jia", text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Event screenshot
# ---------------------------------------------------------------------------
def _save_screenshot(cam, t: float, out_dir: str):
    """從攝影機截圖存檔，回傳檔案路徑。"""
    try:
        import cv2
        frame = cam.snapshot()
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"rbd_event_{ts}_t{int(t)}.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, frame)
        return path
    except Exception as e:
        print(f"\n  [screenshot] 截圖失敗: {e}")
        return None


# ---------------------------------------------------------------------------
# WebSocket dashboard server
# ---------------------------------------------------------------------------
class DashboardServer:
    """在背景執行 WebSocket + HTTP server，推送即時資料給面板。"""
    def __init__(self, host="0.0.0.0", ws_port=8765, http_port=8080):
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port
        self._clients: set = set()
        self._loop = None

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        # Also start simple HTTP server for dashboard HTML
        ht = threading.Thread(target=self._run_http, daemon=True)
        ht.start()
        time.sleep(0.3)
        print(f"[Dashboard] 面板: http://localhost:{self.http_port}")
        print(f"[Dashboard] WebSocket: ws://localhost:{self.ws_port}")

    def _run(self):
        import asyncio
        try:
            import websockets
        except ImportError:
            print("[Dashboard] 需要 websockets: pip install websockets")
            return

        async def handler(ws):
            self._clients.add(ws)
            try:
                async for _ in ws:
                    pass
            finally:
                self._clients.discard(ws)

        async def main():
            self._loop = asyncio.get_event_loop()
            async with websockets.serve(handler, self.host, self.ws_port):
                await asyncio.Future()

        asyncio.run(main())

    def _run_http(self):
        import http.server
        import functools

        dash_dir = os.path.join(os.path.dirname(__file__), "dashboard")
        os.makedirs(dash_dir, exist_ok=True)
        dash_html = os.path.join(dash_dir, "index.html")
        if not os.path.exists(dash_html):
            _write_dashboard_html(dash_html, self.ws_port)

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=dash_dir)

        with http.server.HTTPServer((self.host, self.http_port), handler) as srv:
            srv.serve_forever()

    def broadcast(self, data: dict):
        if not self._clients or self._loop is None:
            return
        import asyncio
        msg = json.dumps(data, ensure_ascii=False)

        async def _send():
            dead = set()
            for ws in self._clients.copy():
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

        asyncio.run_coroutine_threadsafe(_send(), self._loop)


def _write_dashboard_html(path: str, ws_port: int):
    """產生可接收 WebSocket 資料的面板 HTML。"""
    html = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RBD Monitor — Live</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@400;500;600&display=swap">
<style>
:root{--bg:#060a12;--surface:#0a1020;--surface-2:#0e1630;--surface-3:#122040;
--cyan:#00ccee;--cyan-dim:#0088aa;--cyan-glow:rgba(0,204,238,0.25);
--green:#00dd77;--amber:#ddaa22;--blue:#2266dd;--red:#dd2244;--red-glow:rgba(221,34,68,0.3);
--text-1:#d8e4f0;--text-2:#6888a8;--text-3:#384860;
--grid-line:rgba(0,160,220,0.06);--border:rgba(0,180,240,0.12);}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text-1);
padding:16px;min-height:100vh;-webkit-font-smoothing:antialiased;}
body::before{content:'';position:fixed;inset:0;
background:linear-gradient(var(--grid-line) 1px,transparent 1px),linear-gradient(90deg,var(--grid-line) 1px,transparent 1px);
background-size:40px 40px;pointer-events:none;z-index:0;}
.dash{position:relative;z-index:1;}
.hud{position:relative;border:1px solid var(--border);background:var(--surface);
border-radius:2px;overflow:hidden;padding:14px;}
.hud::before,.hud::after{content:'';position:absolute;width:14px;height:14px;
border-color:var(--cyan-dim);border-style:solid;z-index:2;pointer-events:none;}
.hud::before{top:-1px;left:-1px;border-width:2px 0 0 2px;}
.hud::after{bottom:-1px;right:-1px;border-width:0 2px 2px 0;}
.top{display:flex;align-items:center;gap:12px;padding:10px 16px;margin-bottom:10px;flex-wrap:wrap;}
.logo{font-family:'Orbitron';font-size:14px;font-weight:700;letter-spacing:0.12em;
color:var(--cyan);text-shadow:0 0 12px var(--cyan-glow);text-transform:uppercase;}
.dot{width:6px;height:6px;border-radius:50%;animation:p 2s ease-in-out infinite;}
.dot.on{background:var(--green);box-shadow:0 0 6px var(--green);}
.dot.off{background:var(--text-3);box-shadow:none;animation:none;}
@keyframes p{0%,100%{opacity:1}50%{opacity:.5}}
.ind{display:flex;align-items:center;gap:6px;font:11px 'JetBrains Mono',monospace;color:var(--text-2);}
.clock{margin-left:auto;text-align:right;}
.clock-v{font-family:'Orbitron';font-size:22px;font-weight:600;color:var(--cyan);
text-shadow:0 0 8px var(--cyan-glow);letter-spacing:.06em;font-variant-numeric:tabular-nums;}
.clock-s{font:10px 'JetBrains Mono',monospace;color:var(--text-3);letter-spacing:.06em;text-transform:uppercase;}
.sec-label{font-family:'Orbitron';font-size:9px;font-weight:600;color:var(--cyan-dim);
letter-spacing:.16em;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px;}
.sec-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
@media(max-width:700px){.grid{grid-template-columns:1fr;}.grid>*{grid-column:1!important;}}
.wave-panel{grid-column:1/-1;position:relative;min-height:200px;}
canvas.wc{width:100%;height:150px;display:block;}
.wo{position:absolute;top:14px;right:14px;text-align:right;}
.stage-big{font-family:'Orbitron';font-size:36px;font-weight:800;letter-spacing:.04em;line-height:1;}
.stage-sub{font:11px 'JetBrains Mono',monospace;color:var(--text-2);margin-top:4px;}
.gauges{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;}
.g-card{display:flex;flex-direction:column;align-items:center;padding:14px 10px;}
.g-val{font-family:'Orbitron';font-size:20px;font-weight:600;margin-top:6px;letter-spacing:.02em;}
.g-unit{font:10px 'JetBrains Mono',monospace;color:var(--text-3);text-transform:uppercase;letter-spacing:.08em;}
.g-label{font-family:'Orbitron';font-size:8px;font-weight:600;color:var(--text-3);
letter-spacing:.14em;text-transform:uppercase;margin-top:2px;}
.bp-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;}
.bp-name{font:10px 'JetBrains Mono',monospace;color:var(--text-3);width:40px;text-align:right;
text-transform:uppercase;letter-spacing:.04em;}
.bp-track{flex:1;height:8px;background:var(--surface-3);border-radius:1px;overflow:hidden;}
.bp-fill{height:100%;transition:width .4s;border-radius:1px;}
.bp-val{font:10px 'JetBrains Mono',monospace;color:var(--text-2);width:30px;text-align:right;
font-variant-numeric:tabular-nums;}
.bottom{grid-column:1/-1;display:grid;grid-template-columns:1fr 260px;gap:10px;}
@media(max-width:700px){.bottom{grid-template-columns:1fr;}}
.hypno{width:100%;height:110px;display:block;}
.evt-scroll{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;}
.evt-card{flex-shrink:0;min-width:190px;padding:10px 14px;
background:linear-gradient(135deg,rgba(221,34,68,.08),rgba(221,34,68,.02));
border:1px solid rgba(221,34,68,.2);border-left:3px solid var(--red);border-radius:2px;}
.evt-time{font-family:'Orbitron';font-size:11px;font-weight:600;color:var(--red);}
.evt-detail{font-size:11px;color:var(--text-2);margin-top:4px;line-height:1.4;}
.evt-empty{font:11px 'JetBrains Mono',monospace;color:var(--text-3);padding:8px 0;}
.scanline{position:fixed;top:0;left:0;right:0;height:2px;
background:linear-gradient(90deg,transparent,var(--cyan-glow),transparent);
opacity:.4;animation:sc 6s linear infinite;pointer-events:none;z-index:100;}
@keyframes sc{0%{top:0}100%{top:100vh}}
.alert-flash{position:fixed;inset:0;border:2px solid var(--red);
box-shadow:inset 0 0 60px var(--red-glow);opacity:0;pointer-events:none;z-index:99;transition:opacity .1s;}
.alert-flash.on{opacity:1;animation:af .8s ease-out;}
@keyframes af{0%{opacity:1}100%{opacity:0}}
.conn-bar{text-align:center;padding:6px;font:11px 'JetBrains Mono',monospace;
background:rgba(221,34,68,.1);color:var(--red);border:1px solid rgba(221,34,68,.2);
border-radius:2px;margin-bottom:10px;}
.conn-bar.ok{background:rgba(0,221,119,.05);color:var(--green);border-color:rgba(0,221,119,.15);}
</style>
</head><body>
<div class="scanline"></div>
<div class="alert-flash" id="af"></div>
<div class="dash">
<div id="connBar" class="conn-bar">CONNECTING TO MONITOR...</div>
<div class="hud top">
<span class="logo">RBD Monitor</span>
<div style="width:1px;height:20px;background:var(--text-3);"></div>
<div class="ind"><span class="dot on" id="h3d"></span><span>H3</span></div>
<div class="ind"><span class="dot on" id="camd"></span><span>CAM</span></div>
<div class="clock"><div class="clock-s">Elapsed</div><div class="clock-v" id="el">00:00:00</div></div>
</div>
<div class="grid">
<div class="hud wave-panel">
<div class="sec-label">EEG Realtime</div>
<canvas class="wc" id="wc"></canvas>
<div class="wo"><div class="stage-big" id="sb" style="color:var(--amber);">—</div>
<div class="stage-sub" id="ss">waiting for data</div></div>
</div>
<div class="gauges">
<div class="hud g-card"><svg id="hrG" viewBox="0 0 100 60" width="100" height="60"></svg>
<div class="g-val" id="hrV" style="color:var(--green);">—</div><div class="g-unit">bpm</div><div class="g-label">Heart Rate</div></div>
<div class="hud g-card"><svg id="camG" viewBox="0 0 100 60" width="100" height="60"></svg>
<div class="g-val" id="camV" style="color:var(--cyan);font-size:16px;">—</div><div class="g-unit">motion</div><div class="g-label">Camera</div></div>
<div class="hud g-card"><svg id="accG" viewBox="0 0 100 60" width="100" height="60"></svg>
<div class="g-val" id="accV" style="color:var(--amber);font-size:16px;">—</div><div class="g-unit">activity</div><div class="g-label">Accel</div></div>
<div class="hud g-card"><div style="font-family:'Orbitron';font-size:42px;font-weight:800;color:var(--red);text-shadow:0 0 20px var(--red-glow);line-height:1;margin-top:8px;" id="ec">0</div>
<div class="g-unit" style="margin-top:8px;">events</div><div class="g-label">RBD Alert</div></div>
</div>
<div class="bottom">
<div class="hud"><div class="sec-label">Hypnogram</div><svg class="hypno" id="hy" viewBox="0 0 600 110" preserveAspectRatio="none"></svg></div>
<div class="hud">
<div class="sec-label">EEG Band Power</div>
<div class="bp-row"><span class="bp-name">Delta</span><div class="bp-track"><div class="bp-fill" id="fD" style="width:0%;background:var(--blue);box-shadow:0 0 4px rgba(34,102,221,.3);"></div></div><span class="bp-val" id="vD">—</span></div>
<div class="bp-row"><span class="bp-name">Theta</span><div class="bp-track"><div class="bp-fill" id="fT" style="width:0%;background:var(--cyan);box-shadow:0 0 4px var(--cyan-glow);"></div></div><span class="bp-val" id="vT">—</span></div>
<div class="bp-row"><span class="bp-name">Alpha</span><div class="bp-track"><div class="bp-fill" id="fA" style="width:0%;background:var(--amber);box-shadow:0 0 4px rgba(221,170,34,.3);"></div></div><span class="bp-val" id="vA">—</span></div>
<div class="bp-row"><span class="bp-name">Sigma</span><div class="bp-track"><div class="bp-fill" id="fS" style="width:0%;background:#8866cc;box-shadow:0 0 4px rgba(136,102,204,.3);"></div></div><span class="bp-val" id="vS">—</span></div>
<div class="bp-row"><span class="bp-name">Beta</span><div class="bp-track"><div class="bp-fill" id="fB" style="width:0%;background:#cc5588;box-shadow:0 0 4px rgba(204,85,136,.3);"></div></div><span class="bp-val" id="vB">—</span></div>
<div style="margin-top:12px;border-top:1px solid var(--border);padding-top:10px;">
<div class="sec-label" style="margin-bottom:6px;">Distribution</div>
<div style="display:flex;justify-content:space-between;font:10px 'JetBrains Mono',monospace;">
<span><span style="color:var(--amber);">W</span> <span id="sW" style="color:var(--text-2);">—</span></span>
<span><span style="color:var(--blue);">N</span> <span id="sN" style="color:var(--text-2);">—</span></span>
<span><span style="color:var(--cyan);">R</span> <span id="sR" style="color:var(--text-2);">—</span></span></div></div>
</div>
</div>
<div class="hud" style="grid-column:1/-1;">
<div class="sec-label">RBD Event Log</div>
<div class="evt-scroll" id="evts"><div class="evt-empty" id="noevt">// NO EVENTS DETECTED</div></div>
</div>
</div>
</div>
<script>
const WS_PORT = """ + str(ws_port) + r""";
const SC={WAKE:'#ddaa22',NREM:'#2266dd',REM:'#00ccee'};
const SG={WAKE:'rgba(221,170,34,.4)',NREM:'rgba(34,102,221,.4)',REM:'rgba(0,204,238,.4)'};
const SY={WAKE:15,REM:48,NREM:85};
const hist={stages:[],hr:[],cam:[],acc:[]};
let ws,waveStage='WAKE',wavePhase=0,evtN=0;

// Wave
const cv=document.getElementById('wc'),cx=cv.getContext('2d');
function rsz(){const r=cv.getBoundingClientRect();cv.width=r.width*2;cv.height=r.height*2;cx.scale(2,2);}
rsz();window.addEventListener('resize',rsz);
function drawW(){
  const w=cv.width/2,h=cv.height/2,mid=h/2,sc=mid*.7;
  cx.clearRect(0,0,w,h);
  let fr,am,col,gl;
  if(waveStage==='WAKE'){fr=[10,20,8.5,15];am=[.25,.15,.12,.08];col='#ddaa22';gl='rgba(221,170,34,.15)';}
  else if(waveStage==='NREM'){fr=[1.5,2,.8,12];am=[.5,.3,.2,.06];col='#2266dd';gl='rgba(34,102,221,.15)';}
  else{fr=[6,3,1.2,15,.9];am=[.2,.15,.1,.05,.08];col='#00ccee';gl='rgba(0,204,238,.15)';}
  cx.beginPath();
  for(let x=0;x<w;x++){let y=0;const t=(x/w)*8+wavePhase;for(let f=0;f<fr.length;f++)y+=Math.sin(t*fr[f]+f*.7)*am[f];y+=(Math.random()-.5)*.03;const py=mid+y*sc;x===0?cx.moveTo(x,py):cx.lineTo(x,py);}
  cx.strokeStyle=gl;cx.lineWidth=6;cx.stroke();
  cx.beginPath();
  for(let x=0;x<w;x++){let y=0;const t=(x/w)*8+wavePhase;for(let f=0;f<fr.length;f++)y+=Math.sin(t*fr[f]+f*.7)*am[f];y+=(Math.random()-.5)*.02;const py=mid+y*sc;x===0?cx.moveTo(x,py):cx.lineTo(x,py);}
  cx.strokeStyle=col;cx.lineWidth=1.5;cx.lineJoin='round';cx.stroke();
  cx.beginPath();cx.moveTo(0,mid);cx.lineTo(w,mid);cx.strokeStyle='rgba(0,180,240,.06)';cx.lineWidth=1;cx.stroke();
  wavePhase+=.06;requestAnimationFrame(drawW);
}drawW();

function arc(svg,v,mx,col){
  const cx=50,cy=52,r=38,s=Math.PI*.8,e=Math.PI*2.2,rng=e-s,pct=Math.min(v/mx,1);
  const pt=a=>`${cx+r*Math.cos(a)},${cy+r*Math.sin(a)}`;
  const bg=`M${pt(s)} A${r},${r} 0 1,1 ${pt(e)}`;
  const va=s+rng*pct,lg=pct>.5?1:0;
  const vd=pct>0?`M${pt(s)} A${r},${r} 0 ${lg},1 ${pt(va)}`:'';
  svg.innerHTML=`<path d="${bg}" fill="none" stroke="#122040" stroke-width="5" stroke-linecap="round"/>`+(vd?`<path d="${vd}" fill="none" stroke="${col}" stroke-width="5" stroke-linecap="round" style="filter:drop-shadow(0 0 3px ${col})"/>`:'');
}

function drawH(){
  const svg=document.getElementById('hy'),w=600,n=hist.stages.length;if(n<2)return;
  const xs=w/Math.max(n,360);let h='';
  for(const[k,y] of Object.entries(SY)){h+=`<line x1="28" y1="${y}" x2="${w}" y2="${y}" stroke="rgba(0,160,220,.06)"/><text x="0" y="${y+3}" fill="#384860" font-size="8" font-family="JetBrains Mono,monospace">${k[0]}</text>`;}
  let d='';for(let i=0;i<n;i++){const x=28+i*xs,y=SY[hist.stages[i]]||15;d+=(i?'L':'M')+x+','+y;}
  const col=SC[hist.stages[n-1]]||'#ddaa22';
  h+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round" style="filter:drop-shadow(0 0 2px ${col})"/>`;
  const cx=28+(n-1)*xs,cy=SY[hist.stages[n-1]];
  h+=`<circle cx="${cx}" cy="${cy}" r="4" fill="${col}" stroke="#0a1020" stroke-width="2"/>`;
  svg.innerHTML=h;
}

function onMsg(d){
  const cb=document.getElementById('connBar');cb.classList.add('ok');cb.textContent='● LIVE — RECEIVING DATA';
  waveStage=d.stage||'WAKE';
  hist.stages.push(d.stage);if(d.hr)hist.hr.push(d.hr);if(d.cam!=null)hist.cam.push(d.cam);if(d.acc!=null)hist.acc.push(d.acc);
  document.getElementById('el').textContent=d.elapsed||'—';
  const sb=document.getElementById('sb');sb.textContent=d.stage;sb.style.color=SC[d.stage]||'#ddaa22';sb.style.textShadow=`0 0 16px ${SG[d.stage]||''}`;
  document.getElementById('ss').textContent=d.stage_sub||'';
  if(d.hr){document.getElementById('hrV').textContent=Math.round(d.hr);arc(document.getElementById('hrG'),d.hr,120,'#00dd77');}
  if(d.cam!=null){document.getElementById('camV').textContent=d.cam.toFixed(4);arc(document.getElementById('camG'),Math.min(d.cam/.05,1)*100,100,'#00ccee');}
  if(d.acc!=null){document.getElementById('accV').textContent=d.acc.toFixed(4);arc(document.getElementById('accG'),Math.min(d.acc/.3,1)*100,100,'#ddaa22');}
  if(d.bp){const mx=.6;for(const[k,v]of Object.entries(d.bp)){const id=k[0].toUpperCase();const f=document.getElementById('f'+id),vv=document.getElementById('v'+id);if(f)f.style.width=Math.min(v,mx)/mx*100+'%';if(vv)vv.textContent=(v*100).toFixed(0)+'%';}}
  if(d.dist){document.getElementById('sW').textContent=d.dist.WAKE||'—';document.getElementById('sN').textContent=d.dist.NREM||'—';document.getElementById('sR').textContent=d.dist.REM||'—';}
  drawH();
  if(d.rbd_event){
    evtN++;document.getElementById('ec').textContent=evtN;
    const ne=document.getElementById('noevt');if(ne)ne.remove();
    const c=document.createElement('div');c.className='evt-card';
    c.innerHTML=`<div class="evt-time">${d.rbd_event.time||''}</div><div class="evt-detail">${d.rbd_event.detail||''}</div>`;
    document.getElementById('evts').prepend(c);
    document.getElementById('af').classList.add('on');setTimeout(()=>document.getElementById('af').classList.remove('on'),800);
  }
}

function connect(){
  ws=new WebSocket('ws://'+location.hostname+':'+WS_PORT);
  ws.onmessage=e=>{try{onMsg(JSON.parse(e.data));}catch(x){}};
  ws.onclose=()=>{document.getElementById('connBar').classList.remove('ok');document.getElementById('connBar').textContent='DISCONNECTED — RECONNECTING...';setTimeout(connect,2000);};
  ws.onerror=()=>{};
}
connect();
</script>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Camera / config
# ---------------------------------------------------------------------------
def load_tapo_conf(path="tapo_conf.json"):
    with open(path) as f:
        return json.load(f)


def make_cam(conf):
    from eeg_companion.tapo_cam import TapoCam
    host = conf["host"]
    if host == "auto":
        from eeg_companion.tapo_discover import find_tapo
        print("[Monitor] 自動搜尋 Tapo 攝影機...")
        result = find_tapo(conf["user"], conf["password"])
        if result is None:
            print("[Monitor] 找不到攝影機，跳過影像功能")
            return None
        host = result["host"]
        print(f"[Monitor] 使用攝影機: {host}")
    cam = TapoCam(host, conf["user"], conf["password"],
                  cloud_password=conf.get("cloud_password", ""))
    cam.connect()
    return cam


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _run_loop(source, cam, args):
    from eeg_companion.raw_recorder import RawRecorder

    det = RBDDetector(
        motion_threshold=args.motion_threshold,
        min_rem_epochs=args.min_rem,
        cooldown_sec=args.cooldown,
    )

    ts = time.strftime('%Y%m%d_%H%M%S')
    log_path = args.log or f"rbd_log_{ts}.jsonl"
    screenshot_dir = f"rbd_screenshots_{ts}"
    os.makedirs(screenshot_dir, exist_ok=True)

    rec = RawRecorder(subject_id=args.subject, out_dir=".")

    dash = None
    if args.dashboard:
        dash = DashboardServer(ws_port=args.ws_port, http_port=args.http_port)
        dash.start()

    audio_mon = None
    if getattr(args, 'mic', False):
        try:
            from eeg_companion.audio_monitor import AudioMonitor
            rtsp = None
            if cam is not None:
                try:
                    rtsp = cam.stream_url(hd=False)
                    print(f"[Monitor] 音訊來源: 攝影機 RTSP")
                except Exception:
                    pass
            audio_mon = AudioMonitor(
                device=args.mic_device,
                rtsp_url=rtsp,
            )
            audio_mon.start()
        except ImportError as e:
            print(f"[Monitor] 音訊監測無法啟用: {e}")

    print(f"[Monitor] 開始監測，log → {log_path}")
    print(f"[Monitor] 截圖 → {screenshot_dir}/")
    print(f"[Monitor] 原始 EEG 結束後存 EDF")
    if not args.no_audio:
        print(f"[Monitor] 聲音警報已啟用")
    if audio_mon:
        print(f"[Monitor] 麥克風音訊監測已啟用")
    print("-" * 60)

    import json as _json
    log_f = open(log_path, "a")
    stage_start = 0
    prev_stage = None
    stage_counts = {"WAKE": 0, "NREM": 0, "REM": 0}

    try:
        for win in source.windows():
            t = win["t"]

            # Record raw EEG
            rec.feed(win)

            # Camera motion
            video_motion = None
            if cam is not None:
                try:
                    video_motion = cam.motion_score()
                except Exception as e:
                    if int(t) % 30 == 0:
                        print(f"\n  [cam warn] {e}")

            event = det.process(win, video_motion=video_motion)

            stage = det.stage_log[-1][1] if det.stage_log else SleepStage.UNKNOWN
            stage_val = stage.value
            if stage_val != prev_stage:
                stage_start = t
                prev_stage = stage_val
            stage_counts[stage_val] = stage_counts.get(stage_val, 0) + 1

            rem_n = det._rem_count
            status = (f"\r  t={t:6.0f}s  {stage_val:5s}  REM連續={rem_n}")
            if video_motion is not None:
                status += f"  cam={video_motion:.3f}"
            print(status, end="", flush=True)

            # Prepare record
            record = {"t": t, "stage": stage_val, "video_motion": video_motion}

            # Bandpower from last stager features
            bp_data = None
            if det.stager._history:
                feat = det.stager._history[-1]
                bp_data = {k.replace("rel_", ""): feat[k]
                           for k in feat if k.startswith("rel_")}

            # RBD event handling
            rbd_evt_data = None
            if event:
                print(f"\n  ⚠️  RBD EVENT  t={t:.0f}s "
                      f"conf={event.confidence:.2f} {event.detail}")

                # Audio alert
                if not args.no_audio:
                    _alert_sound("critical")
                    _alert_speak(f"偵測到RBD事件，信心度{int(event.confidence*100)}%")

                # Screenshot
                screenshot_path = None
                if cam is not None:
                    screenshot_path = _save_screenshot(cam, t, screenshot_dir)
                    if screenshot_path:
                        print(f"  📸 截圖: {screenshot_path}")

                record["rbd_event"] = {
                    "confidence": event.confidence,
                    "combined_motion": event.combined_motion,
                    "detail": event.detail,
                    "screenshot": screenshot_path,
                }
                rbd_evt_data = {
                    "time": f"{int(t//3600)}h{int((t%3600)//60):02d}m",
                    "detail": event.detail,
                    "confidence": event.confidence,
                }

            # Audio monitoring
            audio_data = None
            if audio_mon:
                af = audio_mon.current
                audio_data = {
                    "event": af.event.value,
                    "rms_db": af.rms_db,
                    "centroid": af.spectral_centroid,
                    "periodicity": af.periodicity,
                }
                record["audio"] = audio_data
                if af.event.value == "SCREAM" and not args.no_audio:
                    _alert_sound("critical")
                    _alert_speak("偵測到大叫事件")

            log_f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            log_f.flush()

            # Push to dashboard
            if dash:
                total = sum(stage_counts.values()) or 1
                dist = {k: f"{v/total*100:.0f}%" for k, v in stage_counts.items()}
                dash.broadcast({
                    "t": t,
                    "elapsed": f"{int(t//3600):02d}:{int((t%3600)//60):02d}:{int(t%60):02d}",
                    "stage": stage_val,
                    "stage_sub": f"elapsed {int(t//3600)}h{int((t%3600)//60):02d}m · stage {int(t - stage_start)}s",
                    "hr": getattr(det, '_last_hr', None),
                    "cam": video_motion,
                    "acc": win.get("activity"),
                    "bp": bp_data,
                    "dist": dist,
                    "rem_idx": det.stager._history[-1].get("rem_idx") if det.stager._history else None,
                    "muscle_tone": det.stager._history[-1].get("muscle_tone") if det.stager._history else None,
                    "rbd_event": rbd_evt_data,
                    "audio": audio_data,
                })

    except KeyboardInterrupt:
        print("\n\n[Monitor] Ctrl+C — 停止監測")
    finally:
        source.close()
        if cam is not None:
            cam.close()
        if audio_mon is not None:
            audio_mon.stop()
        log_f.close()

    # Save raw EEG
    edf_path = rec.save()

    print("\n" + "=" * 60)
    summary = det.summary()
    print(f"總 epoch: {summary['total_epochs']}")
    print(f"分期: {summary['stage_distribution']}")
    print(f"RBD 事件: {summary['rbd_events']}")
    for e in summary["events"]:
        print(f"  t={e['t']:.0f}s  conf={e['confidence']:.2f}  {e['detail']}")
    if audio_mon:
        asummary = audio_mon.summary()
        print(f"打呼事件: {asummary['snore_episodes']}")
        print(f"夢囈事件: {asummary['talk_episodes']}")
        print(f"大叫事件: {asummary['scream_episodes']}")
    print(f"Log: {log_path}")
    if edf_path:
        print(f"EDF: {edf_path}")
    print(f"截圖: {screenshot_dir}/")


def run_replay(bin_path, cam, args):
    from eeg_companion.h3 import H3ReplaySource
    src = H3ReplaySource(bin_path, win_sec=4.0, step_sec=1.0,
                         loops=args.loops, speed=args.speed)
    print(f"[Replay] {src.stats[0]} blocks, {src.stats[1]} checksum errors")
    _run_loop(src, cam, args)


def run_live(cam, args):
    from eeg_companion.h3 import H3LiveSource
    src = H3LiveSource(args.serial, win_sec=4.0, step_sec=1.0)
    _run_loop(src, cam, args)


def main():
    ap = argparse.ArgumentParser(description="RBD 整夜監測")
    ap.add_argument("--cam", action="store_true", help="啟用 Tapo 攝影機")
    ap.add_argument("--cam-conf", default="tapo_conf.json", help="攝影機設定檔")
    ap.add_argument("--replay", type=str, help="H3 錄製檔(.bin)路徑")
    ap.add_argument("--serial", type=str, help="H3 藍牙序號(即時模式)")
    ap.add_argument("--subject", type=str, default="S001", help="受試者編號")
    ap.add_argument("--loops", type=int, default=1, help="replay 重播次數")
    ap.add_argument("--speed", type=float, default=0.0, help="replay 播放速度(0=全速)")
    ap.add_argument("--motion-threshold", type=float, default=1.5,
                    help="動作偵測閾值")
    ap.add_argument("--min-rem", type=int, default=3,
                    help="觸發 RBD 前需連續幾個 REM epoch")
    ap.add_argument("--cooldown", type=float, default=30.0,
                    help="兩次 RBD 事件最短間隔(秒)")
    ap.add_argument("--log", type=str, help="log 檔路徑")
    ap.add_argument("--dashboard", action="store_true", help="啟動即時面板")
    ap.add_argument("--http-port", type=int, default=8080, help="面板 HTTP port")
    ap.add_argument("--ws-port", type=int, default=8765, help="面板 WebSocket port")
    ap.add_argument("--no-audio", action="store_true", help="關閉聲音警報")
    ap.add_argument("--mic", action="store_true", help="啟用麥克風音訊監測")
    ap.add_argument("--mic-device", type=int, default=None,
                    help="麥克風裝置編號 (sounddevice)")
    args = ap.parse_args()

    cam = None
    if args.cam:
        conf = load_tapo_conf(args.cam_conf)
        cam = make_cam(conf)

    if args.replay:
        run_replay(args.replay, cam, args)
    elif args.serial:
        run_live(cam, args)
    else:
        print("RBD 整夜睡眠監測")
        print("=" * 40)
        print("用法:")
        print("  python rbd_monitor.py --replay data.bin --dashboard")
        print("  python rbd_monitor.py --serial <H3位址> --cam --dashboard")
        print()
        print("選項:")
        print("  --cam          啟用 Tapo 攝影機")
        print("  --dashboard    開啟即時面板 (http://localhost:8080)")
        print("  --subject S001 受試者編號")
        print("  --no-audio     關閉聲音警報")
        sys.exit(1)


if __name__ == "__main__":
    main()
