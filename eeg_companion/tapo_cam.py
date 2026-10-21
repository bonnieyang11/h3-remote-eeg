"""Tapo C210 攝影機整合：截圖、動作偵測、RTSP 串流。

優先用 RTSP 直連（最穩定），pytapo API 作為備用。

用法:
    cam = TapoCam("192.168.1.153", "Bonnie801101", "qwerty123!")
    cam.connect()
    frame = cam.snapshot()         # 單張截圖 (numpy BGR)
    score = cam.motion_score()     # 與上一幀比較的動作量 (0~1)
    url   = cam.stream_url()       # RTSP 串流位址
"""
from __future__ import annotations
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class TapoCam:
    def __init__(self, host: str, user: str, password: str,
                 cloud_password: str | None = None):
        self.host = host
        self.user = user
        self.password = password
        self.cloud_password = cloud_password
        self._cap = None
        self._prev_gray = None
        self._stream_url: str | None = None
        self._connected = False

    def connect(self):
        if cv2 is None:
            raise ImportError("pip install opencv-python-headless")

        # RTSP 直連 — 不依賴 pytapo API
        urls = [
            f"rtsp://{self.user}:{self.password}@{self.host}:554/stream1",
            f"rtsp://{self.user}:{self.password}@{self.host}:554/stream2",
        ]
        for url in urls:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    stream_label = "HD" if w > 1000 else "SD"
                    print(f"[TapoCam] RTSP 連線成功: {w}x{h} ({stream_label})")
                    self._stream_url = url
                    self._cap = cap
                    self._connected = True
                    return {"resolution": f"{w}x{h}", "stream": stream_label}
                cap.release()

        raise RuntimeError(
            f"無法連線攝影機 {self.host} — 確認帳密和網路")

    def stream_url(self, hd: bool = True) -> str:
        if self._stream_url:
            stream = "stream1" if hd else "stream2"
            base = f"rtsp://{self.user}:{self.password}@{self.host}:554/{stream}"
            return base
        if not self._connected:
            self.connect()
        return self._stream_url

    def _open_capture(self):
        if cv2 is None:
            raise ImportError("pip install opencv-python-headless")
        if self._cap is not None and self._cap.isOpened():
            return
        if not self._connected:
            self.connect()
            return
        self._cap = cv2.VideoCapture(self._stream_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"無法開啟串流: {self._stream_url}")

    def snapshot(self) -> np.ndarray:
        self._open_capture()
        ret, frame = self._cap.read()
        if not ret:
            self._cap.release()
            self._cap = None
            self._open_capture()
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("擷取畫面失敗")
        return frame

    def motion_score(self) -> float:
        """與前一幀比較的動作量，歸一化到 0~1。"""
        frame = self.snapshot()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        score = float(np.mean(diff) / 255.0)
        return score

    def motion_regions(self, threshold: int = 25, min_area: int = 500):
        """回傳動作區域的 bounding boxes [(x,y,w,h), ...]。"""
        frame = self.snapshot()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return []
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                boxes.append(cv2.boundingRect(c))
        return boxes

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
