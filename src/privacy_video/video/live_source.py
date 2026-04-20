from __future__ import annotations

import time
import cv2

from models import FramePacket, VideoInfo
from video.video_source_base import VideoSource

class LiveVideoSource(VideoSource):
    """
    Basic live source backed by cv2.VideoCapture.

    `source` can be:
      - 0, 1, ... for local webcams
      - a stream URL later
    """

    def __init__(self, source: int | str = 0) -> None:
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._video_info: VideoInfo | None = None
        self._frame_idx: int = 0
        self._start_time: float | None = None
    
    @property
    def is_live(self) -> bool:
        return True
    
    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open live source: {self.source}")
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        if fps <= 0:
            fps = 30.0

        self._video_info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=None,
            is_live=True,
        )
        self._frame_idx = 0
        self._start_time = time.time()

    def get_infor(self) -> VideoInfo:
        if self._video_info is None:
            raise RuntimeError("LiveVideoSource is not opened.")
        return self._video_info
    
    def read(self) -> tuple[bool, FramePacket | None]:
        if self._cap is None or self._video_info is None or self._start_time is None:
            raise RuntimeError("LiveVideoSource is not opened.")
        
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        
        timestamp_sec = time.time() - self._start_time
        packet = FramePacket(
            frame_idx=self._frame_idx,
            timestamp_sec=timestamp_sec,
            image=frame
        )
        self._frame_idx += 1
        return True, packet
    
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None