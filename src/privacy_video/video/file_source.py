from __future__ import annotations

from pathlib import Path
import cv2

from privacy_video.models import FramePacket, VideoInfo
from privacy_video.video.video_source_base import VideoSource


class FileVideoSource(VideoSource):
    def __init__(self, input_path: str | Path) -> None:
        self.input_path = Path(input_path)
        self._cap: cv2.VideoCapture | None = None
        self._video_info: VideoInfo | None = None
        self._frame_idx: int = 0

    @property
    def is_live(self) -> bool:
        return False

    def open(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_path}")

        self._cap = cv2.VideoCapture(str(self.input_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {self.input_path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0

        self._video_info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            is_live=False,
        )
        self._frame_idx = 0

    def get_info(self) -> VideoInfo:
        if self._video_info is None:
            raise RuntimeError("FileVideoSource is not opened.")
        return self._video_info

    def read(self) -> tuple[bool, FramePacket | None]:
        if self._cap is None or self._video_info is None:
            raise RuntimeError("FileVideoSource is not opened.")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None

        timestamp_sec = self._frame_idx / self._video_info.fps
        packet = FramePacket(
            frame_idx=self._frame_idx,
            timestamp_sec=timestamp_sec,
            image=frame,
        )
        self._frame_idx += 1
        return True, packet

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None