from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from privacy_video.models import FramePacket, VideoInfo


class VideoWriter:
    def __init__(
        self,
        output_path: str | Path,
        video_info: VideoInfo,
        codec: str = "mp4v",
    ) -> None:
        self.output_path = Path(output_path)
        self.video_info = video_info
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.video_info.fps,
            (self.video_info.width, self.video_info.height),
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open output video: {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter is not opened.")
        self._writer.write(frame)

    def write_packet(self, packet: FramePacket) -> None:
        self.write(packet.image)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None