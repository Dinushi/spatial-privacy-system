from __future__ import annotations

from abc import ABC, abstractmethod

from privacy_video.models import FramePacket, VideoInfo


class VideoSource(ABC):
    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Whether the source is a live stream."""
        raise NotImplementedError

    @abstractmethod
    def open(self) -> None:
        """Open the source."""
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> VideoInfo:
        """Return source metadata."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> tuple[bool, FramePacket | None]:
        """
        Read one frame.

        Returns:
            (True, FramePacket) if a frame was read successfully
            (False, None) when no more frames are available or capture failed
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the source and release resources."""
        raise NotImplementedError