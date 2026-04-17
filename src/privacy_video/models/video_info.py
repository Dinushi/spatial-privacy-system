from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int | None
    is_live : bool