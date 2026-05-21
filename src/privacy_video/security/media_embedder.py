from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict


MAGIC = b"XRPRIV01"
LEN_FMT = ">Q"  # unsigned 64-bit big endian


class MediaEmbedder:
   
    @staticmethod
    def embed_payload_in_file(media_path: str | Path, payload_bytes: bytes) -> None:
        media_path = Path(media_path)
        with media_path.open("ab") as f:
            f.write(MAGIC)
            f.write(struct.pack(LEN_FMT, len(payload_bytes)))
            f.write(payload_bytes)

    @staticmethod
    def extract_payload_from_file(media_path: str | Path) -> Dict[str, Any]:
        media_path = Path(media_path)
        data = media_path.read_bytes()

        marker_pos = data.rfind(MAGIC)
        if marker_pos == -1:
            raise ValueError("No embedded privacy payload found.")

        header_pos = marker_pos + len(MAGIC)
        payload_len = struct.unpack(LEN_FMT, data[header_pos:header_pos + 8])[0]
        payload_start = header_pos + 8
        payload_end = payload_start + payload_len

        payload_bytes = data[payload_start:payload_end]
        return json.loads(payload_bytes.decode("utf-8"))