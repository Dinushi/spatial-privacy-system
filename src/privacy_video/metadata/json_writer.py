from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JSONWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: Dict[str, Any]) -> None:
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)