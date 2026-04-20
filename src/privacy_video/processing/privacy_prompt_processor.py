from __future__ import annotations

from typing import List, Optional

from privacy_video.processing.base_processor import BaseProcessor


class PrivacyPromptProcessor(BaseProcessor):
    """
    For now returns hard-coded privacy-sensitive object prompts.
    Later this can be replaced with:
      - user-click selected prompts
      - scene-aware prompt generation
      - LLM-generated prompts
    """

    def __init__(self, prompts: Optional[List[str]] = None) -> None:
        self._prompts = prompts or [
            "Laptop screen",
            "Monitor Screen",
            "ceraVe Cream Bottle",
            "Voltren Inti-Inflammotory Cream Tube",
            "Coffe Table",
            "Brown cloth on sofa"
        ]

    def process(self) -> List[str]:
        return self._prompts