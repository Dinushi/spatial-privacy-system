from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import shutil

from privacy_video.metadata.json_writer import JSONWriter
from privacy_video.security.media_embedder import MediaEmbedder


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    JSONWriter(path).write(payload)


def _print_available_object_labels(metadata: Dict[str, Any]) -> List[str]:
    labels = []

    # Your current metadata stores labels using seen_labels / regionIDs_per_label
    if "seen_labels" in metadata:
        labels = list(metadata.get("seen_labels", []))

    elif "objects" in metadata:
        labels = [
            obj.get("label") or obj.get("object_id")
            for obj in metadata.get("objects", [])
            if obj.get("label") or obj.get("object_id")
        ]

    elif "regionIDs_per_label" in metadata:
        labels = list(metadata.get("regionIDs_per_label", {}).keys())

    labels = sorted(set(labels))

    print("\nAvailable protected object labels in this scene:")
    if not labels:
        print("- No labels found in embedded metadata.")
    else:
        for label in labels:
            print(f"- {label}")

    return labels


def run_unmasking_payload_extraction_pipeline(
    source_path: str | Path,
    output_root: str | Path,
    copy_source_media: bool = True,
) -> Dict[str, Any]:
    """
    Extract embedded privacy payload from a protected image/video.

    Expected embedded payload structure:
    {
        "media_id": "...",
        "metadata": {...},
        "encrypted_private_regions": {...},
        "encrypted_key_registry": {...}
    }
    """

    source_path = Path(source_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Input media does not exist: {source_path}")

    print("\nReading embedded payload from media...")
    embedded_payload = MediaEmbedder.extract_payload_from_file(source_path)

    if embedded_payload is None:
        raise RuntimeError(
            "No embedded payload found in this media file. "
            "Make sure it was created using MediaEmbedder.embed_payload_in_file()."
        )

    if isinstance(embedded_payload, dict):
        payload = embedded_payload
    elif isinstance(embedded_payload, bytes):
        payload = json.loads(embedded_payload.decode("utf-8"))
    elif isinstance(embedded_payload, str):
        payload = json.loads(embedded_payload)
    else:
        raise TypeError(f"Unsupported payload type: {type(embedded_payload)}")

    media_id = payload.get("media_id", "unknown_media")
    payloads = payload.get("payloads", {})

    metadata = payloads.get("metadata", {})
    encrypted_private_regions = payloads.get("encrypted_private_regions", {})
    encrypted_aes_key_registry = payloads.get("encrypted_key_registry", {})

    print(f"Extracted payload for media_id: {media_id}")

    metadata_path = output_root / "metadata.json"
    encrypted_regions_path = output_root / "encrypted_private_regions.json"
    encrypted_key_registry_path = output_root / "AESkey_registry_encrypted.json"

    _save_json(metadata_path, metadata)
    _save_json(encrypted_regions_path, encrypted_private_regions)
    _save_json(encrypted_key_registry_path, encrypted_aes_key_registry)

    print("\nSaved extracted payload files:")
    print(f"- {metadata_path}")
    print(f"- {encrypted_regions_path}")
    print(f"- {encrypted_key_registry_path}")

    copied_media_path = None
    if copy_source_media:
        copied_media_path = output_root / source_path.name
        shutil.copy2(source_path, copied_media_path)
        print(f"- Copied protected media: {copied_media_path}")

    labels = _print_available_object_labels(metadata)

    return {
        "status": "success",
        "media_id": media_id,
        "labels": labels,
        "metadata": metadata,
        "encrypted_private_regions": encrypted_private_regions,
        "encrypted_aes_key_registry": encrypted_aes_key_registry,
    }