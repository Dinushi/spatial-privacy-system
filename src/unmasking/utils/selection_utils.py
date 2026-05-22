from __future__ import annotations

from typing import Any, Dict, List, Set


def find_metadata_objects_for_labels(
    metadata: Dict[str, Any],
    approved_labels: List[str],
) -> List[Dict[str, Any]]:
    approved_set = set(approved_labels)

    return [
        obj for obj in metadata.get("objects", [])
        if obj.get("label") in approved_set
    ]


def collect_region_ids_for_labels(
    metadata: Dict[str, Any],
    approved_labels: List[str],
) -> Set[str]:
    selected_objects = find_metadata_objects_for_labels(
        metadata=metadata,
        approved_labels=approved_labels,
    )

    region_ids: Set[str] = set()

    for obj in selected_objects:
        for region_id in obj.get("region_ids", []):
            region_ids.add(region_id)

    return region_ids


def collect_frame_ids_for_labels(
    metadata: Dict[str, Any],
    approved_labels: List[str],
) -> Set[int]:
    selected_objects = find_metadata_objects_for_labels(
        metadata=metadata,
        approved_labels=approved_labels,
    )

    frame_ids: Set[int] = set()

    for obj in selected_objects:
        for frame_id in obj.get("frames", []):
            frame_ids.add(int(frame_id))

    return frame_ids


def find_region_entries_by_ids(
    encrypted_private_regions: Dict[str, Any],
    selected_region_ids: Set[str],
) -> List[Dict[str, Any]]:
    all_regions = encrypted_private_regions.get("encrypted_private_regions", [])

    return [
        region for region in all_regions
        if region.get("region_id") in selected_region_ids
    ]


def find_wrapped_label_key_entry(
    decrypted_key_registry: Dict[str, Any],
    label: str,
) -> Dict[str, Any]:
    for entry in decrypted_key_registry.get("wrapped_label_keys", []):
        if entry.get("label") == label:
            return entry

    raise KeyError(f"No wrapped AES key found for label: {label}")