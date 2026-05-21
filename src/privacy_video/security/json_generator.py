from datetime import datetime, timezone
from typing import Dict, Any
import json


def generate_metadata_json(
    media_id: str,
    media_type: str,
    source_path: str,
    blurred_output_path: str,
    fps: float | None,
    width: int | None,
    height: int | None,
    total_frames: int | None,
    video_stride: int | None,
    seen_labels: set[str],
    framesIDs_per_label: Dict[str, set[int]],
    regionIDs_per_label: Dict[str, list[str]],
    sam_total_time: float,
    post_processing_total_time: float,
) -> Dict[str, Any]:

    return {
        "version": 1,
        "media_id": media_id,
        "media_type": media_type,
        "source_path": source_path,
        "blurred_output_path": blurred_output_path,
        "created_at": datetime.now(timezone.utc).isoformat(),

        "video_info": {
            "fps": fps if media_type == "video" else None,
            "width": width if media_type == "video" else None,
            "height": height if media_type == "video" else None,
            "total_frames": total_frames if media_type == "video" else None,
            "video_stride": video_stride if media_type == "video" else None,
        },
        # pick the list of protected labels from this structure
        "protected_labels": sorted(list(seen_labels)),

        # pick the list of regions related to each of the lable and also relevant frame IDs 
        # (if you only get even region IDs, you can lookup the encrypted private regions file to get the region wise entries by passing the regionID and the region entry has the frameID at which it belongs)
        "objects": [
            {
                "label": label,
                "key_id": f"key_{label}",
                "frames": sorted(list(framesIDs_per_label[label])),
                "region_ids": regionIDs_per_label[label],
            }
            for label in sorted(seen_labels)
        ],
        # TODO: this is not essential : may be remove later
        "timing": {
            "sam_time_seconds": round(sam_total_time, 2),
            "post_processing_time_seconds": round(post_processing_total_time, 2),
            "total_time_seconds": round(
                sam_total_time + post_processing_total_time,
                2,
            ),
        },
    }

def build_final_embedding_payload(
        media_id: str,
        metadata: Dict[str, Any],
        encrypted_private_regions: Dict[str, Any],
        encrypted_key_registry: Dict[str, Any],
    ) -> bytes:

        payload = {
            "version": 1,
            "media_id": media_id,

            "payloads": {
                "metadata": metadata,
                "encrypted_private_regions": encrypted_private_regions,
                "encrypted_key_registry": encrypted_key_registry,
            },
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
