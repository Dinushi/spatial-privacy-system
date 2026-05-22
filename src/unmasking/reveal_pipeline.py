from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from common.security.unmasking_crypto_utils import (
    decrypt_hybrid_json,
    unwrap_label_aes_key,
    aes_gcm_decrypt_from_entry,
)
from unmasking.utils.crop_pasting_utils import decode_png_crop
from unmasking.utils.selection_utils import (
    collect_region_ids_for_labels,
    collect_frame_ids_for_labels,
    find_region_entries_by_ids,
    find_wrapped_label_key_entry,
)
from unmasking.reveal.reveal_reconstructor import (
    reconstruct_image_with_revealed_regions,
    reconstruct_video_with_revealed_regions,
)


def decrypt_selected_region_crops(
    selected_region_entries: List[Dict[str, Any]],
    label_to_aes_key: Dict[str, bytes],
) -> List[tuple[Dict[str, Any], Any]]:
    decrypted_regions = []

    for region_entry in selected_region_entries:
        label = region_entry["object_id"]

        if label not in label_to_aes_key:
            print(f"Skipping region {region_entry['region_id']} because label key is missing: {label}")
            continue

        plaintext_png = aes_gcm_decrypt_from_entry(
            key=label_to_aes_key[label],
            ciphertext_b64=region_entry["ciphertext"],
            crypto=region_entry["crypto"],
        )

        crop = decode_png_crop(plaintext_png)
        decrypted_regions.append((region_entry, crop))

    return decrypted_regions


def reveal_approved_labels_to_media(
    protected_media_path: str | Path,
    output_root: str | Path,
    approved_labels: List[str],
    metadata: Dict[str, Any],
    encrypted_private_regions: Dict[str, Any],
    encrypted_aes_key_registry: Dict[str, Any],
    private_key_path: str | Path,
) -> Dict[str, Any]:

    protected_media_path = Path(protected_media_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not approved_labels:
        print("\nNo labels approved. No reconstruction will be generated.")
        return {
            "status": "no_approved_labels",
            "approved_labels": [],
            "output_path": None,
        }

    print("\nFinding approved region IDs from metadata...")
    selected_region_ids = collect_region_ids_for_labels(
        metadata=metadata,
        approved_labels=approved_labels,
    )

    selected_frame_ids = collect_frame_ids_for_labels(
        metadata=metadata,
        approved_labels=approved_labels,
    )

    print(f"Approved labels: {approved_labels}")
    print(f"Selected region count: {len(selected_region_ids)}")
    print(f"Selected frame count: {len(selected_frame_ids)}")

    print("\nDecrypting AES key registry with device private key...")
    decrypted_key_registry = decrypt_hybrid_json(
        encrypted_payload=encrypted_aes_key_registry,
        private_key_path=private_key_path,
    )

    label_to_aes_key: Dict[str, bytes] = {}

    for label in approved_labels:
        wrapped_key_entry = find_wrapped_label_key_entry(
            decrypted_key_registry=decrypted_key_registry,
            label=label,
        )

        aes_key = unwrap_label_aes_key(
            wrapped_key_b64=wrapped_key_entry["wrapped_key"],
            private_key_path=private_key_path,
        )

        label_to_aes_key[label] = aes_key
        print(f"Unwrapped AES key for label: {label}")

    print("\nFinding encrypted region entries for approved labels...")
    selected_region_entries = find_region_entries_by_ids(
        encrypted_private_regions=encrypted_private_regions,
        selected_region_ids=selected_region_ids,
    )

    print(f"Found encrypted region entries: {len(selected_region_entries)}")

    print("\nDecrypting selected private region crops...")
    decrypted_regions = decrypt_selected_region_crops(
        selected_region_entries=selected_region_entries,
        label_to_aes_key=label_to_aes_key,
    )

    media_type = metadata.get("media_type")
    placement_mode = encrypted_private_regions.get("placement_mode", "mask")

    if media_type == "image":
        output_path = output_root / "consented_reveal_output.png"

        saved_path = reconstruct_image_with_revealed_regions(
            protected_image_path=protected_media_path,
            decrypted_regions=decrypted_regions,
            output_path=output_path,
            placement_mode=placement_mode,
        )

    elif media_type == "video":
        video_info = metadata.get("video_info", {})

        fps = float(video_info.get("fps") or 30.0)
        width = int(video_info["width"])
        height = int(video_info["height"])

        output_path = output_root / "consented_reveal_output.mp4"

        saved_path = reconstruct_video_with_revealed_regions(
            protected_video_path=protected_media_path,
            decrypted_regions=decrypted_regions,
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            placement_mode=placement_mode,
        )

    else:
        raise ValueError(f"Unsupported media_type in metadata: {media_type}")

    print(f"\nSaved reconstructed consented reveal media: {saved_path}")

    return {
        "status": "success",
        "approved_labels": approved_labels,
        "selected_region_ids": sorted(selected_region_ids),
        "selected_frame_ids": sorted(selected_frame_ids),
        "output_path": saved_path,
    }