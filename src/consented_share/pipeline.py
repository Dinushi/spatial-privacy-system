from pathlib import Path
from typing import Any, Dict

from consented_share.payload_loader import load_payloads_from_files
from consented_share.object_matcher import (
    print_available_objects,
    find_matching_objects,
)
from consented_share.decryptor import (
    decrypt_metadata,
    find_registry_entry,
    unwrap_object_key,
    find_region_entries,
    decrypt_region_crop,
)
from consented_share.image_reconstructor import save_reconstructed_image


def run_consented_share_pipeline(
    output_root: str | Path,
    private_key_path: str | Path,
    request_text: str,
    placement_mode: str = "mask",
    output_image_path: str | Path | None = None,
) -> Dict[str, Any]:
    
    if placement_mode not in {"bbox", "mask"}:
        raise ValueError("placement_mode must be either 'bbox' or 'mask'")
    
    output_root = Path(output_root)
    blurred_image_path = output_root / "blurred_output.png"

    if output_image_path is None:
        output_image_path = output_root / "consented_reveal_output.png"

    print("\nReading encrypted payload files...")
    payloads = load_payloads_from_files(output_root)

    encrypted_metadata_file = payloads["encrypted_metadata_file"]
    region_package = payloads["region_package"]
    object_key_registry = payloads["object_key_registry"]

    print("\nDecrypting metadata...")
    metadata_plain = decrypt_metadata(
        encrypted_metadata_file=encrypted_metadata_file,
        private_key_path=str(private_key_path),
    )

    print_available_objects(metadata_plain)

    print(f"\nConsent request: {request_text}")
    matching_objects = find_matching_objects(metadata_plain, request_text)

    if not matching_objects:
        print("\nNo matching hidden object found for this request.")
        return {
            "status": "not_found",
            "request_text": request_text,
            "output_image_path": None,
        }

    print("\nMatching object(s) found:")
    for obj in matching_objects:
        print(
            f"- object_id={obj['object_id']} | "
            f"label={obj.get('label')} | "
            f"region_ids={obj.get('region_ids')}"
        )

    all_decrypted_regions = []

    for obj in matching_objects:
        object_id = obj["object_id"]
        region_ids = obj.get("region_ids", [])

        print(f"\nFinding symmetric key for object_id={object_id}")
        registry_entry = find_registry_entry(object_key_registry, object_id)

        print("Selected registry entry:")
        print({
            "object_id": registry_entry.get("object_id"),
            "wrap_alg": registry_entry.get("wrap_alg"),
            "allowed_region_ids": registry_entry.get("allowed_region_ids"),
            "wrapped_key_preview": str(registry_entry.get("wrapped_key", ""))[:60] + "...",
        })

        object_key = unwrap_object_key(
            registry_entry=registry_entry,
            private_key_path=str(private_key_path),
        )

        print(f"Object symmetric key unwrapped for object_id={object_id}")

        selected_region_entries = find_region_entries(region_package, region_ids)

        print(f"Found {len(selected_region_entries)} encrypted region(s).")

        for region_entry in selected_region_entries:
            print("\nSelected encrypted region package entry:")
            print({
                "region_id": region_entry.get("region_id"),
                "object_id": region_entry.get("object_id"),
                "frame_idx": region_entry.get("frame_idx"),
                "bbox": region_entry.get("bbox"),
                "encoding": region_entry.get("encoding"),
                "crypto": region_entry.get("crypto"),
                "ciphertext_preview": str(region_entry.get("ciphertext", ""))[:60] + "...",
            })

            crop = decrypt_region_crop(region_entry, object_key)
            all_decrypted_regions.append((region_entry, crop))

    print("\nReconstructing revealed image...")
    # saved_path = save_reconstructed_image(
    #     blurred_image_path=blurred_image_path,
    #     region_entries_with_crops=all_decrypted_regions,
    #     output_path=output_image_path,
    # )
    saved_path = save_reconstructed_image(
        blurred_image_path=blurred_image_path,
        region_entries_with_crops=all_decrypted_regions,
        output_path=output_image_path,
        placement_mode=placement_mode,
    )

    print(f"\nSaved consented reveal image: {saved_path}")

    return {
        "status": "success",
        "request_text": request_text,
        "matched_objects": matching_objects,
        "output_image_path": saved_path,
    }