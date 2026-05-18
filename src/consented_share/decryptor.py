from typing import Any, Dict, List
import base64
import cv2
import numpy as np

from common.security import (
    decrypt_json_hybrid,
    load_private_key,
    rsa_unwrap_key,
    aes_gcm_decrypt,
)


def decrypt_metadata(encrypted_metadata_file: Dict[str, Any], private_key_path: str) -> Dict[str, Any]:
    private_key = load_private_key(private_key_path)
    return decrypt_json_hybrid(private_key, encrypted_metadata_file)


def find_registry_entry(
    object_key_registry: Dict[str, Any],
    object_id: str,
) -> Dict[str, Any]:
    for entry in object_key_registry.get("wrapped_object_keys", []):
        if entry.get("object_id") == object_id:
            return entry

    raise ValueError(f"No symmetric key registry entry found for object_id={object_id}")


def unwrap_object_key(registry_entry: Dict[str, Any], private_key_path: str) -> bytes:
    private_key = load_private_key(private_key_path)
    return rsa_unwrap_key(private_key, registry_entry["wrapped_key"])


def find_region_entries(
    region_package: Dict[str, Any],
    region_ids: List[str],
) -> List[Dict[str, Any]]:
    selected = []

    for entry in region_package.get("encrypted_regions", []):
        if entry.get("region_id") in region_ids:
            selected.append(entry)

    return selected


def decrypt_region_crop(region_entry: Dict[str, Any], object_key: bytes) -> np.ndarray:
    crypto = region_entry["crypto"]


    plaintext_png = aes_gcm_decrypt(
        key=object_key,
        nonce_b64=crypto["nonce"],
        tag_b64=crypto["tag"],
        ciphertext_b64=region_entry["ciphertext"],
    )

    png_bytes = np.frombuffer(plaintext_png, dtype=np.uint8)
    crop = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)

    if crop is None:
        raise RuntimeError(f"Failed to decode decrypted crop for region {region_entry['region_id']}")

    return crop