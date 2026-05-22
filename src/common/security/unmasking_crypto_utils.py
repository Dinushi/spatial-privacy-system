from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def load_private_key(private_key_path: str | Path):
    with open(private_key_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
        )


def b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("utf-8"))


def rsa_unwrap_key(private_key, wrapped_key_b64: str) -> bytes:
    wrapped_key = b64decode(wrapped_key_b64)

    return private_key.decrypt(
        wrapped_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def aes_gcm_decrypt_from_entry(
    key: bytes,
    ciphertext_b64: str,
    crypto: Dict[str, Any],
) -> bytes:
    nonce = b64decode(crypto["nonce"])
    tag = b64decode(crypto["tag"])
    ciphertext = b64decode(ciphertext_b64)

    aesgcm = AESGCM(key)

    # cryptography AESGCM expects ciphertext || tag
    return aesgcm.decrypt(
        nonce,
        ciphertext + tag,
        associated_data=None,
    )


def decrypt_hybrid_json(
    encrypted_payload: Dict[str, Any],
    private_key_path: str | Path,
) -> Dict[str, Any]:
    private_key = load_private_key(private_key_path)

    aes_key = rsa_unwrap_key(
        private_key=private_key,
        wrapped_key_b64=encrypted_payload["wrapped_key"],
    )

    plaintext_bytes = aes_gcm_decrypt_from_entry(
        key=aes_key,
        ciphertext_b64=encrypted_payload["ciphertext"],
        crypto=encrypted_payload["crypto"],
    )

    return json.loads(plaintext_bytes.decode("utf-8"))


def unwrap_label_aes_key(
    wrapped_key_b64: str,
    private_key_path: str | Path,
) -> bytes:
    private_key = load_private_key(private_key_path)
    return rsa_unwrap_key(private_key, wrapped_key_b64)