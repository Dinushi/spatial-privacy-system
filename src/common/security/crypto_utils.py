from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def b64d(data: str) -> bytes:
    return base64.b64decode(data.encode("utf-8"))


def load_public_key(public_key_path: str | Path):
    data = Path(public_key_path).read_bytes()
    return serialization.load_pem_public_key(data)


def generate_aes256_key() -> bytes:
    return os.urandom(32)


def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes | None = None) -> Dict[str, str]:
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ct_with_tag = aes.encrypt(nonce, plaintext, aad)

    # AESGCM returns ciphertext || tag, with 16-byte tag at end
    ciphertext = ct_with_tag[:-16]
    tag = ct_with_tag[-16:]

    return {
        "alg": "AES-256-GCM",
        "nonce": b64e(nonce),
        "tag": b64e(tag),
        "ciphertext": b64e(ciphertext),
    }


def aes_gcm_decrypt(
    key: bytes,
    nonce_b64: str,
    tag_b64: str,
    ciphertext_b64: str,
    aad: bytes | None = None,
) -> bytes:
    nonce = b64d(nonce_b64)
    tag = b64d(tag_b64)
    ciphertext = b64d(ciphertext_b64)
    aes = AESGCM(key)
    return aes.decrypt(nonce, ciphertext + tag, aad)


def rsa_wrap_key(public_key, key: bytes) -> str:
    wrapped = public_key.encrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return b64e(wrapped)


def encrypt_json_with_rsa(public_key, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    For small payloads only. Your metadata file may grow, so hybrid encryption is better.
    This function is kept for completeness, but use encrypt_json_hybrid below.
    """
    plaintext = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ciphertext = public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "enc_alg": "RSA-OAEP-SHA256",
        "ciphertext": b64e(ciphertext),
    }


def encrypt_json_hybrid(public_key, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use AES-GCM for the JSON body and wrap that AES key with RSA.
    This is what you should use for encrypted metadata and wrapped key registries.
    """
    plaintext = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    content_key = generate_aes256_key()
    enc = aes_gcm_encrypt(content_key, plaintext)
    wrapped_key = rsa_wrap_key(public_key, content_key)

    return {
        "enc_alg": "HYBRID-RSA-OAEP-SHA256+AES-256-GCM",
        "wrapped_key": wrapped_key,
        "crypto": {
            "alg": enc["alg"],
            "nonce": enc["nonce"],
            "tag": enc["tag"],
        },
        "ciphertext": enc["ciphertext"],
    }