from __future__ import annotations

import argparse
from pathlib import Path
import time

from unmasking.payload_extraction_pipeline import run_unmasking_payload_extraction_pipeline
from unmasking.reveal_pipeline import reveal_approved_labels_to_media


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract embedded encrypted privacy payloads from protected media"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to protected image/video containing embedded privacy payload",
    )
    parser.add_argument(
        "--private-key",
        default="../keys/device_private.pem",
        help="Path to device RSA private key PEM file",
    )

    parser.add_argument(
        "--copy-src-media",
        action="store_true",
        dest="copy_source_media",
        help="Copy the original protected media also into the output folder",
    )

    parser.set_defaults(copy_source_media=True)

    return parser

def prompt_user_consent_per_label(
    app_name: str,
    requested_labels: list[str],
) -> list[str]:

    print("\n" + "=" * 60)
    print("CONSENT REQUEST")
    print("=" * 60)

    print(f"\n{app_name} requests access to the following protected content:\n")

    approved_labels = []

    for idx, label in enumerate(requested_labels, start=1):

        print(f"{idx}. {label}")
        print(f"   Purpose: SAMPLE_PURPOSE_{idx}")

        while True:
            choice = input(
                "\nGrant access? [Y] Yes / [N] No : "
            ).strip().lower()

            if choice in ["y", "yes"]:
                approved_labels.append(label)
                print(f"Access GRANTED for: {label}\n")
                break

            elif choice in ["n", "no"]:
                print(f"Access DENIED for: {label}\n")
                break

            else:
                print("Invalid input. Please enter Y or N.")

    print("=" * 60)

    print("\nFinal approved labels:")
    if approved_labels:
        for label in approved_labels:
            print(f"- {label}")
    else:
        print("- None")

    return approved_labels

def prompt_to_collect_app_requested_labels(labels: list[str]) -> list[str]:
    if not labels:
        print("\nNo protected labels available for selection.")
        return []

    print("\nThink as the AI app running in the smart-glass, what labels do you need to provide your services?")
    print("Select one or more labels by number, separated by commas.")
    print("Example: 1,3")

    for idx, label in enumerate(labels, start=1):
        print(f"{idx}. {label}")

    while True:
        user_input = input("\nEnter label number(s): ").strip()

        if not user_input:
            print("Please enter at least one number.")
            continue

        try:
            selected_indices = [
                int(x.strip())
                for x in user_input.split(",")
                if x.strip()
            ]

            invalid = [
                idx for idx in selected_indices
                if idx < 1 or idx > len(labels)
            ]

            if invalid:
                print(f"Invalid selection(s): {invalid}")
                print(f"Please enter numbers between 1 and {len(labels)}.")
                continue

            selected_labels = [
                labels[idx - 1]
                for idx in selected_indices
            ]

            # remove duplicates while preserving order
            selected_labels = list(dict.fromkeys(selected_labels))

            print("\nAI app requested access to:")
            for label in selected_labels:
                print(f"- {label}")

            return selected_labels

        except ValueError:
            print("Invalid input. Please enter numbers only, separated by commas.")

def main() -> None:
    start_time = time.time()

    parser = build_parser()
    args = parser.parse_args()

    source_path = Path(args.source)

    # create unmasked_i inside same folder as source media
    parent_folder = source_path.parent

    i = 1
    while (parent_folder / f"unmasked_{i}").exists():
        i += 1

    unmasked_output_folder = parent_folder / f"unmasked_{i}"

    result = run_unmasking_payload_extraction_pipeline(
        source_path=args.source,
        output_root=unmasked_output_folder,
        copy_source_media=args.copy_source_media,
    )

    total_sec = time.time() - start_time

    print("\nUnmasking payload extraction completed.")
    print(f"Media ID: {result['media_id']}")
    print(f"Labels: {result['labels']}")
    print(f"Total Time (seconds): {total_sec:.2f}")

    print("\n============== Modeling Consent and Fine-grained sharing==============")

    requested_labels = prompt_to_collect_app_requested_labels(result["labels"])

    approved_labels = prompt_user_consent_per_label(app_name="App 1", requested_labels=requested_labels)
    print(f"\nApproved labels: {approved_labels}")

    reveal_result = reveal_approved_labels_to_media(
        protected_media_path=args.source,
        output_root=unmasked_output_folder,
        approved_labels=approved_labels,
        metadata=result["metadata"],
        encrypted_private_regions=result["encrypted_private_regions"],
        encrypted_aes_key_registry=result["encrypted_aes_key_registry"],
        private_key_path=args.private_key,
    )

    print("\nFine-grained reveal completed.")
    print(f"Reveal status: {reveal_result['status']}")
    print(f"Reveal output: {reveal_result['output_path']}")

if __name__ == "__main__":
    main()

# python3 main_unmasking.py --source ../data/output/predict_1/blurred_output.png
# python3 main_unmasking.py --source ../data/output/predict_2/blurred_output.mp4