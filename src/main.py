from __future__ import annotations

import argparse
from pathlib import Path

from privacy_video.pipeline import run_privacy_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-preserving SAM pipeline")
    parser.add_argument("--source", required=True, help="Path to input image or video")

    parser.add_argument("--prompts", nargs="*", default=None, help="Optional prompt list")
    parser.add_argument("--model", default="../models/sam3.1_multiplex.pt", help="Path to SAM3 model")
    parser.add_argument("--output-root", default="../data/output", help="Directory to save outputs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    i = 1
    while (Path(args.output_root) / f"predict_{i}").exists():i += 1
    predict_output_folder = Path(args.output_root) / f"predict_{i}"

    metadata = run_privacy_pipeline(
        source_path=args.source,
        model_path=args.model,
        output_root=predict_output_folder,
        prompts=args.prompts,
    )

    print("Done.")
    print(f"Input type: {metadata['input_type']}")
    print(f"Blurred output: {metadata['blurred_output_path']}")


if __name__ == "__main__":
    main()

    # python3 main.py --source ../data/input/sample_img.jpeg
    # python3 main.py --source ../data/input/sample1.mp4