from __future__ import annotations

import argparse

from privacy_video.pipeline import copy_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Privacy-preserving video pipeline"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output video file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    copy_video(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()