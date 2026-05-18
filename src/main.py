from __future__ import annotations

import argparse
from pathlib import Path
import time
import json
from pathlib import Path

from privacy_video.pipeline import run_privacy_pipeline


def load_prompts_from_annotation(annotation_file_path):
    """
    Read privacy annotation JSON and extract all labels as prompts.
    """

    with open(annotation_file_path, "r") as f:
        data = json.load(f)

    prompts = []

    privacy_categories = data.get("privacy_categories", {})

    for category_name, category_data in privacy_categories.items():
        labels = category_data.get("Labels", [])

        for label in labels:
            if label and isinstance(label, str):
                prompts.append(label.strip())

    # remove duplicates while preserving order
    prompts = list(dict.fromkeys(prompts))

    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-preserving SAM pipeline")
    parser.add_argument("--source", required=True, help="Path to input image or video")

    parser.add_argument("--prompts", nargs="*", default=None, help="Optional prompt list")
    parser.add_argument("--SAM-type", choices=["SAM3", "FastSAM"], default="SAM3", help="the SAM model processor that should be used for execution")
    parser.add_argument("--model", default="../models/sam3.1_multiplex.pt", help="Path to SAM3 model")
    parser.add_argument("--output-root", default="../data/output", help="Directory to save outputs")
    parser.add_argument("--crop-mode", choices=["mask", "bbox"], default="mask", help="Use mask or bbox for cropping/blur")
    parser.add_argument("--public-key", default="../keys/device_public.pem", help="Path to device RSA public key PEM file")
    parser.add_argument("--vid-stride", default=1, help="Video stride rate for frame skipping for efficiency")
    parser.add_argument("--embed-payloads", action="store_true", help="Embed encrypted payloads into final blurred media file") # this should be true most of the time
    return parser


def main() -> None:
    start_time = time.time()
    parser = build_parser()
    args = parser.parse_args()

    if (args.prompts == None):
        file_name = Path(args.source).stem
        annotation_file = (f"{file_name}_privacy_annotation.json") # need to make sure, the privacy annotation file always ends like this
        print(annotation_file)
        prompts = load_prompts_from_annotation( Path(Path(args.source).parent, annotation_file))
        print(f"Loaded privacy prompts from file: {prompts}")
    else:
        prompts = args.prompts

    i = 1
    while (Path(args.output_root) / f"predict_{i}").exists():i += 1
    predict_output_folder = Path(args.output_root) / f"predict_{i}"

    metadata = run_privacy_pipeline(
        source_path=args.source,
        model_path=args.model,
        output_root=predict_output_folder,
        SAM_type = args.SAM_type,
        prompts=prompts,
        crop_mode=args.crop_mode,
        public_key_path=args.public_key,
        video_stride=int(args.vid_stride),
        embed_payloads=args.embed_payloads,
    )
    end_time = time.time()
    total_sec = end_time - start_time

    print(f"Input type: {metadata['input_type']}")
    print(f"Saved blurred output to: {metadata['blurred_output_path']}")
    print(f"Total Time (seconds): {total_sec:.2f}")


if __name__ == "__main__":
    main()

    # python3 main.py --source ../data/input/sample_img.jpeg --crop-mode mask --embed-payloads

    # python3 main.py --source ../data/input/sample1.mp4 --crop-mode bbox --embed-payloads
    # python3 main.py --source ../data/input/sample1_0.5fps.mp4 --crop-mode mask --embed-payloads


 #CUDA_VISIBLE_DEVICES=1
 # python3 main.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.30_2.40.mp4  --prompts "smartphone screen" "music equipment" "underwear" "books" --crop-mode mask --vid-stride 5

 #python3 main.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.30_2.40.mp4 --prompts "smartphone screen" "TV Screen" "clothes" --crop-mode mask --vid-stride 5

 #python3 main.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.30_2.40.mp4 --prompts "smartphone screen" "TV Screen" "clothes" --crop-mode mask --model ../models/FastSAM-x.pt --SAM-type FastSAM


 # python3 main.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.35_2.40.mp4  --prompts "smartphone screen" "underwear" "books" --crop-mode mask --vid-stride 5




# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc1_script3_seq2_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc1_script5_seq5_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc2_script5_seq2_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10
# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc2_script5_seq7_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script2_seq3_rec2_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script4_seq2_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script5_seq3_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10
# python3 main.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script5_seq5_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10



# python3 main.py --source ../data/input/RealWorld/quest_1.mp4
# python3 main.py --source ../data/input/RealWorld/quest_2.mp4



