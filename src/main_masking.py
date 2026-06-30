from __future__ import annotations

import argparse
from pathlib import Path
import time
import json
from pathlib import Path

from privacy_video.pipeline import run_privacy_pipeline


def load_prompts_from_preference_file(preference_file_path):
    """
    Read privacy preference JSON file and extract all labels as prompts.
    """

    with open(preference_file_path, "r") as f:
        data = json.load(f)

    prompts = []

    privacy_categories = data.get("privacy_categories", {})

    category_wise_prompts = {}

    for category_name, category_data in privacy_categories.items():
        labels = category_data.get("Labels", [])

        category_wise_prompts[category_name] = []

        for label in labels:
            if label and isinstance(label, str):
                prompts.append(label.strip())
                category_wise_prompts[category_name].append(label.strip())

    # add all prompts to one array, remove duplicates while preserving order
    prompts = list(dict.fromkeys(prompts))

    # A reverse lookup dictionary to map each prompt to its category
    promptLabel_to_privacyCategory = {}
    for category, labels in category_wise_prompts.items():
        for label in labels:
            promptLabel_to_privacyCategory[label] = category

    return prompts, promptLabel_to_privacyCategory

###
# Directly invoke this function to run the script for batch of data efficiently
###
def process_one_video(
    source_path,
    common_preference_file,
    output_root,
    SAM_type="SAM3",
    model_path="../models/sam3.1_multiplex.pt",
    crop_mode="mask",
    blur_type="Pb",
    public_key_path="../keys/device_public.pem",
    vid_stride=1,
    save_payloads=True,
    sam_processor=None,
    sam_confidence = 0.95,
    prompts = None,
):
    source_path = Path(source_path)
    file_name = source_path.stem

    if (prompts == None and common_preference_file != None):
        # read any privacy preference file name placed same level as the dataset folder
        prompts, prompt_label_to_privacy_category = load_prompts_from_preference_file(
            Path(source_path.parent.parent, common_preference_file)
        )
    elif(prompts == None):
        # read any specific privacy preference file saved with the data file (old way for single files)
            preference_file = (f"{file_name}_privacy_annotation.json") # need to make sure, the privacy preference file always ends like this
            prompts, prompt_label_to_privacy_category = load_prompts_from_preference_file( Path(Path(source_path).parent, preference_file))
    else:
        # just assign 'other' as privacy category for all prompts
        prompt_label_to_privacy_category = {prompt: "Other" for prompt in prompts}
    
    print(f"Loaded privacy prompts from the file: {prompts}")   
    predict_output_folder = Path(output_root) / file_name

    start_time = time.time()

    official_end_time, total_frames, fps = run_privacy_pipeline(
        source_path=str(source_path),
        model_path=model_path,
        output_root=predict_output_folder,
        SAM_type=SAM_type,
        prompt_label_to_privacy_category=prompt_label_to_privacy_category,
        crop_mode=crop_mode,
        blur_type=blur_type,
        public_key_path=public_key_path,
        video_stride=int(vid_stride),
        save_payloads=save_payloads,
        sam_processor=sam_processor,
        sam_confidence = float(sam_confidence),
        prompts=prompts,
    )

    total_sec = official_end_time - start_time

    print(f"Total Time (seconds): {total_sec:.2f}")

    return {
        "video_uid": file_name,
        "output_dir": str(predict_output_folder),
        "total_time_s": total_sec,
        "total_frames" : total_frames,
        "fps": fps
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-preserving SAM pipeline")
    parser.add_argument("--source", required=True, help="Path to input image or video")
    parser.add_argument("--common-preference-file", default=None, help="pass the name of the common privacy preference file placed in the source folder")
    parser.add_argument("--prompts", nargs="*", default=None, help="Optional prompt list")

    parser.add_argument("--SAM-type", choices=["SAM3", "FastSAM"], default="SAM3", help="the SAM model processor that should be used for execution")
    parser.add_argument("--SAM-confidence", default=0.95, help="the confidence used by SAM model processor to decide a prediction")
    parser.add_argument("--model", default="../models/sam3.1_multiplex.pt", help="Path to SAM3 model")
    parser.add_argument("--output-root", default="../data/output", help="Directory to save outputs")
    parser.add_argument("--crop-mode", choices=["mask", "bbox"], default="mask", help="Use mask or bbox for cropping/blur")
    parser.add_argument("--blur-type", choices=["Gb", "Pb"], default="Pb", help="Type of the bluring technique (Gussian Blur or Pixellation)")
    parser.add_argument("--public-key", default="../keys/device_public.pem", help="Path to device RSA public key PEM file")
    parser.add_argument("--vid-stride", default=1, help="Video stride rate for frame skipping for efficiency")
    parser.add_argument("--no-save-payloads", action="store_false",dest="save_payloads", help="Disable saving intermediate payload files") # this arg param should be just set for computation time measurements
    parser.set_defaults(save_payloads=True)
    return parser

###
#This function can be used to run the scrip eith CLI command line arguments
###
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    process_one_video(
        source_path=args.source,
        common_preference_file=args.common_preference_file,
        prompts = args.prompts,
        output_root=args.output_root,
        SAM_type=args.SAM_type,
        model_path=args.model,
        crop_mode=args.crop_mode,
        blur_type=args.blur_type,
        public_key_path=args.public_key,
        vid_stride=args.vid_stride,
        save_payloads=args.save_payloads,
        sam_processor=None,
        sam_confidence=args.SAM_confidence,
    )


if __name__ == "__main__":
    main()

# def main() -> None:

#     parser = build_parser()
#     args = parser.parse_args()

#     file_name = Path(args.source).stem
#     if (args.prompts == None):
#         if (args.common_preference_file != None):
#             prompts, prompt_label_to_privacy_category = load_prompts_from_preference_file( Path(Path(args.source).parent, args.common_preference_file))
#         else: 
#             # read any file specific privacy preference file
#             preference_file = (f"{file_name}_privacy_annotation.json") # need to make sure, the privacy preference file always ends like this
#             print(preference_file)
#             prompts, prompt_label_to_privacy_category = load_prompts_from_preference_file( Path(Path(args.source).parent, preference_file))
#     else:
#         prompts = args.prompts
    
#     print(f"Loaded privacy prompts from file: {prompts}")  

#     # i = 1
#     # while (Path(args.output_root) / f"{file_name}_{i}").exists():i += 1
#     # predict_output_folder = Path(args.output_root) / f"{file_name}_{i}"

#     predict_output_folder = Path(args.output_root) / f"{file_name}"

#     start_time = time.time()
#     offical_process_pipeline_ending_time = run_privacy_pipeline(
#         source_path=args.source,
#         model_path=args.model,
#         output_root=predict_output_folder,
#         SAM_type = args.SAM_type,
#         prompts=prompts,
#         prompt_label_to_privacy_category = prompt_label_to_privacy_category,
#         crop_mode=args.crop_mode,
#         blur_type=args.blur_type,
#         public_key_path=args.public_key,
#         video_stride=int(args.vid_stride),
#         save_payloads=args.save_payloads,
#     )
#     total_sec = offical_process_pipeline_ending_time - start_time

#     # print(f"Input type: {metadata['input_type']}")
#     # print(f"Saved blurred output to: {metadata['blurred_output_path']}")
#     print(f"Total Time (seconds): {total_sec:.2f}")









# python3 main_masking.py --source ../data/input/sample_img.jpeg --crop-mode mask --no-save-payloads
# python3 main_masking.py --source /home/tran5174/CogAgentVLM/table.jpg --crop-mode mask  --prompts " white container with red labeling, possibly containing skincare or personal care product."

# python3 main_masking.py --source ../data/input/sample1.mp4 --crop-mode bbox --no-save-payloads
# python3 main_masking.py --source ../data/input/sample1_0.5fps.mp4 --crop-mode mask --no-save-payloads


 #CUDA_VISIBLE_DEVICES=1
 # python3 main_masking.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.30_2.40.mp4  --prompts "smartphone screen" "music equipment" "underwear" "books" --crop-mode mask --vid-stride 5

 #python3 main_masking.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.30_2.40.mp4 --prompts "smartphone screen" "TV Screen" "clothes" --crop-mode mask --vid-stride 5

 #python3 main_masking.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.30_2.40.mp4 --prompts "smartphone screen" "TV Screen" "clothes" --crop-mode mask --model ../models/FastSAM-x.pt --SAM-type FastSAM


 # python3 main_masking.py --source ../data/input/AEA/AriaEverydayActivities_1.0.0_loc3_script5_seq6_rec1_preview_rgb_cropped_2.35_2.40.mp4  --prompts "smartphone screen" "underwear" "books" --crop-mode mask --vid-stride 5




# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc1_script3_seq2_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc1_script5_seq5_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc2_script5_seq2_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10
# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc2_script5_seq7_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script2_seq3_rec2_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script4_seq2_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10

# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script5_seq3_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10
# python3 main_masking.py --source ../data/input/AriaAEA_selected/AriaEverydayActivities_1.0.0_loc3_script5_seq5_rec1_preview_rgb_middle10s_10fps.mp4 --vid-stride 10



# CUDA_VISIBLE_DEVICES=1 python3 main.py --source ../data/input/RealWorld/quest_1.mp4
# python3 main.py --source ../data/input/RealWorld/quest_2.mp4



#CUDA_VISIBLE_DEVICES=1

#python3 main_masking.py --source ../data/input/EgoObjects_Videos/Spatial-video2.MOV --crop-mode mask

# EGO Objects new dataset, individual runs

# python3 main_masking.py --source ../data/Ego_Eval/RESTRUCT_EGO_OBJECTS_1/0A0DD98EB432BFD4563DAB1750D552FF_01.mp4 --common-preference-file common_privacy_preference.json --output-root ../data/Ego_Eval/output2
# python3 main_masking.py --source ../data/Ego_Eval/RESTRUCT_EGO_OBJECTS_1/0A9CDF7364EBBFDBD4B762BA2C50EAC0_01.mp4 --common-preference-file common_privacy_preference.json --output-root ../data/Ego_Eval/output2
# python3 main_masking.py --source ../data/Ego_Eval/RESTRUCT_EGO_OBJECTS_1/0B2BD8C2BD046B78D989C864D7ECCB81_01.mp4 --common-preference-file common_privacy_preference.json --output-root ../data/Ego_Eval/output2

#python3 main_masking.py --source ../data/Ego_Eval/RESTRUCT_EGO_OBJECTS_1/8C27026BB289A25638F04C98113D574B_10.mp4 --common-preference-file common_privacy_preference.json --output-root ../data/Ego_Eval/output2



# python3 main_masking.py --source ../data/Ego_Eval/RESTRUCT_EGO_OBJECTS_1/8C27026BB289A25638F04C98113D574B_10.mp4 --prompts KidToys --output-root ../data/Ego_Eval/output2 --SAM-confidence 0.5