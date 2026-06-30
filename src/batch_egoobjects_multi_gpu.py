from pathlib import Path
import os
import csv
import time
import multiprocessing as mp

from main_masking import process_one_video
from privacy_video.pipeline import create_SAM_processor_object


PROJECT_ROOT = Path(__file__).resolve().parent

# INPUT_DIR = PROJECT_ROOT / "../data/Ego_Eval/RESTRUCT_EGO_OBJECTS"
INPUT_DIR = PROJECT_ROOT / "../data/Ego_Eval/EGO_SELECT"

DATA_CAPTURE_TYPE = "01" # None
# OUTPUT_ROOT = PROJECT_ROOT / "../data/Ego_Eval/output01_1"
OUTPUT_ROOT = PROJECT_ROOT / "../data/Ego_Eval/output01_s0.80"

# COMMON_PRIVACY_PREFERENCE_FILE = "common_privacy_preference_0.json"
COMMON_PRIVACY_PREFERENCE_FILE = "common_privacy_preference_s.json"
SAM_CONFIDENCE = 0.80

MODEL_PATH = "../models/sam3.1_multiplex.pt"
SAM_TYPE = "SAM3"
VID_STRIDE = 1
CROP_MODE = "mask"
BLUR_TYPE = "Pb"
PUBLIC_KEY = "../keys/device_public.pem"

GPU_IDS = [0, 1]   # change to [0,1,2,3] if available

SUMMARY_CSV = OUTPUT_ROOT / "batch_summary.csv"


def split_list(items, n):
    return [items[i::n] for i in range(n)]


def gpu_worker(gpu_id, video_paths):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Loading SAM once...")

    sam_processor = create_SAM_processor_object(
        SAM_type=SAM_TYPE,
        video_stride=VID_STRIDE,
        model_path=MODEL_PATH,
        SAM_conf = SAM_CONFIDENCE
    )

    rows = []

    for video_path in video_paths:
        video_uid = video_path.stem
        output_dir = OUTPUT_ROOT / video_uid

        if (output_dir / "encrypted_private_regions.json").exists():
            print(f"[GPU {gpu_id}] Skip existing: {video_uid}")
            rows.append({
                "video_uid": video_uid,
                "gpu_id": gpu_id,
                "status": "skipped_existing",
                "total_time_s": 0,
                "output_dir": str(output_dir),
            })
            continue

        print(f"[GPU {gpu_id}] Processing: {video_uid}")

        start = time.time()

        try:
            result = process_one_video(
                source_path=video_path,
                common_preference_file=COMMON_PRIVACY_PREFERENCE_FILE,# should be placed same level as the dataset directory
                output_root=OUTPUT_ROOT,
                SAM_type=SAM_TYPE,
                model_path=MODEL_PATH,
                crop_mode=CROP_MODE,
                blur_type=BLUR_TYPE,
                public_key_path=PUBLIC_KEY,
                vid_stride=VID_STRIDE,
                save_payloads=True,
                sam_processor=sam_processor,
                sam_confidence= 0.7
            )

            status = "success"

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR on {video_uid}: {e}")
            result = {
                "video_uid": video_uid,
                "output_dir": str(output_dir),
            }
            status = f"error: {e}"

        elapsed = time.time() - start

        rows.append({
            "video_uid": video_uid,
            "gpu_id": gpu_id,
            "status": status,
            "total_time_s": elapsed,
            "total_frames": result.get("total_frames"),
            "fps": result.get("fps"),
            "output_dir": result.get("output_dir", str(output_dir)),
        })

    return rows


def save_summary(all_rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for worker_rows in all_rows:
        flat_rows.extend(worker_rows)

    if not flat_rows:
        return

    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(flat_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Saved summary: {SUMMARY_CSV}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if (DATA_CAPTURE_TYPE == None):
        videos = sorted(INPUT_DIR.glob("*.mp4"))
    else: 
        videos = sorted(INPUT_DIR.glob(f"*_{DATA_CAPTURE_TYPE}.mp4"))

    print(f"Total Videos in the dataset - under selection criteriea {DATA_CAPTURE_TYPE}: {len(videos)}")

    if not videos:
        print(f"No videos found in {INPUT_DIR}")
        return

    video_splits = split_list(videos, len(GPU_IDS))

    jobs = [
        (gpu_id, video_splits[i])
        for i, gpu_id in enumerate(GPU_IDS)
    ]
    print(f"GPUs: {GPU_IDS}")

    with mp.Pool(processes=len(GPU_IDS)) as pool:
        all_rows = pool.starmap(gpu_worker, jobs)

    save_summary(all_rows)


if __name__ == "__main__":
    main()

# python3 batch_egoobjects_multi_gpu.py