import cv2
import numpy as np


def count_frames(path):
    cap = cv2.VideoCapture(path)
    count = 0

    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1

    cap.release()
    return count


def compare_videos(path1, path2, diff_threshold=5.0):
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    frame_idx = 0
    diffs = []

    while True:
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()

        if not ok1 or not ok2:
            break

        if f1.shape != f2.shape:
            raise ValueError(f"Shape mismatch at frame {frame_idx}: {f1.shape} vs {f2.shape}")

        diff = np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32)))
        diffs.append(diff)

        frame_idx += 1

    cap1.release()
    cap2.release()

    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    max_diff = float(np.max(diffs)) if diffs else 0.0

    print("\n--- Comparison Results ---")
    print(f"Total frames compared: {frame_idx}")
    print(f"Average diff: {avg_diff:.4f}")
    print(f"Max diff: {max_diff:.4f}")

    if avg_diff < diff_threshold:
        print("✅ PASS: Videos are visually similar")
        return True
    else:
        print("❌ FAIL: Videos differ significantly")
        return False


def main():
    input_path = "../data/input/sample.mp4"
    output_path = "../data/output/copied.mp4"

    print("Checking frame counts...")

    in_count = count_frames(input_path)
    out_count = count_frames(output_path)

    print(f"Input frames:  {in_count}")
    print(f"Output frames: {out_count}")

    if in_count != out_count:
        print("❌ FAIL: Frame counts do not match")
        return

    print("Frame counts match ✅")

    print("\nComparing frame content...")
    compare_videos(input_path, output_path)


if __name__ == "__main__":
    main()