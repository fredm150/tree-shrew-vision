# sync_and_extract.py

# Align two videos using manually found clap times,
# auto-match to shorter duration,
# optionally trim extra time from the end of BOTH videos,
# then extract synchronized frame pairs.

import cv2
import os
import argparse
import subprocess
import sys



# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync two videos using manual clap times and extract frames."
    )

    parser.add_argument(
        "--left-video",
        default="test_left_outdoor.mov",
        help="Path to left video"
    )
    parser.add_argument(
        "--right-video",
        default="test_right_outdoor.mp4",
        help="Path to right video"
    )
    parser.add_argument(
        "--output-dir",
        default="./input_data",
        help="Directory for extracted frames"
    )
    parser.add_argument(
        "--left-start",
        required=True,
        type=float,
        help="Sync/clap time in left video (seconds)"
    )
    parser.add_argument(
        "--right-start",
        required=True,
        type=float,
        help="Sync/clap time in right video (seconds)"
    )
    parser.add_argument(
        "--joint-end-trim",
        type=float,
        default=0.0,
        help="Extra seconds to trim from end of both synced videos"
    )

    return parser.parse_args()


# Helpers

def get_duration(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for {path}")

    return frames / fps


def trim_video(input_path, start_time, duration, output_path):
    """
    Use ffmpeg to cut synced clip.
    Re-encode for frame-accurate cuts.
    """
    subprocess.run([
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-an",
        output_path
    ], check=True)


def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out_path = os.path.join(
            output_folder,
            f"frame_{count:05d}.jpg"
        )

        cv2.imwrite(
            out_path,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        count += 1

        if count % 200 == 0:
            print(f"  {count}/{total} frames...")

    cap.release()

    print(f"Extracted {count} frames -> {output_folder}")
    return count


# Main

def main():
    args = parse_args()

    print(f"Left sync point:  {args.left_start:.4f}s")
    print(f"Right sync point: {args.right_start:.4f}s")

    # Remaining duration after clap point
    left_remaining = (
        get_duration(args.left_video)
        - args.left_start
    )

    right_remaining = (
        get_duration(args.right_video)
        - args.right_start
    )

    # Auto match to shorter video
    shared_duration = min(
        left_remaining,
        right_remaining
    ) - args.joint_end_trim

    if shared_duration <= 0:
        print("ERROR: joint-end-trim removes entire clip.")
        sys.exit(1)

    print(
        f"Shared synced duration: {shared_duration:.2f}s"
    )

    print("\nTrimming synced videos...")

    trim_video(
        args.left_video,
        args.left_start,
        shared_duration,
        "left_synced.mov"
    )

    trim_video(
        args.right_video,
        args.right_start,
        shared_duration,
        "right_synced.mov"
    )

    print("Synced clips written:")
    print("  left_synced.mov")
    print("  right_synced.mov")

    # --------------------------------------------------
    # Extract frames
    # --------------------------------------------------

    print("\nExtracting frames...")

    left_frames = extract_frames(
        "left_synced.mov",
        os.path.join(
            args.output_dir,
            "left_frames"
        )
    )

    right_frames = extract_frames(
        "right_synced.mov",
        os.path.join(
            args.output_dir,
            "right_frames"
        )
    )

    if left_frames != right_frames:
        print(
            f"\nWARNING frame mismatch:"
            f" left={left_frames}, right={right_frames}"
        )
        print(
            f"Use first {min(left_frames,right_frames)} pairs."
        )
    else:
        print(
            f"\nDone — {left_frames} synced pairs ready in:"
            f" {args.output_dir}/"
        )


if __name__ == "__main__":
    main()