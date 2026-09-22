"""
This script extracts frames from videos in the dataset using FFmpeg.
It creates a structured output folder with one JPEG frame per second.
"""

import os
import subprocess
from pathlib import Path
import logging

# Path to the Dataset folder (containing the 13 class subfolders)
DATASET_DIR = Path("Dataset")

# Output folder where all extracted frames will be stored
OUTPUT_DIR = Path("frames")

# Number of frames extracted per second of video
FPS = 1

# Video extensions recognized by the script
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


# Configure logging to display messages in the terminal with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path: Path, output_folder: Path, fps: int = 1) -> int:
    """
    Extracts frames from a video using FFmpeg.
    """
    # Create the output folder if it doesn't exist yet
    output_folder.mkdir(parents=True, exist_ok=True)

    # Name pattern for the frames, example: Shooting004_x264_t0001s.jpg
    video_name = video_path.stem
    output_pattern = output_folder / f"{video_name}_t%04ds.jpg"

    # Build the FFmpeg command
    command = [
        "ffmpeg",
        "-i", str(video_path),          # source video
        "-vf", f"fps={fps}",            # filter: N frames per second
        "-frame_pts", "1",              # use real timestamp for numbering
        "-q:v", "2",                    # quality of the generated JPEGs
        "-loglevel", "error",           # hide verbose FFmpeg logs
        str(output_pattern)             # output path with timestamp pattern
    ]

    try:
        # Run the FFmpeg command in a subprocess
        subprocess.run(command, check=True)

        # Count the number of frames actually created in the folder
        num_frames = len(list(output_folder.glob("*_t*.jpg")))
        return num_frames

    except subprocess.CalledProcessError as e:
        # FFmpeg encountered an error on this video
        logger.error(f"FFmpeg error on {video_path.name}: {e}")
        return 0

    except FileNotFoundError:
        # FFmpeg is not installed or not in the PATH
        logger.critical("FFmpeg not found. Install it with: conda install -c conda-forge ffmpeg")
        raise


def process_dataset(dataset_dir: Path, output_dir: Path, fps: int = 1):
    """
    Iterates through the entire dataset and extracts frames from each video.
    """
    # Check if the dataset folder exists
    if not dataset_dir.exists():
        logger.error(f"The dataset folder '{dataset_dir}' cannot be found.")
        return

    # Get the list of the 13 classes (subfolders of the dataset)
    class_folders = sorted([f for f in dataset_dir.iterdir() if f.is_dir()])

    if not class_folders:
        logger.warning("No class subfolder found in the dataset.")
        return

    logger.info(f"Dataset found: {len(class_folders)} classes detected")
    logger.info(f"Output frames → {output_dir.resolve()}")
    logger.info(f"Extraction FPS: {fps} frame(s)/second\n")

    # Global counters for the final summary
    total_videos = 0
    total_frames = 0
    total_errors = 0

    # Loop through each class
    for class_folder in class_folders:
        class_name = class_folder.name 

        # List all videos in this class folder
        video_files = sorted([
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ])

        if not video_files:
            logger.warning(f"[{class_name}] No videos found, class skipped.")
            continue

        logger.info(f"[{class_name}] {len(video_files)} video(s) found")

        # Loop through each video in the class
        for video_path in video_files:

            # Name the output folder for the frames of this video, e.g.: frames/Shooting/Shooting004_x264/
            video_stem = video_path.stem
            output_folder = output_dir / class_name / video_stem

            # If the folder already exists and contains frames, skip it
            if output_folder.exists() and any(output_folder.glob("*_t*.jpg")):
                existing = len(list(output_folder.glob("*_t*.jpg")))
                logger.info(f"[{video_stem}] already extracted ({existing} frames), skipped.")
                total_videos += 1
                total_frames += existing
                continue

            logger.info(f"[{video_stem}] extraction in progress...")

            # Call the extraction function
            num_frames = extract_frames_from_video(video_path, output_folder, fps)

            if num_frames > 0:
                logger.info(f"[{video_stem}] {num_frames} frames extracted")
                total_frames += num_frames
            else:
                logger.warning(f"[{video_stem}] failed or empty video")
                total_errors += 1

            total_videos += 1

    # Final summary
    logger.info("EXTRACTION COMPLETED")
    logger.info(f"  Videos processed : {total_videos}")
    logger.info(f"  Frames extracted : {total_frames:,}")
    logger.info(f"  Errors           : {total_errors}")
    logger.info(f"  Output directory : {output_dir.resolve()}")


# ENTRY POINT
if __name__ == "__main__":
    process_dataset(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        fps=FPS
    )