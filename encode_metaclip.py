"""
This script encodes all frames extracted from the UCF-Crime dataset using the MetaCLIP model.

It saves three files at the end in results/<output_name>/:
    <output_name>_embeddings.npy   the vectors of all frames
    <output_name>_metadata.csv     the information associated with each frame (class, video, timestamp)
    <output_name>_metrics.json     the model's performance (time, speed...)
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import open_clip
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

# Display logs with timestamps to track progress and performance
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


MODEL_NAME       = "ViT-B-32-quickgelu"
MODEL_PRETRAINED = "metaclip_400m"
OUTPUT_NAME      = "metaclip"

FRAMES_DIR = Path("frames")
OUTPUT_DIR = Path(f"results/{OUTPUT_NAME}")

BATCH_SIZE = 32

# Checkpoint save interval in seconds (10 minutes)
CHECKPOINT_INTERVAL = 10 * 60


# Class to load frames from disk in an organized way for PyTorch
class FrameDataset(Dataset):

    def __init__(self, frame_paths, preprocess):
        self.frame_paths = frame_paths
        self.preprocess  = preprocess

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        path = self.frame_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            # Apply the model's preprocessor which resizes and normalizes
            # the image according to what the open_clip model expects
            return self.preprocess(image), str(path), True
        except Exception:
            # If an image is corrupted or unreadable, return a dummy tensor
            # rather than crashing the entire script
            dummy = torch.zeros(3, 224, 224)
            return dummy, str(path), False


def save_checkpoint(output_dir, output_name, all_embeddings, all_metadata, errors, nb_frames_processed):
    """Saves a checkpoint to resume later."""
    checkpoint_path = output_dir / f'{output_name}_checkpoint.npz'
    tmp_path        = output_dir / f'{output_name}_checkpoint_tmp.npz'

    if all_embeddings:
        emb_matrix = np.concatenate(all_embeddings, axis=0)
    else:
        emb_matrix = np.empty((0, 512), dtype=np.float32)

    # Save to a temporary file first, then rename to prevent corruption
    # if the script is interrupted during writing
    np.savez(
        tmp_path,
        embeddings=emb_matrix,
        nb_frames_processed=np.array(nb_frames_processed),
        errors=np.array(errors)
    )
    meta_tmp = output_dir / f'{output_name}_checkpoint_meta.csv'
    pd.DataFrame(all_metadata).to_csv(meta_tmp, index=True)

    tmp_path.replace(checkpoint_path)
    logger.info(f"Checkpoint saved: {len(all_metadata)} frames processed")


def load_checkpoint(output_dir, output_name):
    """Loads an existing checkpoint. Returns None if none exists."""
    checkpoint_path = output_dir / f'{output_name}_checkpoint.npz'
    meta_path       = output_dir / f'{output_name}_checkpoint_meta.csv'

    if not checkpoint_path.exists() or not meta_path.exists():
        return None

    data                = np.load(checkpoint_path)
    embeddings          = data['embeddings']
    nb_frames_processed = int(data['nb_frames_processed'])
    errors              = int(data['errors'])

    meta_df  = pd.read_csv(meta_path, index_col=0)
    metadata = meta_df.to_dict('records')

    return {
        'embeddings'          : embeddings,
        'metadata'            : metadata,
        'nb_frames_processed' : nb_frames_processed,
        'errors'              : errors
    }


def cleanup_checkpoint(output_dir, output_name):
    """Deletes checkpoint files once encoding is finished."""
    for suffix in ['_checkpoint.npz', '_checkpoint_meta.csv', '_checkpoint_tmp.npz']:
        p = output_dir / f'{output_name}{suffix}'
        if p.exists():
            p.unlink()
    logger.info("Checkpoint files cleaned up")


# Main function that encodes frames and saves the results
def encode_frames(frames_dir, output_dir, model_name, model_pretrained, output_name, batch_size):

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if a GPU is available, otherwise run on CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device used: {device}")
    if device == 'cuda':
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # open_clip automatically downloads weights on the first run
    # and caches them in ~/.cache/huggingface/ or ~/.cache/open_clip/
    logger.info(f"Loading model {model_name} ({model_pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=model_pretrained
    )
    model = model.to(device)
    model.eval()  # Switch to evaluation mode to disable dropout and other training mechanisms
    logger.info(f"Model loaded on {device}")

    # Get the list of all dataset frames
    logger.info(f"Searching for frames in {frames_dir}...")
    frame_paths = sorted(frames_dir.rglob('*_t*.jpg'))

    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    logger.info(f"{len(frame_paths):,} frames found")

    # Check for an existing checkpoint to resume encoding
    resume_checkpoint = load_checkpoint(output_dir, output_name)
    skip_frames       = 0

    all_embeddings = []
    all_metadata   = []
    errors         = 0

    if resume_checkpoint is not None:
        skip_frames = resume_checkpoint['nb_frames_processed']
        all_embeddings.append(resume_checkpoint['embeddings'])
        all_metadata = resume_checkpoint['metadata']
        errors       = resume_checkpoint['errors']
        logger.info(f"Checkpoint found: {skip_frames:,} frames already processed, resuming...")

        frame_paths = frame_paths[skip_frames:]
        logger.info(f"{len(frame_paths):,} frames remaining to encode")

        if len(frame_paths) == 0:
            logger.info("All frames have already been encoded!")
            embeddings_matrix = resume_checkpoint['embeddings']
            encoding_time     = 0
            nb_encoded        = len(embeddings_matrix)
            _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                                encoding_time, nb_encoded, model_name, model_pretrained,
                                output_name, batch_size, device)
            cleanup_checkpoint(output_dir, output_name)
            return
    else:
        logger.info("No checkpoint found, starting from scratch")

    # The DataLoader will load images in batches and in parallel
    # which is much faster than loading them one by one
    dataset = FrameDataset(frame_paths, preprocess)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )

    # Counter for frames processed in this session (for the checkpoint we add skip_frames)
    frames_this_session = 0

    logger.info("Starting encoding...")
    # Start the timer here to measure only pure encoding time
    # without counting model loading
    start_time           = time.time()
    last_checkpoint_time = start_time

    with torch.no_grad():  # Disable gradient calculation since we are not training the model
        for images, paths, valid_flags in tqdm(loader, desc=f"{model_name}"):

            valid_mask   = valid_flags.bool()
            valid_images = images[valid_mask].to(device)
            valid_paths  = [p for p, v in zip(paths, valid_flags) if v]

            if len(valid_images) == 0:
                errors              += len(paths)
                frames_this_session += len(paths)
                continue

            # Pass images through the model to get embeddings
            # open_clip uses encode_image like CLIP and MobileCLIP
            embeddings = model.encode_image(valid_images)

            # Normalize vectors so their norm equals 1
            # this is necessary for cosine similarity to work correctly with FAISS
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().float().numpy())

            # Extract metadata from the filename to track
            # which video and what time each frame belongs to
            for path_str in valid_paths:
                p          = Path(path_str)
                class_name = p.parts[-3]
                video_name = p.parts[-2]
                filename   = p.name
                timestamp  = int(filename.rsplit('_t', 1)[1].replace('s.jpg', ''))

                all_metadata.append({
                    'filepath'  : path_str,
                    'class'     : class_name,
                    'video'     : video_name,
                    'timestamp' : timestamp,
                    'filename'  : filename
                })

            errors              += (~valid_mask).sum().item()
            frames_this_session += len(paths)

            # Periodic checkpoint saving every CHECKPOINT_INTERVAL seconds
            now = time.time()
            if now - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                total_processed = skip_frames + frames_this_session
                save_checkpoint(output_dir, output_name, all_embeddings, all_metadata,
                                errors, total_processed)
                last_checkpoint_time = now

    encoding_time = time.time() - start_time
    nb_encoded    = sum(len(b) for b in all_embeddings)

    logger.info(f"Encoding finished in {encoding_time:.1f}s ({encoding_time/60:.1f} min)")
    logger.info(f"{nb_encoded:,} frames encoded, {errors} errors")
    if encoding_time > 0:
        logger.info(f"Average speed: {nb_encoded/encoding_time:.1f} frames/sec")

    # Concatenate all batches into a single numpy matrix
    embeddings_matrix = np.concatenate(all_embeddings, axis=0)

    _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                        encoding_time, nb_encoded, model_name, model_pretrained,
                        output_name, batch_size, device)

    # Delete the checkpoint since encoding finished successfully
    cleanup_checkpoint(output_dir, output_name)


def _save_final_results(output_dir, embeddings_matrix, all_metadata, errors,
                        encoding_time, nb_encoded, model_name, model_pretrained,
                        output_name, batch_size, device):
    """Saves the final files (embeddings, metadata, metrics)."""

    # Save embeddings
    embeddings_path = output_dir / f'{output_name}_embeddings.npy'
    np.save(embeddings_path, embeddings_matrix)
    logger.info(f"Embeddings saved to {embeddings_path}, shape: {embeddings_matrix.shape}")

    # Save metadata, the index of each row matches
    # the vector's position in the embeddings matrix
    metadata_path = output_dir / f'{output_name}_metadata.csv'
    pd.DataFrame(all_metadata).to_csv(metadata_path, index=True)
    logger.info(f"Metadata saved to {metadata_path}")

    # Save performance metrics, this file will be compared
    # with those of other models for the charts
    metrics = {
        'model'             : f'{model_name} ({model_pretrained})',
        'nb_frames_encoded' : int(nb_encoded),
        'nb_errors'         : int(errors),
        'encoding_time_sec' : round(encoding_time, 2),
        'encoding_time_min' : round(encoding_time / 60, 2),
        'frames_per_second' : round(nb_encoded / max(encoding_time, 0.01), 2),
        'embedding_dim'     : int(embeddings_matrix.shape[1]),
        'batch_size'        : batch_size,
        'device'            : device
    }

    metrics_path = output_dir / f'{output_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    encode_frames(
        frames_dir       = FRAMES_DIR,
        output_dir       = OUTPUT_DIR,
        model_name       = MODEL_NAME,
        model_pretrained = MODEL_PRETRAINED,
        output_name      = OUTPUT_NAME,
        batch_size       = BATCH_SIZE
    )