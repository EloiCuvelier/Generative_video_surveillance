"""
This script evaluates the model's ability to classify frames
without having been specifically trained on UCF-Crime (Zero-shot).

The zero-shot principle is as follows: we encode the 13 text labels of the dataset
(fighting, shooting, etc.) with the same model used to encode the images. 
Then, for each frame, we check which text label is the most similar. 
The class with the highest similarity score becomes the prediction.

We then compare the predictions with the ground truth classes to compute the accuracy,
which is the percentage of correctly classified frames.

Results are saved in results/metaclip/zeroshot/:
    predictions.csv    the prediction and true label for each frame
    accuracy.json      the final score and details per class
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
import open_clip

# Configure logging to track script progress and detect potential issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


RESULTS_DIR = Path("results")

# 13 classes formulated as sentences because models were trained on text descriptions
LABELS = [
    "a person fighting",
    "a robbery",
    "a car accident",
    "vandalism",
    "a person being arrested",
    "a burglary",
    "a person shooting",
    "an abuse",
    "an arson",
    "an assault",
    "a road accident",
    "shoplifting",
    "a person stealing"
]

# Mapping between dataset folder names and the label indices above
DATASET_CLASSES = [
    "Fighting",
    "Robbery",
    "Accident",
    "Vandalism",
    "Arrest",
    "Burglary",
    "Shooting",
    "Abuse",
    "Arson",
    "Assault",
    "RoadAccidents",
    "Shoplifting",
    "Stealing"
]

# Configuration restricted to MetaCLIP only
MODELS = [
    {
        "name"            : "MetaCLIP",
        "folder"          : RESULTS_DIR / "metaclip",
        "prefix"          : "metaclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32-quickgelu",
        "model_pretrained": "metaclip_400m"
    }
]

def encode_labels_metaclip(labels, model_name, model_pretrained, device):
    """
    Encodes text labels using MetaCLIP (via the open_clip library).
    """
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=model_pretrained
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)
    tokens    = tokenizer(labels).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(tokens)
        # Normalize the embeddings for cosine similarity
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return text_embeddings.cpu().float().numpy()


def evaluate_zeroshot(config):
    """
    Performs zero-shot classification for the given model configuration.
    """
    name   = config["name"]
    folder = config["folder"]
    prefix = config["prefix"]

    logger.info(f"Starting zero-shot classification for {name}...")

    embeddings_path = folder / f"{prefix}_embeddings.npy"
    metadata_path   = folder / f"{prefix}_metadata.csv"

    if not embeddings_path.exists() or not metadata_path.exists():
        logger.warning(f"Missing files for {name}, skipping model.")
        return

    # Load previously extracted video frame embeddings and their metadata
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata   = pd.read_csv(metadata_path, index_col=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Encoding text labels...")
    text_embeddings = encode_labels_metaclip(
        LABELS, config["model_name"], config["model_pretrained"], device
    )
    logger.info(f"Labels encoded, shape: {text_embeddings.shape}")

    # Calculate similarity between each frame and each label via matrix multiplication
    logger.info("Calculating similarities...")
    scores = embeddings @ text_embeddings.T

    # For each frame, pick the index of the label with the highest score
    predictions_idx = np.argmax(scores, axis=1)
    predictions     = [DATASET_CLASSES[i] for i in predictions_idx]

    # Calculate global accuracy by comparing predictions with true classes
    true_classes = metadata["class"].tolist()
    correct      = sum(p == t for p, t in zip(predictions, true_classes))
    accuracy     = correct / len(true_classes)

    logger.info(f"Global accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Calculate accuracy per class to see where the model struggles the most
    accuracy_per_class = {}
    for cls in DATASET_CLASSES:
        mask       = [t == cls for t in true_classes]
        num_frames = sum(mask)
        if num_frames == 0:
            continue
        num_correct = sum(
            p == t for p, t, m in zip(predictions, true_classes, mask) if m
        )
        accuracy_per_class[cls] = round(num_correct / num_frames, 4)

    # Save frame-by-frame predictions for error analysis
    output_dir = folder / "zeroshot"
    output_dir.mkdir(parents=True, exist_ok=True)

    results               = metadata.copy()
    results["prediction"] = predictions
    results["correct"]    = [p == t for p, t in zip(predictions, true_classes)]
    results["score_max"]  = scores.max(axis=1)

    predictions_path = output_dir / "predictions.csv"
    results.to_csv(predictions_path, index=True)
    logger.info(f"Predictions saved to {predictions_path}")

    # Save performance summary for comparison charts
    accuracy_data = {
        "model"              : name,
        "global_accuracy"    : round(accuracy, 4),
        "accuracy_percent"   : round(accuracy * 100, 2),
        "total_frames"       : len(true_classes),
        "total_correct"      : correct,
        "accuracy_per_class" : accuracy_per_class
    }

    accuracy_path = output_dir / "accuracy.json"
    with open(accuracy_path, "w") as f:
        json.dump(accuracy_data, f, indent=2)
    logger.info(f"Accuracy summary saved to {accuracy_path}\n")


if __name__ == "__main__":
    logger.info("Starting zero-shot classification evaluation...\n")

    for config in MODELS:
        evaluate_zeroshot(config)

    logger.info("Classification evaluation completed.")