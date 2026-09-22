"""
faiss_indexing.py

This script builds the FAISS index for the MetaCLIP model.

It loads the embeddings produced by the encoding script,
builds a FAISS index, and saves it to disk along with its metadata.

Files produced in results/metaclip/index/:
    index.faiss      the FAISS index containing all vectors
    metadata.csv     copy of the metadata aligned with the index (same order)
    index_info.json  information about the index (dimension, vector count, type...)
"""

import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
import logging


# Display informational messages to track script progress and detect issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Root folder containing the encoding results
RESULTS_DIR = Path("results")

# Configuration restricted to MetaCLIP only
MODELS = [
    {
        "name"   : "MetaCLIP",
        "folder" : RESULTS_DIR / "metaclip",
        "prefix" : "metaclip"
    }
]


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Builds a FAISS index from an embeddings matrix."""
    dimension = embeddings.shape[1]

    # Inner Product (cosine similarity since vectors are already normalized)
    index = faiss.IndexFlatIP(dimension)

    # Wrap the index in an IndexIDMap to associate each vector with a numeric ID 
    # that matches its row in the metadata CSV.
    index_with_ids = faiss.IndexIDMap(index)

    # IDs go from 0 to N-1, one per frame, in the exact same order as the CSV
    ids = np.arange(len(embeddings), dtype=np.int64)

    # Ensure embeddings are in float32 format as required by FAISS
    embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

    index_with_ids.add_with_ids(embeddings_f32, ids)

    return index_with_ids


def index_model(config: dict):
    """
    Main function that processes the model: loads embeddings and metadata, 
    builds the index, and saves it to disk.
    """
    name   = config["name"]
    folder = config["folder"]
    prefix = config["prefix"]

    logger.info(f"Processing {name}...")

    # Verify that the necessary files exist
    embeddings_path = folder / f"{prefix}_embeddings.npy"
    metadata_path   = folder / f"{prefix}_metadata.csv"

    if not embeddings_path.exists():
        logger.warning(f"Embeddings file not found for {name}: {embeddings_path}, model skipped")
        return

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found for {name}: {metadata_path}, model skipped")
        return

    # Load the embeddings produced by the encoding script
    logger.info(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    logger.info(f"{len(embeddings):,} vectors loaded, dimension {embeddings.shape[1]}")

    # Load the metadata containing class, video, and timestamp per frame
    metadata = pd.read_csv(metadata_path, index_col=0)

    # Verify that the number of vectors matches the number of rows in the CSV
    # If not, something went wrong during encoding
    if len(embeddings) != len(metadata):
        logger.error(
            f"Mismatch for {name}: {len(embeddings)} embeddings "
            f"but {len(metadata)} rows in metadata, model skipped"
        )
        return

    # Build the FAISS index
    logger.info("Building FAISS index...")
    index = build_index(embeddings)
    logger.info(f"Index built with {index.ntotal} vectors")

    # Save in an index/ subfolder to keep things organized
    output_dir = folder / "index"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index to disk
    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"Index saved to {index_path}")

    # Save a copy of the metadata next to the index so they stay aligned
    metadata_output_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_output_path, index=True)
    logger.info(f"Metadata saved to {metadata_output_path}")

    # Save an information file about the index
    index_info = {
        "model"          : name,
        "vector_count"   : int(index.ntotal),
        "dimension"      : int(embeddings.shape[1]),
        "index_type"     : "IndexIDMap(IndexFlatIP)",
        "normalization"  : "L2",
        "embeddings_src" : str(embeddings_path),
        "metadata_src"   : str(metadata_path)
    }

    info_path = output_dir / "index_info.json"
    with open(info_path, "w") as f:
        json.dump(index_info, f, indent=2)
    logger.info(f"Index information saved to {info_path}")

    logger.info(f"Indexing for {name} completed\n")


def test_index(config: dict):
    """
    Test function to verify that the index works correctly by performing a similarity search.
    """
    name   = config["name"]
    folder = config["folder"]
    prefix = config["prefix"]

    index_path    = folder / "index" / "index.faiss"
    metadata_path = folder / "index" / "metadata.csv"

    if not index_path.exists():
        logger.warning(f"Index not found for {name}, test skipped")
        return

    logger.info(f"Testing index for {name}...")

    index    = faiss.read_index(str(index_path))
    metadata = pd.read_csv(metadata_path, index_col=0)

    # Load the first embedding to use as a query vector
    embeddings = np.load(folder / f"{prefix}_embeddings.npy")
    query      = np.ascontiguousarray(embeddings[0:1], dtype=np.float32)

    # Search for the 5 most similar frames
    scores, ids = index.search(query, k=5)

    logger.info(f"Test results for {name}:")
    for rank, (score, frame_id) in enumerate(zip(scores[0], ids[0])):
        frame_info = metadata.iloc[frame_id]
        logger.info(
            f"  rank {rank + 1} : score {score:.4f} "
            f"{frame_info['class']} "
            f"{frame_info['video']} "
            f"t={frame_info['timestamp']}s"
        )

    logger.info("")


if __name__ == "__main__":

    logger.info("Starting FAISS indexing...\n")

    # Index the model
    for config in MODELS:
        index_model(config)

    # Quick test of the index to verify everything works
    logger.info("Running index verification tests...")
    for config in MODELS:
        test_index(config)

    logger.info("Indexing completed.")