"""
gradio_interface.py (MetaCLIP Exclusive Version)

Gradio interface with 5 tabs to explore the UCF-Crime dataset using the MetaCLIP model.
"""

import numpy as np
import pandas as pd
import faiss
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import gradio as gr
import logging
import open_clip
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")

# Keeping only MetaCLIP
AVAILABLE_MODELS = {
    "MetaCLIP": {
        "folder"          : RESULTS_DIR / "metaclip",
        "prefix"          : "metaclip",
        "type"            : "openclip",
        "model_name"      : "ViT-B-32-quickgelu",
        "model_pretrained": "metaclip_400m"
    }
}

# 13 classes with 5 descriptions each (prompt ensembling, desc_4_mean_embed method)
DATASET_CLASSES = [
    "Fighting", "Robbery", "Explosion", "Vandalism", "Arrest",
    "Burglary", "Shooting", "Abuse", "Arson", "Assault",
    "RoadAccidents", "Shoplifting", "Stealing"
]

MULTI_DESCRIPTIONS = {
    "Fighting": [
        "a person fighting",
        "two or more people engaging in a mutual physical fight",
        "opponents punching and kicking each other simultaneously",
        "a brawl where multiple parties are actively hitting one another",
        "a reciprocal physical altercation between individuals"
    ],
    "Robbery": [
        "a robbery",
        "someone robbing a person by using immediate force or a weapon",
        "a victim being threatened with a weapon to hand over valuables",
        "an aggressive mugging where a person is held up",
        "stealing from a person through direct confrontation and intimidation"
    ],
    "Explosion": [
        "an explosion",
        "a sudden violent burst of fire and smoke from a device or building",
        "a large blast with flames and debris flying outward",
        "an explosive detonation causing destruction and shockwave",
        "a bomb or gas explosion caught on surveillance camera"
    ],
    "Vandalism": [
        "vandalism",
        "someone intentionally destroying or defacing public property",
        "a person spray painting graffiti on a wall or building",
        "breaking windows, mirrors or damaging structures",
        "deliberate destruction of property caught on camera"
    ],
    "Arrest": [
        "a person being arrested",
        "police officers in uniform apprehending a suspect",
        "law enforcement officers placing handcuffs on an individual",
        "a suspect being detained and led away by police",
        "official police intervention to take a person into custody"
    ],
    "Burglary": [
        "a burglary",
        "someone breaking into a closed building or house to steal",
        "a person forcing entry through a door or window of a property",
        "illegal entry into a home or business, usually while unoccupied",
        "a burglar sneaking into a private premises to commit theft"
    ],
    "Shooting": [
        "a person shooting",
        "someone discharging a firearm or handgun",
        "a person aiming and firing a gun at a target",
        "active gunfire and muzzle flashes from a weapon",
        "an individual using a firearm in a violent incident"
    ],
    "Abuse": [
        "an abuse",
        "a person physically mistreating a child, elderly, or defenseless person",
        "someone inflicting repetitive harm on a vulnerable individual",
        "physical violence against a victim who is not fighting back",
        "domestic or interpersonal violence caught on camera"
    ],
    "Arson": [
        "an arson",
        "a person intentionally starting a fire with fuel or a lighter",
        "someone deliberately setting fire to a building or vehicle",
        "the act of lighting a structure on fire to cause destruction",
        "intentional ignition of a fire caught on surveillance"
    ],
    "Assault": [
        "an assault",
        "one person suddenly attacking or beating a victim",
        "a violent physical attack where the victim is being overpowered",
        "an aggressor punching or hitting a person who is retreating",
        "a person being physically assaulted by an attacker"
    ],
    "RoadAccidents": [
        "a road accident",
        "a car crash occurring in the middle of a road or intersection",
        "vehicles colliding during traffic flow",
        "a traffic mishap involving moving cars or pedestrians on a street",
        "motor vehicle accident caught on a roadway camera"
    ],
    "Shoplifting": [
        "shoplifting",
        "a customer stealing merchandise inside a retail store",
        "someone concealing store items in their clothes or bag",
        "taking products from shop shelves without paying at the counter",
        "theft committed during business hours inside a commercial shop"
    ],
    "Stealing": [
        "a person stealing",
        "someone taking an unattended object that does not belong to them",
        "theft of a bag, bicycle, or item from a public space",
        "a person picking up and walking away with someone else's property",
        "non-violent theft of belongings in an open environment"
    ]
}

# Colors for the 13 classes used in the similarity plot (tab 4)
CLASS_COLORS = [
    "#E05C5C", "#4A6FD4", "#3DAA6E", "#E07B39", "#C45BAA",
    "#2ABBE8", "#D4A017", "#7B68EE", "#5FAD8E", "#D45B8A",
    "#8BC34A", "#FF7043", "#607D8B"
]


# ─────────────────────────────────────────────────────────────────────────────
# CACHE (models, FAISS index, raw embeddings for tabs 3 and 4)
# ─────────────────────────────────────────────────────────────────────────────

_encoder_cache      = {}   # model_name -> tuple (type, model, extra, device)
_index_cache        = {}   # model_name -> (faiss_index, metadata_df)
_embeddings_cache   = {}   # model_name -> (embeddings np.array, metadata_df)
_class_embed_cache  = {}   # model_name -> class embeddings matrix (13 x dim)
_tsne_cache         = {}   # model_name -> DataFrame (x, y, class, video) after t-SNE


# ─────────────────────────────────────────────────────────────────────────────
# MODEL AND INDEX LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(model_name: str):
    """Loads and caches the requested text encoder (MetaCLIP)."""
    if model_name in _encoder_cache:
        return _encoder_cache[model_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg    = AVAILABLE_MODELS[model_name]
    logger.info(f"Loading text encoder for {model_name}...")

    model, _, _ = open_clip.create_model_and_transforms(
        cfg["model_name"], pretrained=cfg["model_pretrained"]
    )
    model     = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["model_name"])
    
    _encoder_cache[model_name] = ("openclip", model, tokenizer, device)

    logger.info(f"Encoder loaded for {model_name}")
    return _encoder_cache[model_name]


def load_index(model_name: str):
    """Loads and caches the FAISS index + metadata for the model."""
    if model_name in _index_cache:
        return _index_cache[model_name]

    cfg     = AVAILABLE_MODELS[model_name]
    folder  = cfg["folder"]
    idx_p   = folder / "index" / "index.faiss"
    meta_p  = folder / "index" / "metadata.csv"

    if not idx_p.exists():
        raise FileNotFoundError(
            f"FAISS index not found for {model_name} ({idx_p}).\n"
            "Run the faiss_indexing.py script first."
        )

    logger.info(f"Loading FAISS index for {model_name}...")
    index    = faiss.read_index(str(idx_p))
    metadata = pd.read_csv(meta_p, index_col=0)
    _index_cache[model_name] = (index, metadata)
    logger.info(f"Index loaded: {index.ntotal:,} vectors")
    return _index_cache[model_name]


def load_raw_embeddings(model_name: str):
    """
    Loads and caches the raw embeddings matrix and metadata.
    Used by tabs 3 and 4 which require video-level embedding access.
    """
    if model_name in _embeddings_cache:
        return _embeddings_cache[model_name]

    cfg     = AVAILABLE_MODELS[model_name]
    emb_p   = cfg["folder"] / f"{cfg['prefix']}_embeddings.npy"
    meta_p  = cfg["folder"] / f"{cfg['prefix']}_metadata.csv"

    if not emb_p.exists():
        raise FileNotFoundError(
            f"Embeddings not found for {model_name} ({emb_p}).\n"
            "Run the corresponding encoding script first."
        )

    logger.info(f"Loading raw embeddings for {model_name}...")
    embeddings = np.load(emb_p).astype(np.float32)
    metadata   = pd.read_csv(meta_p, index_col=0)
    _embeddings_cache[model_name] = (embeddings, metadata)
    logger.info(f"Embeddings loaded: {embeddings.shape}")
    return _embeddings_cache[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# TEXT ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_text(text_or_list, encoder):
    """
    Encodes a single text or a list of texts using the provided encoder (MetaCLIP).
    Returns a normalized numpy array (N x dim).
    """
    type_m, model, extra, device = encoder
    texts = [text_or_list] if isinstance(text_or_list, str) else text_or_list

    with torch.no_grad():
        tokens = extra(texts).to(device)
        emb    = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().float().numpy()


def encode_single_query(text: str, model_name: str) -> np.ndarray:
    """Encodes a user query into a vector (1 x dim)."""
    encoder = load_encoder(model_name)
    return encode_text(text, encoder)


def get_class_embeddings(model_name: str) -> np.ndarray:
    """
    Builds and caches the class matrix (13 x dim) via prompt ensembling:
    for each class, we encode its 5 descriptions, average their embeddings,
    and normalize — desc_4_mean_embed method recommended by the CLIP paper.
    """
    if model_name in _class_embed_cache:
        return _class_embed_cache[model_name]

    logger.info(f"Building class embeddings (mean_embed) for {model_name}...")
    encoder = load_encoder(model_name)
    vectors = []

    for cls_name in DATASET_CLASSES:
        emb   = encode_text(MULTI_DESCRIPTIONS[cls_name], encoder)  # (5 x dim)
        avg   = emb.mean(axis=0)
        avg   = avg / np.linalg.norm(avg)
        vectors.append(avg)

    matrix = np.stack(vectors).astype(np.float32)
    _class_embed_cache[model_name] = matrix
    logger.info(f"Class matrix built: {matrix.shape}")
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FRAME SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_frames(query: str, model_name: str, top_n: int):
    """
    Direct FAISS search: returns the N closest frames to the query vector.
    Identical to the original interface behavior.
    """
    if not query.strip():
        return [], "Please enter a text query."

    try:
        vector           = encode_single_query(query, model_name)
        index, metadata  = load_index(model_name)
        scores, ids      = index.search(vector, k=int(top_n))

        images, lines = [], []
        for rank, (score, frame_id) in enumerate(zip(scores[0], ids[0])):
            if frame_id < 0 or frame_id >= len(metadata):
                continue
            info = metadata.iloc[frame_id]
            path = Path(info["filepath"])
            images.append(
                Image.open(path).convert("RGB") if path.exists()
                else Image.new("RGB", (224, 224), (80, 80, 80))
            )
            lines.append(
                f"#{rank+1}  sim={score:.3f}  "
                f"class={info['class']}  video={info['video']}  t={info['timestamp']}s"
            )

        summary = (
            f"Found {len(images)} frames with {model_name} "
            f"for « {query} »"
        )
        return images, summary + "\n\n" + "\n".join(lines)

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        logger.error(f"Tab 1 error: {e}")
        return [], f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VIDEO SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_videos(query: str, model_name: str, top_n: int, top_k_frames: int):
    """
    Calculates an aggregated score for each video in the dataset:
        video_score = average of its top-K FAISS frame scores.
    Returns the M most relevant videos with their most representative frame.

    Strategy:
      1. Encode query and perform a broad FAISS search (10,000 results)
         to cover all potentially relevant videos.
      2. Group scores by video and compute the mean of top-K scores.
      3. Sort and return the top M videos.
    """
    if not query.strip():
        return [], "Please enter a text query."

    try:
        vector          = encode_single_query(query, model_name)
        index, metadata = load_index(model_name)

        # Request enough results to cover all videos
        k_large = min(index.ntotal, 50_000)
        scores_flat, ids_flat = index.search(vector, k=k_large)

        # Group by video: list of (score, frame_id) for each video
        scores_by_video = defaultdict(list)
        best_frame      = {}   # video_key -> (score, frame_id) of the highest scoring frame

        for score, frame_id in zip(scores_flat[0], ids_flat[0]):
            if frame_id < 0 or frame_id >= len(metadata):
                continue
            info      = metadata.iloc[frame_id]
            video_key = (info["class"], info["video"])
            scores_by_video[video_key].append((float(score), int(frame_id)))

            # Keep the highest scoring frame as the thumbnail
            if video_key not in best_frame or score > best_frame[video_key][0]:
                best_frame[video_key] = (float(score), int(frame_id))

        # Calculate video score: average of the top-K frames
        video_scores = {}
        for video_key, score_list in scores_by_video.items():
            score_list.sort(key=lambda x: x[0], reverse=True)
            top_k = score_list[:top_k_frames]
            video_scores[video_key] = sum(s for s, _ in top_k) / len(top_k)

        # Sort videos by descending score
        sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_videos = sorted_videos[:int(top_n)]

        images, lines = [], []
        for rank, ((cls_name, video), v_score) in enumerate(sorted_videos):
            # Representative frame = frame with the highest individual score
            _, best_id = best_frame[(cls_name, video)]
            best_info  = metadata.iloc[best_id]
            path       = Path(best_info["filepath"])

            images.append(
                Image.open(path).convert("RGB") if path.exists()
                else Image.new("RGB", (224, 224), (80, 80, 80))
            )
            nb_frames = len(scores_by_video[(cls_name, video)])
            lines.append(
                f"#{rank+1}  avg_score={v_score:.3f}  "
                f"class={cls_name}  video={video}  "
                f"({nb_frames} frames in dataset)"
            )

        summary = (
            f"Found {len(images)} videos with {model_name} "
            f"for « {query} » (top-{top_k_frames} frames aggregated)"
        )
        return images, summary + "\n\n" + "\n".join(lines)

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        logger.error(f"Tab 2 error: {e}")
        return [], f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SEQUENCE SEARCH (sliding window)
# ─────────────────────────────────────────────────────────────────────────────

def search_sequences(
    query       : str,
    model_name  : str,
    top_n       : int,
    window_size : int
):
    """
    Applies a sliding window over the embeddings of each video.
    Window score = mean cosine similarity between the query vector
    and the frame embeddings inside the window.

    Returns the top N segments with their central frame as a thumbnail,
    and the timestamp interval [t_start, t_end].
    """
    if not query.strip():
        return [], "Please enter a text query."

    try:
        vector               = encode_single_query(query, model_name)   # (1 x dim)
        vector_1d            = vector[0]  # (dim,)
        embeddings, metadata = load_raw_embeddings(model_name)

        # Calculate cosine similarity of every frame with the query in one operation
        # Shape: (total_frames,) — we have direct access to all embeddings
        all_sims = embeddings @ vector_1d  # dot product, equivalent to cosine since normalized

        # Group frames by video, in chronological order
        videos = defaultdict(list)  # video_key -> list of (timestamp, global_idx)
        for idx, row in metadata.iterrows():
            video_key = (row["class"], row["video"])
            # idx is the pandas index corresponding to the position in the embeddings matrix
            videos[video_key].append((row["timestamp"], int(idx)))

        # Sort by timestamp to ensure chronological order within each video
        for key in videos:
            videos[key].sort(key=lambda x: x[0])

        # Sliding window over each video
        best_segments = []  # list of (score, class, video, start_ts, end_ts, center_frame_idx)

        for (cls_name, video), sorted_frames in videos.items():
            nb = len(sorted_frames)
            if nb < window_size:
                # Video too short: take all frames as a single window
                global_indices = [idx for _, idx in sorted_frames]
                score    = float(all_sims[global_indices].mean())
                start_ts = sorted_frames[0][0]
                end_ts   = sorted_frames[-1][0]
                mid_idx  = global_indices[len(global_indices) // 2]
                best_segments.append((score, cls_name, video, start_ts, end_ts, mid_idx))
                continue

            for i in range(nb - window_size + 1):
                window         = sorted_frames[i : i + window_size]
                global_indices = [idx for _, idx in window]
                score          = float(all_sims[global_indices].mean())
                start_ts       = window[0][0]
                end_ts         = window[-1][0]
                # Central frame for thumbnail
                mid_idx        = global_indices[window_size // 2]
                best_segments.append((score, cls_name, video, start_ts, end_ts, mid_idx))

        # Sort by descending score, then deduplicate by video + position
        # (keep the best segment per video to avoid overlapping results)
        best_segments.sort(key=lambda x: x[0], reverse=True)

        # Deduplication: keep only one segment per video in final results
        seen     = set()
        filtered = []
        for seg in best_segments:
            _, cls_name, video, start_ts, end_ts, _ = seg
            key = (cls_name, video)
            if key not in seen:
                seen.add(key)
                filtered.append(seg)
            if len(filtered) >= int(top_n):
                break

        images, lines = [], []
        for rank, (score, cls_name, video, start_ts, end_ts, mid_idx) in enumerate(filtered):
            info = metadata.iloc[mid_idx]
            path = Path(info["filepath"])
            images.append(
                Image.open(path).convert("RGB") if path.exists()
                else Image.new("RGB", (224, 224), (80, 80, 80))
            )
            lines.append(
                f"#{rank+1}  score={score:.3f}  "
                f"class={cls_name}  video={video}  "
                f"t={start_ts}s → t={end_ts}s  ({window_size}-frame window)"
            )

        summary = (
            f"Found {len(images)} sequences with {model_name} "
            f"for « {query} » (sliding window of {window_size} frames)"
        )
        return images, summary + "\n\n" + "\n".join(lines)

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        logger.error(f"Tab 3 error: {e}")
        return [], f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — VIDEO SIMILARITY PLOT
# ─────────────────────────────────────────────────────────────────────────────

def list_videos(model_name: str):
    """
    Returns a list of all available videos for the chosen model,
    formatted as "Class / VideoName" sorted alphabetically.
    Used to populate the dropdown menu in tab 4.
    """
    try:
        _, metadata = load_raw_embeddings(model_name)
        pairs       = metadata[["class", "video"]].drop_duplicates()
        return sorted(f"{r['class']} / {r['video']}" for _, r in pairs.iterrows())
    except Exception:
        return []


def plot_video_similarity(
    video_selection: str,
    model_name     : str,
    visible_classes: list
):
    """
    For the selected video, plots the cosine similarity of each frame
    against the 13 classes (via averaged 5-description embeddings).

    Returns a matplotlib figure containing:
      - One curve per selected class (temporal frame-by-frame plot)
      - A colored background indicating the video's ground truth class
    """
    if not video_selection:
        return None, "Please select a video from the list."

    try:
        embeddings, metadata = load_raw_embeddings(model_name)
        class_matrix         = get_class_embeddings(model_name)  # (13 x dim)

        # Extract class and video name from the "Class / VideoName" selection
        parts = video_selection.split(" / ", maxsplit=1)
        if len(parts) != 2:
            return None, "Invalid selection format."
        target_class, target_video = parts[0].strip(), parts[1].strip()

        # Filter frames for this video
        mask       = (metadata["class"] == target_class) & (metadata["video"] == target_video)
        frame_idxs = metadata.index[mask].tolist()

        if not frame_idxs:
            return None, f"No frames found for {video_selection}."

        # Extract and sort by timestamp
        frames_meta = metadata.loc[frame_idxs].copy()
        frames_meta = frames_meta.sort_values("timestamp")
        sorted_idxs = frames_meta.index.tolist()

        video_emb  = embeddings[sorted_idxs]        # (nb_frames x dim)
        timestamps = frames_meta["timestamp"].tolist()

        # Compute similarities: (nb_frames x 13)
        sims = video_emb @ class_matrix.T

        # Build Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        true_class_idx = DATASET_CLASSES.index(target_class) if target_class in DATASET_CLASSES else -1

        # Colored background to indicate ground truth class
        ax.axhspan(
            sims.min() - 0.02, sims.max() + 0.02,
            alpha=0.04,
            color=CLASS_COLORS[true_class_idx] if true_class_idx >= 0 else "#CCCCCC",
            label=f"_background"
        )

        # Plot each selected class
        classes_to_plot = visible_classes if visible_classes else DATASET_CLASSES
        for i, cls_name in enumerate(DATASET_CLASSES):
            if cls_name not in classes_to_plot:
                continue
            lw        = 2.5 if cls_name == target_class else 1.0
            ls        = "-"  if cls_name == target_class else "--"
            alpha     = 0.95 if cls_name == target_class else 0.55
            label_txt = f"★ {cls_name} (ground truth)" if cls_name == target_class else cls_name
            ax.plot(
                timestamps, sims[:, i],
                color=CLASS_COLORS[i],
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                label=label_txt,
                zorder=3 if cls_name == target_class else 2
            )

        # Horizontal line at 0 for reference
        ax.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle=":")

        ax.set_xlabel("Timestamp (seconds)", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title(
            f"Class Similarity — {target_video}  [{target_class}]  |  {model_name}\n"
            f"({len(timestamps)} frames, class embeddings: average of 5 descriptions)",
            fontsize=11, pad=12
        )
        ax.set_xlim(timestamps[0], timestamps[-1])
        ax.grid(True, linestyle="--", alpha=0.35, color="gray")
        ax.set_facecolor("#F8F9FA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Two-column legend to prevent clutter
        ax.legend(
            loc="upper right",
            fontsize=8,
            ncol=2,
            framealpha=0.92,
            edgecolor="#DDDDDD",
            fancybox=True
        )

        plt.tight_layout()

        # Text Summary
        true_class_sim = sims[:, true_class_idx] if true_class_idx >= 0 else np.zeros(len(timestamps))
        summary = (
            f"Video: {target_video}  |  Ground truth: {target_class}  |  "
            f"{len(timestamps)} frames  |  Model: {model_name}\n"
            f"Avg sim. true class: {true_class_sim.mean():.3f}  |  "
            f"Max sim. true class: {true_class_sim.max():.3f}"
        )
        return fig, summary

    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        logger.error(f"Tab 4 error: {e}")
        return None, f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — VIDEO t-SNE MAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_video_embeddings(model_name: str) -> pd.DataFrame:
    """
    For each video in the dataset, computes the average embedding across all its frames
    (1 normalized vector per video). Caches the result.
    Returns a DataFrame with columns: video, class, embedding (array).
    """
    if model_name in _tsne_cache:
        return _tsne_cache[model_name]

    logger.info(f"Computing average video embeddings for {model_name}...")
    embeddings, metadata = load_raw_embeddings(model_name)

    rows = []
    groups = metadata.groupby(["class", "video"])
    for (cls_name, video), group in groups:
        idx     = group.index.tolist()
        avg_emb = embeddings[idx].mean(axis=0)
        norm    = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm
        rows.append({"class": cls_name, "video": video, "embedding": avg_emb})

    df = pd.DataFrame(rows)
    _tsne_cache[model_name] = df
    logger.info(f"Average embeddings computed: {len(df)} videos")
    return df


def tsne_map(
    model_name     : str,
    visible_classes: list,
    perplexity     : int,
    n_iter         : int
):
    """
    Applies t-SNE on average video embeddings and plots the interactive
    Plotly map (1 point = 1 video, color = class, hover = video name).

    t-SNE parameters exposed to user:
      - perplexity : controls local/global balance (recommended: 5–50)
      - n_iter     : number of optimization iterations (recommended: 500–2000)
    """
    if not visible_classes:
        return None, "Please select at least one class to display."

    try:
        df_videos = compute_video_embeddings(model_name)

        # Filter on selected classes
        df_filtered = df_videos[df_videos["class"].isin(visible_classes)].copy()
        if df_filtered.empty:
            return None, "No video found for the selected classes."

        # Embedding matrix (N x dim)
        X = np.stack(df_filtered["embedding"].values).astype(np.float32)

        # t-SNE — perplexity must be < number of points
        eff_perp = min(int(perplexity), len(df_filtered) - 1)
        logger.info(
            f"t-SNE on {len(df_filtered)} videos "
            f"(perplexity={eff_perp}, n_iter={n_iter})..."
        )
        tsne = TSNE(
            n_components=2,
            perplexity=eff_perp,
            max_iter=int(n_iter),
            random_state=42,
            init="pca",
            learning_rate="auto"
        )
        coords = tsne.fit_transform(X)   # (N x 2)

        df_filtered = df_filtered.copy()
        df_filtered["x"] = coords[:, 0]
        df_filtered["y"] = coords[:, 1]

        # ── Build Plotly Figure ───────────────────────────────────────────
        fig = go.Figure()

        for i, cls_name in enumerate(DATASET_CLASSES):
            if cls_name not in visible_classes:
                continue
            subset = df_filtered[df_filtered["class"] == cls_name]
            if subset.empty:
                continue

            color = CLASS_COLORS[DATASET_CLASSES.index(cls_name)]

            fig.add_trace(go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                name=cls_name,
                marker=dict(
                    color=color,
                    size=7,
                    opacity=0.80,
                    line=dict(width=0.5, color="white")
                ),
                # Text displayed on hover
                text=subset["video"],
                customdata=subset["class"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Class: %{customdata}<br>"
                    "x: %{x:.2f} | y: %{y:.2f}"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=dict(
                text=(
                    f"UCF-Crime Videos t-SNE Map — {model_name}<br>"
                    f"<sup>{len(df_filtered)} videos · "
                    f"perplexity={eff_perp} · {n_iter} iterations</sup>"
                ),
                font=dict(size=14)
            ),
            xaxis=dict(title="t-SNE dim 1", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
            yaxis=dict(title="t-SNE dim 2", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
            legend=dict(
                title="Class",
                itemsizing="constant",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#DDDDDD",
                borderwidth=1
            ),
            plot_bgcolor="#F8F9FA",
            paper_bgcolor="white",
            height=650,
            margin=dict(l=40, r=40, t=80, b=40),
            hoverlabel=dict(bgcolor="white", font_size=12)
        )

        summary = (
            f"{len(df_filtered)} projected videos | "
            f"{len(visible_classes)} classes displayed | "
            f"Model: {model_name}"
        )
        return fig, summary

    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        logger.error(f"Tab 5 error: {e}")
        return None, f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# DROPDOWN UPDATE CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def update_video_list(model_name: str):
    """
    Called when the user changes the model in tab 4.
    Reloads the available videos list for this model.
    """
    videos = list_videos(model_name)
    if videos:
        return gr.update(choices=videos, value=videos[0])
    return gr.update(choices=[], value=None)


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_interface():
    model_names = list(AVAILABLE_MODELS.keys())

    with gr.Blocks(title="UCF-Crime — CLIP Search", theme=gr.themes.Soft()) as interface:

        gr.Markdown(
            "#UCF-Crime Video Search — MetaCLIP Model\n"
            "Explore the UCF-Crime dataset using text queries."
        )

        # ── TAB 1 : FRAMES ─────────────────────────────────────────────────
        with gr.Tab("Frames"):
            gr.Markdown(
                "**Search by individual frames** — "
                "Returns the top N frames from the dataset most similar to your query."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    t1_query  = gr.Textbox(
                        label="Text Query",
                        placeholder="Ex : person fighting, car accident, robbery...",
                        lines=1
                    )
                with gr.Column(scale=1):
                    t1_model  = gr.Dropdown(
                        choices=model_names, value="MetaCLIP", label="Model"
                    )
            with gr.Row():
                t1_nb       = gr.Slider(1, 24, value=6, step=1, label="Number of frames")
                t1_btn      = gr.Button("Search", variant="primary")

            t1_summary  = gr.Textbox(label="Summary", lines=4, interactive=False)
            t1_gallery  = gr.Gallery(label="Frames", columns=4, height=520)

            t1_btn.click(
                fn=search_frames,
                inputs=[t1_query, t1_model, t1_nb],
                outputs=[t1_gallery, t1_summary]
            )
            t1_query.submit(
                fn=search_frames,
                inputs=[t1_query, t1_model, t1_nb],
                outputs=[t1_gallery, t1_summary]
            )

            gr.Markdown(
                "**Examples:** `person fighting` | `car crash on road` | "
                "`robbery in store` | `person with gun` | `fire and smoke`"
            )

        # ── TAB 2 : VIDEOS ─────────────────────────────────────────────────
        with gr.Tab("Videos"):
            gr.Markdown(
                "**Search by videos** — "
                "Aggregates the scores of all frames in each video "
                "(average of the top-K frames). "
                "The thumbnail shown is the most similar frame of the video."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    t2_query  = gr.Textbox(
                        label="Text Query",
                        placeholder="Ex : person fighting, car accident...",
                        lines=1
                    )
                with gr.Column(scale=1):
                    t2_model  = gr.Dropdown(
                        choices=model_names, value="MetaCLIP", label="Model"
                    )
            with gr.Row():
                t2_nb       = gr.Slider(1, 20, value=6, step=1, label="Number of videos")
                t2_topk     = gr.Slider(1, 30, value=5, step=1,
                                        label="Top-K frames per video for aggregation")
                t2_btn      = gr.Button("Search", variant="primary")

            t2_summary  = gr.Textbox(label="Summary", lines=5, interactive=False)
            t2_gallery  = gr.Gallery(
                label="Videos (most representative frame)", columns=3, height=520
            )

            t2_btn.click(
                fn=search_videos,
                inputs=[t2_query, t2_model, t2_nb, t2_topk],
                outputs=[t2_gallery, t2_summary]
            )
            t2_query.submit(
                fn=search_videos,
                inputs=[t2_query, t2_model, t2_nb, t2_topk],
                outputs=[t2_gallery, t2_summary]
            )

        # ── TAB 3 : SEQUENCES ──────────────────────────────────────────────
        with gr.Tab("Sequences"):
            gr.Markdown(
                "**Search by sequences** — "
                "Sliding window over the frames of each video. "
                "Returns the continuous segments most similar to the query "
                "with their timestamp interval `[t_start → t_end]`."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    t3_query  = gr.Textbox(
                        label="Text Query",
                        placeholder="Ex : person fighting, fire and smoke...",
                        lines=1
                    )
                with gr.Column(scale=1):
                    t3_model  = gr.Dropdown(
                        choices=model_names, value="MetaCLIP", label="Model"
                    )
            with gr.Row():
                t3_nb       = gr.Slider(1, 20, value=6, step=1, label="Number of sequences")
                t3_window   = gr.Slider(2, 30, value=5, step=1,
                                        label="Window size (frames)")
                t3_btn      = gr.Button("Search", variant="primary")

            t3_summary  = gr.Textbox(label="Summary", lines=6, interactive=False)
            t3_gallery  = gr.Gallery(
                label="Sequences (central frame of the window)", columns=3, height=520
            )

            t3_btn.click(
                fn=search_sequences,
                inputs=[t3_query, t3_model, t3_nb, t3_window],
                outputs=[t3_gallery, t3_summary]
            )
            t3_query.submit(
                fn=search_sequences,
                inputs=[t3_query, t3_model, t3_nb, t3_window],
                outputs=[t3_gallery, t3_summary]
            )

        # ── TAB 4 : SIMILARITY PLOT ────────────────────────────────────────
        with gr.Tab("Class Similarity"):
            gr.Markdown(
                "**Temporal Similarity Plot** — "
                "Select a video from the dataset. The graph plots, frame by frame, "
                "the cosine similarity with each of the 13 classes. "
                "Class embeddings are computed via **prompt ensembling**: "
                "averaging 5 description vectors per class (the `mean_embed` method "
                "recommended by the original CLIP paper). "
                "The video's ground truth class is highlighted (solid line, ★)."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    t4_model  = gr.Dropdown(
                        choices=model_names, value="MetaCLIP", label="Model"
                    )
                with gr.Column(scale=3):
                    # Initializing video list on first load
                    init_videos = list_videos("MetaCLIP")
                    t4_video    = gr.Dropdown(
                        choices=init_videos,
                        value=init_videos[0] if init_videos else None,
                        label="Video (Class / VideoName)"
                    )

            # Filter for classes to display (all by default)
            t4_classes  = gr.CheckboxGroup(
                choices=DATASET_CLASSES,
                value=DATASET_CLASSES,
                label="Classes to display on the plot"
            )
            with gr.Row():
                t4_btn_all  = gr.Button("Select All",   size="sm")
                t4_btn_none = gr.Button("Deselect All", size="sm")

            t4_btn_all.click(
                fn=lambda: DATASET_CLASSES,
                inputs=[],
                outputs=[t4_classes]
            )
            t4_btn_none.click(
                fn=lambda: [],
                inputs=[],
                outputs=[t4_classes]
            )

            t4_btn      = gr.Button("Plot Graph", variant="primary")
            t4_summary  = gr.Textbox(label="Summary", lines=2, interactive=False)
            t4_graph    = gr.Plot(label="Temporal Similarity by Class")

            # Update video list when model changes
            t4_model.change(
                fn=update_video_list,
                inputs=[t4_model],
                outputs=[t4_video]
            )

            t4_btn.click(
                fn=plot_video_similarity,
                inputs=[t4_video, t4_model, t4_classes],
                outputs=[t4_graph, t4_summary]
            )

        # ── TAB 5 : t-SNE MAP ──────────────────────────────────────────────
        with gr.Tab("t-SNE Map"):
            gr.Markdown(
                "**t-SNE Map of Videos** — "
            )

            with gr.Row():
                with gr.Column(scale=1):
                    t5_model  = gr.Dropdown(
                        choices=model_names, value="MetaCLIP", label="Model"
                    )
                with gr.Column(scale=1):
                    t5_perp   = gr.Slider(
                        5, 100, value=30, step=1,
                        label="t-SNE Perplexity",
                        info="Local/global balance. Recommended: 20–50."
                    )
                with gr.Column(scale=1):
                    t5_niter  = gr.Slider(
                        250, 2000000, value=1000, step=250,
                        label="t-SNE Iterations",
                        info="Higher = better convergence, but slower."
                    )

            # Filter for classes to display
            t5_classes  = gr.CheckboxGroup(
                choices=DATASET_CLASSES,
                value=DATASET_CLASSES,
                label="Classes to display on the map"
            )
            with gr.Row():
                t5_btn_all  = gr.Button("Select All",   size="sm")
                t5_btn_none = gr.Button("Deselect All", size="sm")

            t5_btn_all.click(
                fn=lambda: DATASET_CLASSES,
                inputs=[],
                outputs=[t5_classes]
            )
            t5_btn_none.click(
                fn=lambda: [],
                inputs=[],
                outputs=[t5_classes]
            )

            t5_btn      = gr.Button("Generate t-SNE Map", variant="primary")
            t5_summary  = gr.Textbox(label="Summary", lines=1, interactive=False)
            t5_map      = gr.Plot(label="t-SNE Map")

            t5_btn.click(
                fn=tsne_map,
                inputs=[t5_model, t5_classes, t5_perp, t5_niter],
                outputs=[t5_map, t5_summary]
            )

    return interface


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Gradio v2 interface (5 tabs)...")
    interface = build_interface()
    interface.launch(share=False)