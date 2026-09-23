# Generative Video Surveillance (UCF-Crime & MetaCLIP)

A complete end-to-end computer vision project designed to analyze and search through surveillance video datasets using **MetaCLIP**, **FAISS**, and an interactive **Gradio** interface.

---

##  Project Overview

This project allows users to search through video surveillance footage using natural language text queries. Instead of watching hours of video, the system converts videos into images, extracts visual features using artificial intelligence, indexes them for fast searching, and provides a web dashboard to explore the results.

---

## Project Structure

```text
Generative_Video_Surveillance/
│
├── Dataset/                     # Raw UCF-Crime video dataset (.mp4)
├── frames/                      # Extracted video frames (1 fps)
├── results/                     # Embeddings, FAISS index & evaluation outputs
├── .gitignore                   # Excludes heavy folders (Dataset/, frames/, results/)
├── requirements.txt             # Required Python packages
├── extract_frames.py            # Step 1: Extract frames from videos
├── encode_metaclip.py           # Step 2: Generate MetaCLIP embeddings
├── indexation_faiss.py          # Step 3: Build the FAISS search index
├── classification_zeroshot.py   # Step 4: Evaluate zero-shot classification
├── interface_gradio.py          # Step 5: Launch the Gradio web dashboard
└── README.md                    # Project documentation
```

## Installation

```bash
git clone https://github.com/EloiCuvelier/Generative_video_surveillance.git
cd Generative_video_surveillance
# Install system dependencies (FFmpeg is required for frame extraction)
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
pip install -r requirements.txt

```
---

## Step-by-Step Pipeline

Here is a simple explanation of each step in the project:

1. **Frame Extraction (`extract_frames.py`)**  
   We take the raw videos from the dataset and use FFmpeg to slice each video into individual still images (photos) at a rate of one frame per second. This allows us to work with separate images instead of heavy video files.
    ```bash
   python extract_frames.py
    ```
2. **Model Encoding (`encode_metaclip.py`)**
    We pass all these extracted images through the MetaCLIP AI model. The model analyzes each image and translates it into a long sequence of numbers (a vector), converting visual content into a mathematical format that the computer can process.
    ```bash 
    python encode_metaclip.py
     ```
3. **FAISS Indexing (`indexation_faiss.py`)**  
   We gather all these mathematical vectors into an ultra-fast database called FAISS. This enables instant similarity searches without having to scan through every single image one by one.
   ```bash
   python indexation_faiss.py
   ```

4. **Zero-Shot Classification (`classification_zeroshot.py`)**  
   We test the model's ability to automatically recognize different categories in our dataset (such as fights, accidents, or thefts) without needing prior task-specific training on them.
   ```bash
   python classification_zeroshot.py
   ```
5. **Interactive Web Interface (`interface_gradio.py`)**  
   We launch a visual dashboard on a web page using Gradio. It features 5 tabs to search for images using text queries, rank entire video clips, detect temporal time sequences, display similarity curves, or explore an interactive 2D map (t-SNE).
   ```bash
   python interface_gradio.py
   ```
## Web Dashboard Overview (Gradio)

### Tab 1 — Frame-Level Search
Returns the top N frames from the dataset most similar to your natural language query.
<img width="1532" height="951" alt="Tab 1 Frames" src="https://github.com/user-attachments/assets/64f98794-44a7-4931-ae61-b73dbf52f3b9" />

### Tab 2 — Video-Level Search
Aggregates the scores of all frames in each video (average of the top-K frames). The thumbnail shown is the most similar frame of the video.
<img width="1527" height="957" alt="Tab 2 Videos" src="https://github.com/user-attachments/assets/333acb4d-b473-4560-bafb-f9d38f5abe8c" />

### Tab 3 — Sequence Detection
Sliding window over the frames of each video. Returns the continuous segments most similar to the query with their timestamp interval `[t_start → t_end]`.
<img width="1527" height="947" alt="Tab 3 Sequences" src="https://github.com/user-attachments/assets/cddd640c-2b38-40fd-931e-afa5c4632e8a" />

### Tab 4 — Class Similarity Timeline
Select a video from the dataset. The graph plots, frame by frame, the cosine similarity with each of the 13 classes. Class embeddings are computed via prompt ensembling: averaging 5 description vectors per class.
<img width="1580" height="837" alt="Tab 4 Class Similarity" src="https://github.com/user-attachments/assets/cdad0577-3f7b-4366-8ac2-6989ff2351a8" />

### Tab 5 — 2D t-SNE Embedding Map
2D projection of the 512D MetaCLIP video embeddings, preserving distances between vectors to reveal clusters by crime class.
<img width="1542" height="942" alt="Tab 5 t-SNE Map" src="https://github.com/user-attachments/assets/3ad9865f-4be9-4ffb-bdfb-ecb6d01b8ffb" />



---

## Model Selection & Benchmarking

Originally conducted as a collaborative research project, this repository focuses exclusively on **MetaCLIP**, which was selected following a comparative evaluation across 8 vision-language architectures on zero-shot surveillance anomaly classification.

<p align="center">
  <img width="2000" height="437" alt="image" src="https://github.com/user-attachments/assets/da301ae9-5678-4a56-b0eb-48da24f1ae4d" />
 
</p>

*Models were evaluated across single-frame inference and multiple temporal aggregation strategies (`video_mean`, `video_best`, `video_vote`) at Top-1, Top-3, and Top-5 accuracy.*

