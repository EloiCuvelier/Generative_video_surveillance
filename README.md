# Generative Video Surveillance (UCF-Crime & MetaCLIP)

A complete end-to-end computer vision project designed to analyze and search through surveillance video datasets using **MetaCLIP**, **FAISS**, and an interactive **Gradio** interface.

---

##  Project Overview

This project allows users to search through video surveillance footage using natural language text queries. Instead of watching hours of video, the system converts videos into images, extracts visual features using artificial intelligence, indexes them for fast searching, and provides a web dashboard to explore the results.

---

## Project Structure


Generative_Video_Surveillance/
│
├── .gitignore                   # Excludes heavy datasets and temporary results
├── requirements.txt             # List of required Python packages and versions
├── extract_frames.py            # Step 1: Extract images from videos
├── encode_metaclip.py           # Step 2: Convert images into AI embeddings
├── indexation_faiss.py          # Step 3: Build the fast search index
├── classification_zeroshot.py   # Step 4: Evaluate zero-shot classification
└── interface_gradio.py          # Step 5: Launch the interactive web app


---

## Installation

```bash
git clone [https://github.com/EloiCuvelier/Generative_video_surveillance.git](https://github.com/EloiCuvelier/Generative_video_surveillance.git)
cd Generative_video_surveillance
pip install -r requirements.txt

```
---

## Step-by-Step Pipeline

Here is a simple explanation of each step in the project:

1. **Frame Extraction (`extract_frames.py`)**  
   We take the raw videos from the dataset and use FFmpeg to slice each video into individual still images (photos) at a rate of one frame per second. This allows us to work with separate images instead of heavy video files.
   python extract_frames.py

2. **Model Encoding (`encode_metaclip.py`)**
    We pass all these extracted images through the MetaCLIP AI model. The model analyzes each image and translates it into a long sequence of numbers (a vector), converting visual content into a mathematical format that the computer can process.
    
    python encode_metaclip.py

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

---

## Note on Model Selection & Benchmarking

Originally conducted as a collaborative group project, this repository focuses exclusively on **MetaCLIP**, which emerged as the optimal architecture after a rigorous comparative study benchmarking **8 different vision-language models** (including various standard CLIP checkpoints and MobileCLIP). The complete evaluation logs, comparative metrics, and performance charts are preserved in the `results/` directory to document this engineering decision.

---

