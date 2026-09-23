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
├── .gitignore                   # Excludes heavy datasets and temporary results
├── requirements.txt             # List of required Python packages and versions
├── extract_frames.py            # Step 1: Extract images from videos
├── encode_metaclip.py           # Step 2: Convert images into AI embeddings
├── indexation_faiss.py          # Step 3: Build the fast search index
├── classification_zeroshot.py   # Step 4: Evaluate zero-shot classification
└── interface_gradio.py          # Step 5: Launch the interactive web app
```

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
<img width="1532" height="951" alt="image" src="https://github.com/user-attachments/assets/64f98794-44a7-4931-ae61-b73dbf52f3b9" />
Tab 1 Frames : Search by individual frames — Returns the top N frames from the dataset most similar to your query.


<img width="1527" height="957" alt="image" src="https://github.com/user-attachments/assets/333acb4d-b473-4560-bafb-f9d38f5abe8c" />
Tab 2 Videos : Search by videos — Aggregates the scores of all frames in each video (average of the top-K frames). The thumbnail shown is the most similar frame of the video.


<img width="1527" height="947" alt="image" src="https://github.com/user-attachments/assets/cddd640c-2b38-40fd-931e-afa5c4632e8a" />
Tab 3 Sequences : Search by sequences — Sliding window over the frames of each video. Returns the continuous segments most similar to the query with their timestamp interval [t_start → t_end].

<img width="1580" height="837" alt="image" src="https://github.com/user-attachments/assets/cdad0577-3f7b-4366-8ac2-6989ff2351a8" />
Tab 4 Class Similarity :  Select a video from the dataset. The graph plots, frame by frame, the cosine similarity with each of the 13 classes. Class embeddings are computed via prompt ensembling: averaging 5 description vectors per class.

<img width="1542" height="942" alt="image" src="https://github.com/user-attachments/assets/3ad9865f-4be9-4ffb-bdfb-ecb6d01b8ffb" />
Tab 5 t-SNE Map : 2D projection of the 512D MetaCLIP video embeddings, preserving distances between vectors to reveal clusters by crime class.




---

## Note on Model Selection & Benchmarking

Originally conducted as a collaborative group project, this repository focuses exclusively on MetaCLIP, which emerged as the optimal architecture after a rigorous comparative study benchmarking 8 different vision-language models.

---

