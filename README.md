# Video Object Segmentation & Tracking (SAM3)

End-to-end pipeline for video object segmentation and tracking with
mask propagation across time using **Segment Anything Model v3 (SAM3)**.


---

## Key Features

- Video segmentation using text prompts (people, vehicles, license plates) with mask propagation over time  
- Multi-class segmentation via independent model passes  
- Custom visualization module with masks, bounding boxes, and object IDs  

---

## Tech Stack

- Python  
- PyTorch  
- CUDA  
- OpenCV  
- NumPy  
- SAM3  

---

## Project Structure
```
video-object-segmentation-sam3/
│
├─ data/            # input videos and intermediate outputs (ignored in repo)
├─ notebookfiles/   # Jupyter notebook demonstrating the pipeline
├─ scripts/         # segmentation and visualization scripts
├─ video_outputs/   # final segmented video
├─ sam3/            # external SAM3 repository (see Installation)
├─ requirements.txt
└─ README.md
```

---

## Installation

```bash
git clone https://github.com/your_username/video-object-segmentation-sam3.git
cd video-object-segmentation-sam3
git clone https://github.com/facebookresearch/sam3
pip install -r requirements.txt
```

## Usage

- **Notebook workflow**  
  `notebookfiles/video_segmentation.ipynb`  
  Demonstrates single-class video segmentation and saving intermediate results.

- **Visualization script**  
  `scripts/put_masks_on_video.py`  
  Combines segmentation outputs and renders final video with masks, bounding boxes, and object IDs.


# Results

The repository includes a short (~25 seconds) example video demonstrating
people segmentation and tracking with stable mask propagation.