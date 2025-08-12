# Driver Behavior Assessment & Traffic Sign Detection (YOLOv8 + Gemini)

A two-part project:
1) **Traffic Sign Detection** using YOLOv8.
2) **Driver Behavior Assessment** that interprets detections with **Google Gemini** to generate DMV-style feedback.

This repo includes two notebooks:
- `notebooks/traffic_signs_detection_using_yolov8.ipynb`
- `notebooks/Driver_Behavior_YOLO_Gemini.ipynb`

---

## Features
- Train a YOLOv8 model for traffic sign detection.
- Run inference on images/videos and export detections (boxes, classes, confidences).
- Feed detections + frames into Gemini and get structured driver feedback
  (e.g., Situation Summary, Rule Violated, Score Adjustment, Recommendation).

---

## Quick Start

### 1) Clone & setup environment
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (recommended) create a virtual env
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> If you are on a headless server (no GUI), replace `opencv-python` with `opencv-python-headless` in `requirements.txt`.

### 2) Configure Gemini API key
Create a `.env` file in the project root (or set an environment variable).

```
GOOGLE_API_KEY=your_api_key_here
```

Or on macOS/Linux:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 3) Launch Jupyter / open notebooks
```bash
python -m pip install jupyter
jupyter notebook
# then open:
#  - notebooks/traffic_signs_detection_using_yolov8.ipynb
#  - notebooks/Driver_Behavior_YOLO_Gemini.ipynb
```

---

## Project Structure (suggested)
```
<repo-root>/
├─ notebooks/
│  ├─ traffic_signs_detection_using_yolov8.ipynb
│  └─ Driver_Behavior_YOLO_Gemini.ipynb
├─ data/
│  ├─ images/            # your images or video frames
│  └─ videos/            # optional, raw videos
├─ outputs/
│  ├─ detections/        # json/csv of detections exported from YOLO
│  └─ reports/           # Gemini evaluation outputs
├─ models/               # trained weights
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## Training (YOLOv8)
Use Ultralytics’ CLI. Example:
```bash
# Example: train a small model
yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=50 imgsz=640
# Weights will be saved under runs/detect/train/weights/best.pt
```

**Data config (`data.yaml`)** should define your train/val paths and class names. Example:
```yaml
path: ./data
train: images/train
val: images/val
names:
  0: speed_25
  1: speed_30
  2: red_light
  3: green_light
  # ... add your classes
```

---

## Inference (YOLOv8)
```bash
# Predict on images or a video
yolo predict model=runs/detect/train/weights/best.pt source=data/images     # folder
yolo predict model=runs/detect/train/weights/best.pt source=data/videos/clip.mp4
# Results are saved in runs/detect/predict/
```

You can also run inference programmatically:
```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("data/images/your_image.jpg")
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy()
```

Save detections to JSON/CSV for the behavior step.

---

## Driver Behavior Assessment (Gemini)
The second notebook loads detections + a reference frame and asks Gemini to generate structured feedback.

Minimal example:
```python
import os, json
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# detections from YOLO (your exported JSON)
with open("outputs/detections/frame_0123.json", "r") as f:
    det = json.load(f)

frame = Image.open("data/images/frame_0123.jpg")

prompt = """
You are a driving examiner. Based on the detections and frame, evaluate the driver's action.
Return fields: Situation Summary, Driver Action Assessment, Rule Violated, Score Adjustment, Recommendation.
"""

model = genai.GenerativeModel("gemini-1.5-flash")
resp = model.generate_content([prompt, str(det), frame])
print(resp.text)
```

Tip: keep your prompts consistent and versioned for reproducible results.

---

## Requirements

If you don’t already have a `requirements.txt`, start with this (covers both notebooks):

```
ipython
matplotlib
numpy
opencv-python
pandas
pillow
ray
seaborn
tqdm
ultralytics
google-generativeai
```

For Google Colab add:
```
google-colab
```

> If notebook cells call `!pip install ...`, prefer moving those to `requirements.txt` so the environment is reproducible.

---

## Repro Tips
- Pin versions once your pipeline is stable (example: `ultralytics==8.3.*`, `numpy==1.26.*`).
- Save model weights under `models/`.
- Save raw detections under `outputs/detections/` and Gemini responses under `outputs/reports/`.
- For headless servers, use `opencv-python-headless`.

---

## Contributing
Issues and PRs welcome! Please include a short description, reproduction steps, and environment details.

---

## License
MIT (or choose another). Add a `LICENSE` file if you want others to reuse the code.
