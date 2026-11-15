# CropGuard

A small object detection project built using YOLO11 to identify crops and weeds in agricultural field images.

I trained a YOLO11n model on a crop and weed dataset. The model detects two classes: `Crop` and `Weed`, and draws bounding boxes around detected objects.

## What it does

- Detects crops and weeds in images.
- Shows bounding boxes with class labels and confidence scores.
- Works on a single image or a complete folder.
- Supports real-time detection using a webcam.
- Saves prediction results automatically.
- Allows changing the confidence threshold.

## Model details

| Detail | Value |
|---|---|
| Model | YOLO11n |
| Framework | Ultralytics |
| Task | Object Detection |
| Classes | Crop, Weed |
| Image size | 640 |
| Default confidence | 0.25 |

## Test results

I evaluated the trained model on 412 test images containing 3,394 annotated objects.

| Metric | Result |
|---|---:|
| Precision | 67.7% |
| Recall | 67.2% |
| mAP50 | 71.1% |
| mAP50-95 | 50.9% |

### Class-wise results

| Metric | Crop | Weed |
|---|---:|---:|
| Precision | 76.8% | 58.6% |
| Recall | 79.9% | 54.5% |
| mAP50 | 85.6% | 56.6% |
| mAP50-95 | 67.1% | 34.6% |

Crop detection is currently better than weed detection. Improving weed detection is one of the main things I would like to work on next.

## Project structure

```text
CropGuard/
│
├── models/
│   └── cropguard_best.pt
│
├── sample_images/
│
├── camera.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `outputs/` folder is created automatically when predictions are run. Generated prediction images are not included in the repository.

## Setup

You need Python 3.10 or later installed on your system.

### 1. Clone the repository

```bash
git clone https://github.com/amrit-maker/CropGuard.git
cd CropGuard
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the model

### Predict on a single image

```bash
python predict.py --source "sample_images/ave-0045-0013.jpg"
```

### Predict on a folder

```bash
python predict.py --source "sample_images"
```

### Change confidence threshold

```bash
python predict.py --source "sample_images/ave-0045-0013.jpg" --conf 0.40
```

The default confidence threshold is 0.25.

Prediction results are saved in:

```text
outputs/predictions/
```

## Live Camera Detection

CropGuard also supports real-time detection using a webcam.

Run:

```bash
python camera.py
```

The default camera device is 0.

You can change the confidence threshold:

```bash
python camera.py --conf 0.60
```

If you have multiple cameras, you can select another one:

```bash
python camera.py --camera 1
```

Press `Q` to close the camera window.

## Requirements

- Python 3.10+
- Ultralytics 8.4.211
- PyTorch
- OpenCV (for live camera detection)
- NVIDIA GPU recommended, but CPU inference is also supported.

Dependencies are listed in `requirements.txt`.

## Future work

Some things I would like to improve:

- Better weed detection.
- Training with more diverse agricultural images.
- Testing on images captured in real field conditions.
- Improving live camera detection under different lighting conditions.

## Note

This is an ongoing project. The model is still experimental, and its predictions may not always be correct, especially in cases where weeds are difficult to distinguish from crops.
