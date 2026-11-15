# CropGuard 

A small object detection project built using YOLO11 to identify crops and weeds in agricultural field images.

I trained a YOLO11n model on a crop and weed dataset. The model detects two classes: `Crop` and `Weed`, and draws bounding boxes around the detected objects.

## What it does

- Detects crops and weeds in images.
- Shows bounding boxes with class labels and confidence scores.
- Works on a single image or a complete folder.
- Saves the output images automatically.
- Allows changing the confidence threshold.

## Model details

| Detail | Value |
|---|---|
| Model | YOLO11n |
| Framework | Ultralytics |
| Task | Object detection |
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

Class-wise results:

| Metric | Crop | Weed |
|---|---:|---:|
| Precision | 76.8% | 58.6% |
| Recall | 79.9% | 54.5% |
| mAP50 | 85.6% | 56.6% |
| mAP50-95 | 67.1% | 34.6% |

Crop detection is currently better than weed detection. Improving weed detection is one of the main things I would like to work on next.

## Folder structure

```text
CropGuard/
│
├── models/
│   └── cropguard_best.pt
│
├── sample_images/
├── outputs/
├── predict.py
├── requirements.txt
└── README.md
```

## Setup

You need Python installed on your system.

First, clone the repository:

```bash
git clone https://github.com/amrit-maker/CropGuard.git
cd CropGuard
```

Create a virtual environment:

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

Install the required library:

```bash
pip install -r requirements.txt
```

## Running the model

To run CropGuard on one image:

```bash
python predict.py --source "path/to/image.jpg"
```

To run it on a folder containing multiple images:

```bash
python predict.py --source "path/to/folder"
```

You can also change the confidence threshold:

```bash
python predict.py --source "path/to/image.jpg" --conf 0.40
```

The default confidence threshold is 0.25.

The predicted images are saved in:

```text
outputs/predictions/
```

## Requirements

- Python 3.10+
- Ultralytics 8.4.211
- PyTorch (installed as an Ultralytics dependency)
- NVIDIA GPU recommended, but CPU inference is also supported.

## Future work

Some things I would like to improve:

- Better weed detection.
- Training with more diverse agricultural images.
- Testing on images captured in real field conditions.
- Adding real-time camera detection.

## Note

This is an ongoing project. The model is still experimental, and its predictions may not always be correct, especially in cases where weeds are difficult to distinguish from crops.

