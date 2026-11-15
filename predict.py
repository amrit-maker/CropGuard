
from pathlib import Path
import argparse
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cropguard_best.pt"
OUTPUT_DIR = BASE_DIR / "outputs"

def main():
    parser = argparse.ArgumentParser(
        description="CropGuard - Crop and Weed Detection"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image or folder containing images"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )

    args = parser.parse_args()

    source = Path(args.source)

    if not source.exists():
        print(f"Error: Source not found: {source}")
        return

    if not MODEL_PATH.exists():
        print(f"Error: Model not found: {MODEL_PATH}")
        return

    print("Loading CropGuard model...")

    model = YOLO(str(MODEL_PATH))

    print("Running Crop and Weed detection...")

    results = model.predict(
        source=str(source),
        imgsz=640,
        conf=args.conf,
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
        exist_ok=True
    )

    print(f"\nPrediction completed!")
    print(f"Images processed: {len(results)}")
    print(f"Results saved in: {OUTPUT_DIR / 'predictions'}")


if __name__ == "__main__":
    main()
