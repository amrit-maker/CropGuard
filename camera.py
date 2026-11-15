
from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cropguard_best.pt"


def main():
    parser = argparse.ArgumentParser(
        description="CropGuard Live Camera Detection"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold (default: 0.50)"
    )

    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        print("Error: Confidence must be between 0 and 1.")
        return

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    print("Loading CropGuard model...")
    model = YOLO(str(MODEL_PATH))

    print("Opening camera...")
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        print("Check camera permissions or try another camera index.")
        return

    print("\nCropGuard is running!")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            results = model.predict(
                source=frame,
                imgsz=640,
                conf=args.conf,
                verbose=False
            )
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()

                    print(
                        f"DETECTED: {model.names[cls]} | "
                        f"Confidence: {conf:.4f} | "
                        f"Coordinates: {coords}"
                    )
            annotated_frame = results[0].plot()

            cv2.imshow("CropGuard - Live Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                cv2.imwrite("camera_test.jpg", frame)
                cv2.imwrite("camera_test_annotated.jpg", annotated_frame)

                print("\n--- LIVE PREDICTION FOR SAVED FRAME ---")

                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()

                    print(f"Class: {model.names[cls]}")
                    print(f"Confidence: {conf:.4f}")
                    print(f"Coordinates: {coords}")

                print("Frame saved!")

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping CropGuard...")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()


