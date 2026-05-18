import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on VisDrone DET.")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "visdrone_det.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT_DIR / "yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="visdrone_yolov8s_lowvram")
    parser.add_argument("--project", type=Path, default=ROOT_DIR / "runs" / "task2" / "detect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(args.weights))
    print("Starting VisDrone YOLOv8 fine-tuning...")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        patience=15,
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.05,
        degrees=10.0,
        translate=0.1,
        scale=0.7,
        shear=5.0,
        fliplr=0.5,
        flipud=0.2,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cache=False,
        amp=True,
        workers=4,
        pretrained=True,
        val=True,
        save=True,
        save_period=5,
    )

    best_path = Path(args.project).resolve() / args.name / "weights" / "best.pt"
    print(f"Training complete. Best model: {best_path}")


if __name__ == "__main__":
    main()
