import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "runs" / "task2" / "outputs"
DEFAULT_MODEL = (
    ROOT_DIR
    / "runs"
    / "task2"
    / "detect"
    / "visdrone_yolov8s_lowvram"
    / "weights"
    / "best.pt"
)


def resolve_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    pattern = "visdrone_yolov8s_lowvram*/weights/best.pt"
    search_roots = [ROOT_DIR / "runs" / "task2" / "detect", ROOT_DIR / "runs" / "detect"]
    candidates = []
    for root in search_roots:
        candidates.extend(root.glob(pattern))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        selected = candidates[0]
        print(f"Default model not found, using latest trained model: {selected}")
        return selected

    raise FileNotFoundError(
        "No trained YOLO model found. Expected one of:\n"
        f"  {model_path}\n"
        f"  {ROOT_DIR / 'runs' / 'task2' / 'detect' / 'visdrone_yolov8s_lowvram-*' / 'weights' / 'best.pt'}\n"
        "Please train first with: python .\\task2_visdrone_tracking\\train.py"
    )


def make_annotators() -> tuple[sv.BoxAnnotator, sv.LabelAnnotator, sv.TraceAnnotator]:
    return (
        sv.BoxAnnotator(thickness=3),
        sv.LabelAnnotator(text_thickness=2, text_scale=0.7),
        sv.TraceAnnotator(thickness=3, trace_length=60),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection, tracking, and line counting.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--video", type=Path, default=DATA_DIR / "video.mp4")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "result_tracking_custom.mp4")
    parser.add_argument("--csv-output", type=Path, default=OUTPUT_DIR / "tracking_log.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--line-ratio", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = YOLO(str(model_path))
    tracker = sv.ByteTrack()
    box_annotator, label_annotator, trace_annotator = make_annotators()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    line_y = int(height * args.line_ratio)
    line_start = (50, line_y)
    line_end = (width - 50, line_y)
    crossed_ids: set[int] = set()
    last_center_y: dict[int, float] = {}
    total_crossed = 0

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["frame", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotated = frame.copy()
            cv2.line(annotated, line_start, line_end, (0, 255, 255), 4)

            labels = []
            if detections.tracker_id is not None:
                for xyxy, track_id, class_id, confidence in zip(
                    detections.xyxy,
                    detections.tracker_id,
                    detections.class_id,
                    detections.confidence,
                ):
                    if track_id is None:
                        continue

                    track_id = int(track_id)
                    class_name = model.names[int(class_id)]
                    confidence = float(confidence)
                    center_y = float((xyxy[1] + xyxy[3]) / 2)

                    if (
                        track_id not in crossed_ids
                        and track_id in last_center_y
                        and last_center_y[track_id] <= line_y < center_y
                    ):
                        crossed_ids.add(track_id)
                        total_crossed += 1
                    last_center_y[track_id] = center_y

                    labels.append(f"#{track_id} {class_name} {confidence:.2f}")
                    writer_csv.writerow(
                        {
                            "frame": frame_index,
                            "track_id": track_id,
                            "class_name": class_name,
                            "confidence": f"{confidence:.4f}",
                            "x1": f"{xyxy[0]:.2f}",
                            "y1": f"{xyxy[1]:.2f}",
                            "x2": f"{xyxy[2]:.2f}",
                            "y2": f"{xyxy[3]:.2f}",
                        }
                    )

            cv2.putText(
                annotated,
                f"Crossed: {total_crossed}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
            annotated = trace_annotator.annotate(scene=annotated, detections=detections)
            writer.write(annotated)

            frame_index += 1
            if frame_index % 50 == 0:
                print(f"Processed {frame_index}/{total_frames or '?'} frames")

    capture.release()
    writer.release()
    print(f"Tracking video: {args.output}")
    print(f"Tracking log: {args.csv_output}")
    print(f"Line count: {total_crossed}")


if __name__ == "__main__":
    main()
