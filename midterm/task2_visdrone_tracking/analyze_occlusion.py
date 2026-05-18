import argparse
import csv
from pathlib import Path

import cv2
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
    candidates = []
    for root in [ROOT_DIR / "runs" / "task2" / "detect", ROOT_DIR / "runs" / "detect"]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze occlusion and ID switches.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--video", type=Path, default=DATA_DIR / "video.mp4")
    parser.add_argument("--start-frame", type=int, default=165)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "silver_suv_analysis")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--target-class", type=str, default="car")
    parser.add_argument("--min-target-conf", type=float, default=0.25)
    parser.add_argument("--right-half-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = YOLO(str(model_path))
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.85)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "target_history.csv"
    switches_path = args.output_dir / "id_switches.txt"

    history = []
    saved_frames = []
    prev_id = None
    prev_center: tuple[float, float] | None = None
    switches = []

    for offset in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            break

        frame_index = args.start_frame + offset
        result = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = []
        candidates = []
        if detections.tracker_id is not None:
            for track_id, class_id, confidence, xyxy in zip(
                detections.tracker_id,
                detections.class_id,
                detections.confidence,
                detections.xyxy,
            ):
                if track_id is None:
                    continue

                track_id = int(track_id)
                class_name = model.names[int(class_id)]
                confidence = float(confidence)
                labels.append(f"#{track_id} {class_name} {confidence:.2f}")

                x_center = float((xyxy[0] + xyxy[2]) / 2)
                y_center = float((xyxy[1] + xyxy[3]) / 2)
                in_target_area = not args.right_half_only or x_center > frame.shape[1] * 0.5
                if (
                    class_name == args.target_class
                    and confidence >= args.min_target_conf
                    and in_target_area
                ):
                    candidates.append(
                        {
                            "frame": frame_index,
                            "track_id": track_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                            "center_x": x_center,
                            "center_y": y_center,
                            "area": float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])),
                        }
                    )

        if candidates:
            if prev_center is None:
                selected = max(candidates, key=lambda row: (row["center_x"], row["area"]))
            else:
                selected = min(
                    candidates,
                    key=lambda row: (row["center_x"] - prev_center[0]) ** 2
                    + (row["center_y"] - prev_center[1]) ** 2,
                )

            if prev_id is not None and selected["track_id"] != prev_id:
                switches.append((frame_index, prev_id, selected["track_id"]))
            prev_id = selected["track_id"]
            prev_center = (selected["center_x"], selected["center_y"])
            history.append(
                {
                    "frame": frame_index,
                    "track_id": selected["track_id"],
                    "class_name": selected["class_name"],
                    "confidence": selected["confidence"],
                    "x1": selected["x1"],
                    "y1": selected["y1"],
                    "x2": selected["x2"],
                    "y2": selected["y2"],
                }
            )

        if offset < 20 or offset % 8 == 0:
            annotated = frame.copy()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
            cv2.putText(
                annotated,
                "Occlusion / ID Switch Analysis",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 255),
                3,
            )
            frame_path = args.output_dir / f"frame_{frame_index:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
            saved_frames.append(frame_path)

    capture.release()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["frame", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    lines = [
        "Occlusion and ID switch summary",
        f"Video: {args.video}",
        f"Frame range: {args.start_frame}-{args.start_frame + args.frames - 1}",
        "Target rule: "
        f"class={args.target_class}, right_half_only={args.right_half_only}, "
        "continuity=nearest selected target",
        "",
    ]
    if switches:
        lines.append("ID switches:")
        lines.extend(f"Frame {frame}: ID {old} -> {new}" for frame, old, new in switches)
    else:
        lines.append("No ID switch detected by the configured target rule.")
    lines.extend(
        [
            "",
            "Notebook observation:",
            "The 6-8 second segment shows a silver SUV partly blocked by a tree trunk.",
            "The ID changes are consistent with temporary target loss and weak appearance cues.",
        ]
    )
    switches_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {len(saved_frames)} annotated frames to {args.output_dir}")
    print(f"Target history: {csv_path}")
    print(f"ID switch summary: {switches_path}")


if __name__ == "__main__":
    main()
