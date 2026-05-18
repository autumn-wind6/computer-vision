import argparse
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve
from zipfile import BadZipFile, ZipFile

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DATA_DIR = ROOT_DIR / "data"

VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

VISDRONE_DET_SPLITS = {
    "VisDrone2019-DET-train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}


def split_is_ready(visdrone_root: Path, split: str) -> bool:
    split_dir = visdrone_root / split
    return (split_dir / "images").exists() and (split_dir / "annotations").exists()


def download_file(url: str, destination: Path, retries: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded * 100 / total_size
        print(f"\rDownloading {destination.name}: {percent:5.1f}%", end="")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Download attempt {attempt}/{retries}")
            urlretrieve(url, destination, reporthook)
            print()
            return
        except (URLError, OSError) as exc:
            last_error = exc
            print()
            print(f"Download failed: {exc}")
            if destination.exists():
                destination.unlink()
            if attempt < retries:
                time.sleep(min(5 * attempt, 20))

    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def download_visdrone_det(visdrone_root: Path) -> None:
    """Download and extract the VisDrone DET train/val splits if missing."""
    visdrone_root.mkdir(parents=True, exist_ok=True)
    archive_dir = visdrone_root / "_downloads"

    for split, url in VISDRONE_DET_SPLITS.items():
        if split_is_ready(visdrone_root, split):
            print(f"{split} already exists, skipping download.")
            continue

        archive_path = archive_dir / f"{split}.zip"
        if not archive_path.exists():
            print(f"{split} not found. Downloading from:\n  {url}")
            download_file(url, archive_path)
        else:
            print(f"Using existing archive: {archive_path}")

        print(f"Extracting {archive_path.name}...")
        try:
            with ZipFile(archive_path) as zip_file:
                zip_file.extractall(visdrone_root)
        except BadZipFile:
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{archive_path} is not a valid zip file. Re-run this script to download it again."
            )

        if not split_is_ready(visdrone_root, split):
            raise FileNotFoundError(
                f"Downloaded {split}, but expected images/ and annotations/ were not found."
            )

    print("VisDrone DET download/check complete.")


def convert_visdrone_to_yolo(visdrone_root: Path) -> None:
    """Convert VisDrone DET annotations to YOLO txt labels in-place."""
    visdrone_root = visdrone_root.resolve()
    if not visdrone_root.exists():
        raise FileNotFoundError(
            "VisDrone root does not exist.\n"
            f"Expected: {visdrone_root}\n"
            "Please unzip/copy the dataset so it contains:\n"
            "  VisDrone2019-DET-train/images\n"
            "  VisDrone2019-DET-train/annotations\n"
            "  VisDrone2019-DET-val/images\n"
            "  VisDrone2019-DET-val/annotations\n"
            "Or pass the real location with --visdrone-root."
        )

    print(f"Dataset root: {visdrone_root}")

    for split in ["VisDrone2019-DET-train", "VisDrone2019-DET-val"]:
        image_dir = visdrone_root / split / "images"
        annotation_dir = visdrone_root / split / "annotations"
        label_dir = visdrone_root / split / "labels"
        if not image_dir.exists() or not annotation_dir.exists():
            raise FileNotFoundError(
                "VisDrone split is incomplete. Expected:\n"
                f"  {image_dir}\n"
                f"  {annotation_dir}\n"
                "Please put the original VisDrone images and annotations under data/VisDrone."
            )

        label_dir.mkdir(parents=True, exist_ok=True)

        annotation_files = sorted(annotation_dir.glob("*.txt"))
        print(f"Processing {split}: {len(annotation_files)} annotation files")

        for annotation_file in annotation_files:
            image_file = image_dir / annotation_file.with_suffix(".jpg").name
            if not image_file.exists():
                continue

            with Image.open(image_file) as image:
                image_width, image_height = image.size

            yolo_lines = []
            with annotation_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue

                    try:
                        x, y, width, height = map(float, parts[:4])
                        category = int(parts[5])
                    except ValueError:
                        continue

                    if category == 0 or category > len(VISDRONE_NAMES):
                        continue

                    if width <= 0 or height <= 0:
                        continue

                    class_id = category - 1
                    x_center = (x + width / 2) / image_width
                    y_center = (y + height / 2) / image_height
                    width_norm = width / image_width
                    height_norm = height / image_height

                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width_norm = max(0.0, min(1.0, width_norm))
                    height_norm = max(0.0, min(1.0, height_norm))

                    yolo_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} "
                        f"{width_norm:.6f} {height_norm:.6f}"
                    )

            if yolo_lines:
                (label_dir / annotation_file.name).write_text(
                    "\n".join(yolo_lines), encoding="utf-8"
                )

    print("Label conversion complete.")


def write_dataset_yaml(visdrone_root: Path, output_yaml: Path) -> None:
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = os.path.relpath(visdrone_root.resolve(), output_yaml.parent.resolve())
    dataset_path = Path(dataset_path).as_posix()
    names = "\n".join(f"  - {name}" for name in VISDRONE_NAMES)
    content = f"""path: {visdrone_root.as_posix()}
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images

nc: {len(VISDRONE_NAMES)}
names:
{names}
"""
    content = content.replace(visdrone_root.as_posix(), dataset_path)
    output_yaml.write_text(content, encoding="utf-8")
    print(f"Wrote {output_yaml}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VisDrone DET labels for YOLO.")
    parser.add_argument("--visdrone-root", type=Path, default=DATA_DIR / "VisDrone")
    parser.add_argument("--output-yaml", type=Path, default=PROJECT_DIR / "visdrone_det.yaml")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download missing VisDrone DET train/val splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_download:
        download_visdrone_det(args.visdrone_root)
    convert_visdrone_to_yolo(args.visdrone_root)
    write_dataset_yaml(args.visdrone_root, args.output_yaml)


if __name__ == "__main__":
    main()
