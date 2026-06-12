import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from pipeline import apply_perspective, extract_corners

# .run_pipeline.py /path/to/input_images /path/to/output_images --model best.pt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the bending-machine pipeline on every image in a folder."
    )
    parser.add_argument(
        "input_dir",
        help="Folder containing input images.",
    )
    parser.add_argument(
        "output_dir",
        help="Folder where processed images and report.csv will be saved.",
    )
    parser.add_argument(
        "--model",
        default="best.pt",
        help="Path to YOLO segmentation model. Default: best.pt",
    )
    parser.add_argument(
        "--method",
        choices=["ApproxPolyDP", "MinAreaRect"],
        default="ApproxPolyDP",
        help="Corner extraction method. Default: ApproxPolyDP",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="ApproxPolyDP epsilon factor. Default: 0.05",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Optional YOLO confidence threshold.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Only save final warped images, without mask/corner preview images.",
    )
    return parser.parse_args()


def find_images(input_dir):
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def save_rgb(path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask * 255).astype(np.uint8))


def draw_corners(image_rgb, corners):
    preview = image_rgb.copy()
    for point in corners:
        cv2.circle(
            preview,
            (int(point[0]), int(point[1])),
            15,
            (0, 255, 0),
            -1,
        )
    return preview


def run_segmentation_quiet(image_rgb, model, confidence_threshold):
    predict_kwargs = {
        "source": image_rgb,
        "verbose": False,
    }
    if confidence_threshold is not None:
        predict_kwargs["conf"] = confidence_threshold

    results = model.predict(**predict_kwargs)
    if len(results) == 0 or results[0].masks is None:
        return None, None

    mask = results[0].masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
    confidence = float(results[0].boxes[0].conf)

    return mask, confidence


def run_one_image(image_path, output_dir, relative_path, model, args):
    image_rgb = load_image_rgb(image_path)

    mask, confidence = run_segmentation_quiet(image_rgb, model, args.conf)

    if mask is None:
        return {
            "image": str(relative_path),
            "status": "failed",
            "confidence": "",
            "message": "No mask detected",
        }

    corners = extract_corners(
        mask,
        method=args.method,
        epsilon_factor=args.epsilon,
    )

    if corners is None:
        if not args.no_debug:
            save_mask(output_dir / "debug" / "masks" / relative_path.with_suffix(".png"), mask)
        return {
            "image": str(relative_path),
            "status": "failed",
            "confidence": f"{confidence:.4f}",
            "message": "Could not extract exactly 4 corners",
        }

    warped = apply_perspective(image_rgb, corners)
    output_path = output_dir / "warped" / relative_path.with_suffix(".png")
    save_rgb(output_path, warped)

    if not args.no_debug:
        save_mask(output_dir / "debug" / "masks" / relative_path.with_suffix(".png"), mask)
        corner_preview = draw_corners(image_rgb, corners)
        save_rgb(output_dir / "debug" / "corners" / relative_path.with_suffix(".png"), corner_preview)

    return {
        "image": str(relative_path),
        "status": "ok",
        "confidence": f"{confidence:.4f}",
        "message": str(output_path),
    }


def write_report(output_dir, rows):
    report_path = output_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "status", "confidence", "message"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = find_images(input_dir)

    if not image_paths:
        raise ValueError(f"No supported image files found in: {input_dir}")

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(input_dir)
        print(f"[{index}/{len(image_paths)}] {relative_path}")
        try:
            rows.append(
                run_one_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    relative_path=relative_path,
                    model=model,
                    args=args,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "image": str(relative_path),
                    "status": "failed",
                    "confidence": "",
                    "message": str(exc),
                }
            )

    report_path = write_report(output_dir, rows)
    ok_count = sum(row["status"] == "ok" for row in rows)

    print(f"Done: {ok_count}/{len(rows)} images processed successfully")
    print(f"Warped images: {output_dir / 'warped'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
