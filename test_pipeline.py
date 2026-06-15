import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from pipeline import apply_perspective, extract_corners, extract_largest_contour, run_segmentation


MODEL_PATH = "best.pt"
OUTPUT_DIR = "outputs"
CORNER_METHOD = "ApproxPolyDP"
EPSILON_FACTOR = 0.05
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def save_rgb(path, image_np):
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask * 255).astype(np.uint8))


def draw_corners(image_np, corners):
    image_corners = image_np.copy()
    for point in corners:
        cv2.circle(
            image_corners,
            (int(point[0]), int(point[1])),
            15,
            (0, 255, 0),
            -1,
        )
    return image_corners


def find_images(input_folder_path):
    input_folder_path = Path(input_folder_path)
    return sorted(
        path
        for path in input_folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_one_image(image_path, model):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Cannot find image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Step 1: segmentation, same as app.py
    mask, conf = run_segmentation(image_np, model)
    if mask is None:
        raise RuntimeError("Model cannot detect a mask on this image.")

    # Step 2: corner extraction
    corners = extract_corners(
        mask,
        method=CORNER_METHOD,
        epsilon_factor=EPSILON_FACTOR,
    )
    if corners is None:
        raise RuntimeError(
            f"{CORNER_METHOD} cannot find exactly 4 corners. "
            f"Try changing EPSILON_FACTOR, current value: {EPSILON_FACTOR}"
        )

    image_corners = draw_corners(image_np, corners)

    # Step 3: perspective transform
    warped_img = apply_perspective(image_np, corners)

    return mask, image_corners, warped_img, conf


def main(image_path):
    output_dir = Path(OUTPUT_DIR)
    model = YOLO(MODEL_PATH)
    mask, image_corners, warped_img, conf = run_one_image(image_path, model)

    save_mask(output_dir / "01_mask.png", mask)
    save_rgb(output_dir / "02_corners.png", image_corners)
    save_rgb(output_dir / "03_output.png", warped_img)

    print(f"Done: {image_path}")
    print(f"Confidence: {conf:.2f}")
    print(f"Saved results to: {output_dir}")


def process_images(input_folder_path):
    input_folder_path = Path(input_folder_path)
    all_steps_dir = Path(OUTPUT_DIR) / "all_steps"

    if not input_folder_path.is_dir():
        raise FileNotFoundError(f"Cannot find input folder: {input_folder_path}")

    image_paths = find_images(input_folder_path)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_folder_path}")

    model = YOLO(MODEL_PATH)
    success_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(input_folder_path)
        output_name = relative_path.with_suffix(".png")

        print(f"[{index}/{len(image_paths)}] {relative_path}")

        try:
            mask, image_corners, warped_img, conf = run_one_image(image_path, model)

            save_mask(all_steps_dir / "masks" / output_name, mask)
            save_rgb(all_steps_dir / "corners" / output_name, image_corners)
            save_rgb(all_steps_dir / "outputs" / output_name, warped_img)

            success_count += 1
            print(f"  OK, confidence: {conf:.2f}")
        except Exception as exc:
            print(f"  Failed: {exc}")

    print(f"Done: {success_count}/{len(image_paths)} images processed")
    print(f"All steps saved to: {all_steps_dir}")


if __name__ == "__main__":
    # Single image:
    main("images/0000_random-102-_jpg.rf.373979855e58b83d18e0b4222960bbe3.jpg")

    # Folder of images:
    # process_images("images")
