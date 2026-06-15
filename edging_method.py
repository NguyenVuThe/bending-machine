import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from scipy.interpolate import interp1d
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import warp

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

def nearest_idx(contour, point):
    d = np.linalg.norm(contour - point, axis=1)
    return np.argmin(d)

def draw_corners(image_np, corners):
    image_corners = image_np.copy()

    labels = ["TL", "TR", "BR", "BL"]

    for point, label in zip(corners, labels):

        x, y = map(int, point)

        cv2.circle(
            image_corners,
            (x, y),
            15,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            image_corners,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return image_corners

def draw_contours(image_np, largest_contour):
    image_contours = image_np.copy()
    cv2.drawContours(image_contours, [largest_contour], -1, (0, 255, 0), 3)
    return image_contours

def draw_edges(image_np, left_edge, bottom_edge, right_edge, top_edge):
    image_edges = image_np.copy()

    for p in left_edge:
        cv2.circle(image_edges, tuple(p), 2, (255,0,0), -1)

    for p in bottom_edge:
        cv2.circle(image_edges, tuple(p), 2, (0,255,0), -1)

    for p in right_edge:
        cv2.circle(image_edges, tuple(p), 2, (0,0,255), -1)

    for p in top_edge:
        cv2.circle(image_edges, tuple(p), 2, (255,255,0), -1)

    return image_edges

def draw_resampled_edges(image_np, top, bottom):
    resample_edges = image_np.copy()

    for p in top:
        cv2.circle(resample_edges, tuple(p.astype(int)), 2, (0,0,255), -1)

    for p in bottom:
        cv2.circle(resample_edges, tuple(p.astype(int)), 2, (255,0,0), -1)

    return resample_edges

def draw_mesh(image_np, mesh):
    image_mesh = image_np.copy()

    for column in mesh:
        for p in column:
            cv2.circle(image_mesh, tuple(p.astype(int)), 2, (0,255,0), -1)

    return image_mesh

def resample_curve(points, n_points=100):

    points = np.asarray(points)

    dist = np.sqrt(
        np.sum(
            np.diff(points, axis=0)**2,
            axis=1
        )
    )

    dist = np.insert(np.cumsum(dist), 0, 0)

    fx = interp1d(dist, points[:, 0])
    fy = interp1d(dist, points[:, 1])

    new_dist = np.linspace(
        0,
        dist[-1],
        n_points
    )

    x_new = fx(new_dist)
    y_new = fy(new_dist)

    return np.column_stack([x_new, y_new])



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
    print(f"Extracted corners: {corners}")
    if corners is None:
        raise RuntimeError(
            f"{CORNER_METHOD} cannot find exactly 4 corners. "
            f"Try changing EPSILON_FACTOR, current value: {EPSILON_FACTOR}"
        )

    image_corners = draw_corners(image_np, corners)
    TL, TR, BR, BL = corners
    largest_contour = extract_largest_contour(mask)
    boundary = largest_contour.squeeze()

    idx_tl = nearest_idx(boundary, TL)
    idx_tr = nearest_idx(boundary, TR)
    idx_br = nearest_idx(boundary, BR)
    idx_bl = nearest_idx(boundary, BL)

    print(
        "TL:", idx_tl,
        "TR:", idx_tr,
        "BR:", idx_br,
        "BL:", idx_bl
    )

    left_edge = boundary[idx_tl:idx_bl+1]

    bottom_edge = boundary[idx_bl:idx_br+1]

    right_edge = boundary[idx_br:idx_tr+1]

    top_edge = np.concatenate([
        boundary[idx_tr:],
        boundary[:idx_tl+1]
    ])

    image_edges = draw_edges(image_np, left_edge, bottom_edge, right_edge, top_edge)
    save_rgb(Path(OUTPUT_DIR) / "debug_edges.png", image_edges)

    print("Length of top edge:", len(top_edge))
    print("Length of bottom edge:", len(bottom_edge))
    print("Length of left edge:", len(left_edge))
    print("Length of right edge:", len(right_edge))

    top = resample_curve(top_edge, 100)
    bottom = resample_curve(bottom_edge, 100)

    left = resample_curve(left_edge, 60)
    right = resample_curve(right_edge, 60)

    rows = 60

    resample_edges = draw_resampled_edges(image_np, top, bottom)
    save_rgb(Path(OUTPUT_DIR) / "debug_resample_edges.png", resample_edges)

    mesh = []

    top = top[::-1]

    for i in range(100):

        column = np.linspace(
            top[i],
            bottom[i],
            rows
        )

        mesh.append(column)

    mesh = np.array(mesh)

    image_mesh = draw_mesh(image_np, mesh)
    save_rgb(Path(OUTPUT_DIR) / "debug_mesh.png", image_mesh)

    cols = mesh.shape[0]
    rows = mesh.shape[1]

    W = 1000
    H = 1400

    target_mesh = []

    for i in range(cols):

        x = i * W / (cols - 1)

        column = []

        for j in range(rows):

            y = j * H / (rows - 1)

            column.append([x, y])

        target_mesh.append(column)

    target_mesh = np.array(target_mesh)

    src_pts = mesh.reshape(-1, 2)
    dst_pts = target_mesh.reshape(-1, 2)

    tform = PiecewiseAffineTransform()

    tform.estimate(
        dst_pts,
        src_pts
    )

    warped = warp(
        image_np,
        tform,
        output_shape=(H, W)
    )
    warped = (warped * 255).astype(np.uint8)
    cv2.imwrite(
        "dewarped.png",
        cv2.cvtColor(
            warped,
            cv2.COLOR_RGB2BGR
        )
    )
    # Step 2.5: contour extraction (dành cho debug, không dùng trong app.py)
    # largest_contour = extract_largest_contour(mask)
    # if largest_contour is not None:
    #     image_contours = draw_contours(image_np, largest_contour)
    #     save_rgb(Path(OUTPUT_DIR) / "debug_contours.png", image_contours)

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
    main("images/1ElWxI_3_png_jpg.rf.8c659d5d4fd3587e056b5c05bf078f30.jpg")

    # Folder of images:
    # process_images("images")
