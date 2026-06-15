import csv
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

from pipeline import extract_corners, extract_largest_contour, run_segmentation

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


def extract_edge_safe(boundary, start_idx, end_idx):
    """
    Trích xuất cạnh an toàn từ contour khép kín.
    Đảm bảo luôn lấy đoạn ngắn nhất giữa 2 index (tránh lỗi wrap-around của mảng).
    """
    if start_idx <= end_idx:
        path_fwd = boundary[start_idx:end_idx+1]
    else:
        path_fwd = np.concatenate([boundary[start_idx:], boundary[:end_idx+1]])
        
    if start_idx >= end_idx:
        path_bwd = boundary[end_idx:start_idx+1][::-1]
    else:
        path_bwd = np.concatenate([boundary[end_idx:], boundary[:start_idx+1]])[::-1]
        
    return path_fwd if len(path_fwd) <= len(path_bwd) else path_bwd


def draw_corners(image_np, corners):
    image_corners = image_np.copy()
    labels = ["TL", "TR", "BR", "BL"]

    for point, label in zip(corners, labels):
        x, y = map(int, point)
        cv2.circle(image_corners, (x, y), 15, (0, 255, 0), -1)
        cv2.putText(
            image_corners, label, (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
    return image_corners


def draw_contours(image_np, largest_contour):
    image_contours = image_np.copy()
    cv2.drawContours(image_contours, [largest_contour], -1, (0, 255, 0), 3)
    return image_contours


def draw_edges(image_np, left_edge, bottom_edge, right_edge, top_edge):
    image_edges = image_np.copy()

    for p in left_edge:
        cv2.circle(image_edges, tuple(p.astype(int)), 2, (255,0,0), -1)
    for p in bottom_edge:
        cv2.circle(image_edges, tuple(p.astype(int)), 2, (0,255,0), -1)
    for p in right_edge:
        cv2.circle(image_edges, tuple(p.astype(int)), 2, (0,0,255), -1)
    for p in top_edge:
        cv2.circle(image_edges, tuple(p.astype(int)), 2, (255,255,0), -1)

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
    
    # Xử lý trường hợp mảng bị rỗng hoặc lỗi
    if len(points) < 2:
        return np.zeros((n_points, 2))

    dist = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    dist = np.insert(np.cumsum(dist), 0, 0)

    # Đảm bảo mảng distance tăng ngặt để tránh lỗi nội suy
    dist, indices = np.unique(dist, return_index=True)
    points = points[indices]
    
    if len(points) < 2:
        return np.zeros((n_points, 2))

    fx = interp1d(dist, points[:, 0], kind='linear')
    fy = interp1d(dist, points[:, 1], kind='linear')

    new_dist = np.linspace(0, dist[-1], n_points)
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

    # Step 1: Segmentation
    mask, conf = run_segmentation(image_np, model)
    if mask is None:
        raise RuntimeError("Model cannot detect a mask on this image.")

    # Step 2: Corner extraction
    corners = extract_corners(mask, method=CORNER_METHOD, epsilon_factor=EPSILON_FACTOR)
    print(f"Extracted corners: {corners}")
    if corners is None:
        raise RuntimeError(
            f"{CORNER_METHOD} cannot find exactly 4 corners. "
            f"Try changing EPSILON_FACTOR, current value: {EPSILON_FACTOR}"
        )

    image_corners = draw_corners(image_np, corners)
    TL, TR, BR, BL = [np.array(pt) for pt in corners] # Chuyển về numpy array để tính toán an toàn
    
    largest_contour = extract_largest_contour(mask)
    boundary = largest_contour.squeeze()

    idx_tl = nearest_idx(boundary, TL)
    idx_tr = nearest_idx(boundary, TR)
    idx_br = nearest_idx(boundary, BR)
    idx_bl = nearest_idx(boundary, BL)

    print("TL:", idx_tl, "TR:", idx_tr, "BR:", idx_br, "BL:", idx_bl)

    # Trích xuất 4 cạnh một cách an toàn bằng hàm trợ giúp mới
    top_edge = extract_edge_safe(boundary, idx_tl, idx_tr)
    bottom_edge = extract_edge_safe(boundary, idx_bl, idx_br)
    left_edge = extract_edge_safe(boundary, idx_tl, idx_bl)
    right_edge = extract_edge_safe(boundary, idx_tr, idx_br)

    print("Length of top edge:", len(top_edge))
    print("Length of bottom edge:", len(bottom_edge))
    print("Length of left edge:", len(left_edge))
    print("Length of right edge:", len(right_edge))

    cols = 100
    rows = 60

    # Nội suy đều các điểm trên 4 cạnh
    top = resample_curve(top_edge, cols)
    bottom = resample_curve(bottom_edge, cols)
    left = resample_curve(left_edge, rows)
    right = resample_curve(right_edge, rows)

    # Tạo Lưới Coons Patch (Source Mesh)
    mesh = []
    for c in range(cols):
        u = c / (cols - 1)
        column = []
        for r in range(rows):
            v = r / (rows - 1)

            # Nội suy từ viền
            C = (1 - v) * top[c] + v * bottom[c]
            D = (1 - u) * left[r] + u * right[r]

            # Hiệu chỉnh 4 góc (Bilinear corner correction)
            B = (1 - u) * (1 - v) * TL + u * (1 - v) * TR + (1 - u) * v * BL + u * v * BR

            # Điểm thực tế của lưới (Coons Patch)
            pt = C + D - B
            column.append(pt)
        mesh.append(column)

    mesh = np.array(mesh)  # Hình dạng (cols, rows, 2)

    image_mesh = draw_mesh(image_np, mesh)

    # Tính toán kích thước output động theo tỷ lệ thực tế
    W = int(max(np.linalg.norm(TR - TL), np.linalg.norm(BR - BL)))
    H = int(max(np.linalg.norm(BL - TL), np.linalg.norm(BR - TR)))
    print(f"Dynamic Output Size: {W}x{H}")

    # Tạo Lưới phẳng (Target Mesh)
    target_mesh = []
    for c in range(cols):
        x = c * W / (cols - 1)
        column = []
        for r in range(rows):
            y = r * H / (rows - 1)
            column.append([x, y])
        target_mesh.append(column)

    target_mesh = np.array(target_mesh)  # Hình dạng (cols, rows, 2)

    # Định dạng điểm cho thuật toán Piecewise Affine
    src_pts = mesh.reshape(-1, 2)
    dst_pts = target_mesh.reshape(-1, 2)

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_pts, src_pts)

    # Áp dụng Dewarping lên ảnh
    warped = warp(image_np, tform, output_shape=(H, W))
    warped_img = (warped * 255).astype(np.uint8)

    return mask, image_corners, image_mesh, warped_img, conf


def main(image_path):
    output_dir = Path(OUTPUT_DIR)
    model = YOLO(MODEL_PATH)
    mask, image_corners, image_mesh, warped_img, conf = run_one_image(image_path, model)

    save_mask(output_dir / "01_mask.png", mask)
    save_rgb(output_dir / "02_corners.png", image_corners)
    save_rgb(output_dir / "03_mesh.png", image_mesh)
    save_rgb(output_dir / "04_output.png", warped_img)

    print(f"Done: {image_path}")
    print(f"Confidence: {conf:.2f}")
    print(f"Saved results to: {output_dir}")


def process_images(input_folder_path):
    input_folder_path = Path(input_folder_path)
    all_steps_dir = Path(OUTPUT_DIR) / "all_steps"
    status_csv_path = all_steps_dir / "status.csv"

    if not input_folder_path.is_dir():
        raise FileNotFoundError(f"Cannot find input folder: {input_folder_path}")

    image_paths = find_images(input_folder_path)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_folder_path}")

    model = YOLO(MODEL_PATH)
    success_count = 0
    rows = []

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(input_folder_path)
        output_name = relative_path.with_suffix(".png")

        print(f"[{index}/{len(image_paths)}] {relative_path}")

        try:
            mask, image_corners, image_mesh, warped_img, conf = run_one_image(image_path, model)

            save_mask(all_steps_dir / "masks" / output_name, mask)
            save_rgb(all_steps_dir / "corners" / output_name, image_corners)
            save_rgb(all_steps_dir / "meshes" / output_name, image_mesh)
            save_rgb(all_steps_dir / "outputs" / output_name, warped_img)

            success_count += 1
            rows.append(
                {
                    "image": str(relative_path),
                    "status": "processed",
                    "confidence": f"{conf:.4f}",
                }
            )
            print(f"  OK, confidence: {conf:.2f}")
        except Exception as exc:
            rows.append(
                {
                    "image": str(relative_path),
                    "status": "not_processed",
                    "confidence": "",
                }
            )
            print(f"  Failed: {exc}")

    status_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with status_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "status", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: {success_count}/{len(image_paths)} images processed")
    print(f"All steps saved to: {all_steps_dir}")
    print(f"Status CSV saved to: {status_csv_path}")


if __name__ == "__main__":
    # Single image:
    # main("images/0KU0NH_4_png_jpg.rf.b677abe04219a4536df3570380d429d5.jpg")

    # Folder of images:
    process_images("images")
