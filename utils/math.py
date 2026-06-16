import numpy as np
import cv2
from scipy.interpolate import interp1d

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_corners(mask, method="ApproxPolyDP", epsilon_factor=0.05):
    """
    Tìm 4 góc từ mask dựa trên thuật toán được chỉ định.
    """
    binary_mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
        
    c = max(contours, key=cv2.contourArea)
    
    # Phương pháp 1: Xấp xỉ đa giác (ApproxPolyDP)
    if method == "ApproxPolyDP":
        epsilon = epsilon_factor * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            return order_points(pts)
        else:
            return None # Trả về None nếu không tìm được đúng 4 góc
            
    # Phương pháp 2: Hình chữ nhật bao quanh nhỏ nhất (MinAreaRect)
    elif method == "MinAreaRect":
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        pts = np.intp(box) # Ép kiểu an toàn cho numpy mới
        return order_points(pts)

    return None

def extract_largest_contour(mask):
    binary_mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(
        largest_contour
    )
    print(area)
    print(largest_contour.shape)
    return largest_contour

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


