import cv2
import numpy as np

def run_segmentation(image_np, model):
    """Chạy model YOLO để lấy mask và bounding box"""
    results = model(image_np)
    
    if len(results) > 0 and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        
        box = results[0].boxes[0]
        conf = float(box.conf)
        
        return mask, conf
    return None, None

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

def extract_corners(mask, method="ApproxPolyDP", epsilon_factor=0.02):
    """
    Tìm 4 góc từ mask dựa trên thuật toán được chỉ định.
    """
    binary_mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

def apply_perspective(image_np, corners):
    """Căng phẳng ảnh dựa trên 4 điểm ảnh"""
    (tl, tr, br, bl) = corners
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image_np, M, (maxWidth, maxHeight))
    
    return warped