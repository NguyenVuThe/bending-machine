import cv2

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