# src/court_features.py
import cv2 as cv
import numpy as np
from src import config


# ---------------------------------------------------------
# 1) ROI MASK basata sulla proiezione verticale dei bordi
# ---------------------------------------------------------
def build_roi_mask(gray):
    h, w = gray.shape
    sobel_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    proj = np.sum(np.abs(sobel_y), axis=1)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)

    thresh = proj.mean() * 1.5
    rows = np.where(proj > thresh)[0]

    mask = np.zeros_like(gray, dtype=np.uint8)
    if len(rows) > 0:
        top = max(0, rows[0] - 20)
        bottom = min(h, rows[-1] + 20)
        mask[top:bottom, :] = 255

    return mask


# ---------------------------------------------------------
# 2) Filtro colore bianco
# ---------------------------------------------------------
def extract_white_pixels(image_bgr):
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)
    return mask


# ---------------------------------------------------------
# 3) Merge dei segmenti collineari
# ---------------------------------------------------------
def merge_collinear_segments(segments, angle_tol=5, dist_thresh=20):
    if len(segments) == 0:
        return segments

    merged = []
    used = np.zeros(len(segments), dtype=bool)

    def seg_angle(seg):
        x1, y1, x2, y2 = seg
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        return ang

    def seg_dist(a, b):
        ax = (a[0] + a[2]) / 2
        ay = (a[1] + a[3]) / 2
        bx = (b[0] + b[2]) / 2
        by = (b[1] + b[3]) / 2
        return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    for i in range(len(segments)):
        if used[i]:
            continue

        group = [segments[i]]
        used[i] = True

        ai = seg_angle(segments[i])

        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            aj = seg_angle(segments[j])

            if abs(ai - aj) < angle_tol or abs(abs(ai - aj) - 180) < angle_tol:
                if seg_dist(segments[i], segments[j]) < dist_thresh:
                    used[j] = True
                    group.append(segments[j])

        xs, ys, xe, ye = [], [], [], []
        for (x1, y1, x2, y2) in group:
            xs += [x1, x2]
            ys += [y1, y2]
            xe = xs
            ye = ys

        merged.append([min(xs), min(ys), max(xs), max(ys)])

    return np.array(merged, dtype=np.int32)


# ---------------------------------------------------------
# 4) Funzione principale — RESTITUISCE SEGMENTI (x1,y1,x2,y2)
# ---------------------------------------------------------
def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    if image_data is None:
        return np.array([])

    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common = config.HOUGH_COMMON_PARAMS

    # --- Preprocessing ---
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1.0)

    # --- ROI ---
    roi_mask = build_roi_mask(gray)
    masked_gray = cv.bitwise_and(gray, gray, mask=roi_mask)

    # --- White mask ---
    white_mask = extract_white_pixels(image_data)
    masked_gray = cv.bitwise_and(masked_gray, masked_gray, mask=white_mask)

    # --- Canny ---
    edges = cv.Canny(masked_gray, params['CANNY_LOW'], params['CANNY_HIGH'])

    # --- Hough Probabilistico ---
    linesP = cv.HoughLinesP(
        edges,
        rho=common['RHO'],
        theta=common['THETA'],
        threshold=params['HOUGH_THRESHOLD'],
        minLineLength=common['MIN_LENGTH'],
        maxLineGap=common['MAX_GAP']
    )

    if linesP is None:
        return np.array([])

    segments = linesP.reshape(-1, 4)

    # -----------------------------------------------------
    #  Filtri su angolo (verticali/orizzontali)
    # -----------------------------------------------------
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    angles = np.abs(np.degrees(np.arctan2(dy, dx)) % 180)

    tol = common['ANGLE_TOLERANCE_DEG']
    valid_angle = (
        (angles < tol) |                      # orizzontali
        (angles > 180 - tol) |                # orizzontali 180°
        ((angles > 90 - tol) & (angles < 90 + tol))  # verticali
    )

    segments = segments[valid_angle]

    # -----------------------------------------------------
    #  Merge finale
    # -----------------------------------------------------
    merged = merge_collinear_segments(segments)

    return merged
