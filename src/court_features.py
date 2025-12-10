# src/court_features.py - Versione con filtri ROI + WhiteLine + Merge

import cv2 as cv
import numpy as np
from src import config 


# -------------------------------------------------------
# 1) CREA UNA MASCHERA DEL CAMPO (ROI)
# -------------------------------------------------------
def crea_mask_campo(image, surface):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    if surface == "ERBA":
        # verde
        lower = np.array([25, 30, 30])
        upper = np.array([85, 255, 255])

    elif surface == "TERRA_BATTUTA":
        # arancione/rosso
        lower = np.array([5, 60, 40])
        upper = np.array([20, 255, 255])

    elif surface == "CEMENTO":
        # blu/ciano (indoor)
        lower = np.array([90, 40, 40])
        upper = np.array([140, 255, 255])

    mask = cv.inRange(hsv, lower, upper)

    # pulizia per evitare buchi
    kernel = np.ones((25,25), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask



# -------------------------------------------------------
# 2) CONTROLLO: LINEA DENTRO CAMPO?
# -------------------------------------------------------
def line_inside_mask(line, mask):
    x1, y1, x2, y2 = line
    h, w = mask.shape

    # clip per sicurezza
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)

    # almeno uno dei punti deve stare nella ROI campo
    return (mask[y1, x1] > 0) or (mask[y2, x2] > 0)



# -------------------------------------------------------
# 3) FILTRO "WHITE-LINE": linee bianche vere del campo
# -------------------------------------------------------
def is_white_line_segment(line, image):
    x1, y1, x2, y2 = line

    # estrai piccola regione attorno al segmento
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    seg = image[ymin:ymax, xmin:xmax]
    if seg.size < 50:
        return False

    hsv = cv.cvtColor(seg, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # Le linee del campo sono bianchissime e poco sature
    return (np.mean(v) > 170 and np.mean(s) < 60)



# -------------------------------------------------------
# 4) MERGE LINEE SIMILI
# -------------------------------------------------------
def merge_similar_lines(lines, angle_tol=5, dist_tol=25):
    if len(lines) == 0:
        return lines

    merged = []
    used = set()

    for i, l1 in enumerate(lines):
        if i in used:
            continue

        group = [l1]
        x1, y1, x2, y2 = l1
        angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        for j, l2 in enumerate(lines):
            if j == i or j in used:
                continue

            x3, y3, x4, y4 = l2
            angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))

            # linee parallele?
            if abs(angle1 - angle2) < angle_tol:
                # stessa zona?
                if abs(y1 - y3) < dist_tol or abs(x1 - x3) < dist_tol:
                    group.append(l2)
                    used.add(j)

        used.add(i)

        # creo UNA sola linea unendo tutte
        xs = [l[0] for l in group] + [l[2] for l in group]
        ys = [l[1] for l in group] + [l[3] for l in group]
        merged.append([min(xs), min(ys), max(xs), max(ys)])

    return np.array(merged)



# -------------------------------------------------------
# 5) FUNZIONE PRINCIPALE MIGLIORATA
# -------------------------------------------------------
def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    if image_data is None:
        return np.array([])

    # parametri
    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common_hough = config.HOUGH_COMMON_PARAMS

    # preprocessing
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    edges = cv.Canny(blurred, params['CANNY_LOW'], params['CANNY_HIGH'])

    raw = cv.HoughLinesP(
        edges,
        rho=common_hough['RHO'],
        theta=common_hough['THETA'],
        threshold=params['HOUGH_THRESHOLD'],
        minLineLength=common_hough['MIN_LENGTH'],
        maxLineGap=common_hough['MAX_GAP']
    )

    if raw is None:
        return np.array([])

    lines = raw.reshape(-1, 4)

    # -----------------------------
    # A) FILTRO GEOMETRICO
    # -----------------------------
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    angles = np.abs(np.degrees(np.arctan2(dy, dx)) % 180)

    min_len = common_hough['MIN_LENGTH']
    is_len_ok = lengths >= min_len

    ANG_T = common_hough['ANGLE_TOLERANCE_DEG']
    is_ang_ok = (angles < ANG_T) | (angles > 180 - ANG_T) | \
                ((angles > 90 - ANG_T) & (angles < 90 + ANG_T))

    lines = lines[is_len_ok & is_ang_ok]

    # -----------------------------
    # B) FILTRO ROI (campo)
    # -----------------------------
    mask_campo = crea_mask_campo(image_data, surface_type)
    lines = np.array([l for l in lines if line_inside_mask(l, mask_campo)])

    # -----------------------------
    # C) FILTRO COLORE (solo linee bianche)
    # -----------------------------
    lines = np.array([l for l in lines if is_white_line_segment(l, image_data)])

    # -----------------------------
    # D) MERGE LINEE
    # -----------------------------
    lines = merge_similar_lines(lines)

    return lines
