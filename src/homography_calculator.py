# src/homography_calculator.py

import cv2 as cv
import numpy as np
from src import config


# --------------------------------------------------------------------
# Funzione di utilità per calcolare intersezione tra due linee
# Ogni linea è [x1, y1, x2, y2]
# --------------------------------------------------------------------
def _intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    A = np.array([[x2-x1, x3-x4],
                  [y2-y1, y3-y4]], dtype=float)
    B = np.array([x3-x1, y3-y1], dtype=float)

    det = np.linalg.det(A)
    if abs(det) < 1e-5:
        return None  # linee parallele

    t, u = np.linalg.solve(A, B)
    px = x1 + t*(x2-x1)
    py = y1 + t*(y2-y1)
    return (px, py)


# --------------------------------------------------------------------
# Classifica le linee raggruppando in 4 categorie:
# - fondo (orizzontale alta)
# - rete (orizzontale bassa)
# - lato sinistro (verticale sinistra)
# - lato destro (verticale destra)
# --------------------------------------------------------------------
def _classify_lines(segments, h, w):
    horiz = []
    vert = []

    for s in segments:
        x1, y1, x2, y2 = s
        dx = x2 - x1
        dy = y2 - y1
        ang = abs(np.degrees(np.arctan2(dy, dx)) % 180)
        if ang < 45 or ang > 135:
            horiz.append(s)
        else:
            vert.append(s)

    if len(horiz) < 2 or len(vert) < 2:
        return None

    # ordina orizzontali per y
    horiz = sorted(horiz, key=lambda s: (s[1] + s[3]) / 2)
    bottom_line = horiz[0]
    net_line = horiz[-1]

    # ordina verticali per x
    vert = sorted(vert, key=lambda s: (s[0] + s[2]) / 2)
    left_line = vert[0]
    right_line = vert[-1]

    return {
        "bottom": bottom_line,
        "net": net_line,
        "left": left_line,
        "right": right_line
    }


# --------------------------------------------------------------------
# Ottiene i 4 angoli del rettangolo d'interesse (servizio vicino)
# --------------------------------------------------------------------
def _get_service_corners(lines):
    P_bl = _intersection(lines["bottom"], lines["left"])
    P_br = _intersection(lines["bottom"], lines["right"])
    P_tl = _intersection(lines["net"],    lines["left"])
    P_tr = _intersection(lines["net"],    lines["right"])

    if None in (P_bl, P_br, P_tl, P_tr):
        return None

    return np.float32([P_bl, P_br, P_tl, P_tr])


# --------------------------------------------------------------------
# CALCOLO OMOLOGRAFIA AUTOMATICA
# --------------------------------------------------------------------
def compute_homography(segments: np.ndarray, image_shape) -> tuple:
    """
    Restituisce:
      - H: matrice 3x3 di omografia (pixel → metri)
      - pts_src: punti sorgente in pixel
      - pts_dst: punti destinazione (mondo in metri)

    Ritorna (None, None, None) se non trovabili.
    """

    if segments is None or len(segments) < 4:
        return None, None, None

    h, w = image_shape[:2]

    # 1) Classifica linee
    lines = _classify_lines(segments, h, w)
    if lines is None:
        print("Homography: linee insufficienti.")
        return None, None, None

    # 2) Intersezioni = corner area servizio
    pts_src = _get_service_corners(lines)
    if pts_src is None:
        print("Homography: impossibile trovare gli incroci.")
        return None, None, None

    # 3) World coordinates (metri)
    pts_dst = config.POINTS_WORLD_METERS.copy()

    # 4) Homografia
    H, status = cv.findHomography(pts_src, pts_dst, cv.RANSAC, 5.0)

    if H is None:
        print("Homography: fallita.")
        return None, None, None

    return H, pts_src, pts_dst
