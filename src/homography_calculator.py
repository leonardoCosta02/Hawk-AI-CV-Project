# src/homography_calculator.py
import cv2 as cv
import numpy as np
from . import config

RED = "\033[91m"
ENDC = "\033[0m"


# ============================================================
#  Intersezione tra due linee
# ============================================================
def find_intersection(s1, s2):
    x1,y1,x2,y2 = s1
    x3,y3,x4,y4 = s2

    D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(D) < 1e-4:
        return None, None

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / D
    Px = x1 + t*(x2-x1)
    Py = y1 + t*(y2-y1)
    return Px, Py


# ============================================================
#  K-MEANS SEMPLICE (2 CLUSTER) SU VETTORI ANGOLARI
# ============================================================
def kmeans_2d(vecs, iters=20):
    c1 = vecs[0]
    c2 = vecs[len(vecs)//3]

    for _ in range(iters):
        d1 = np.linalg.norm(vecs - c1, axis=1)
        d2 = np.linalg.norm(vecs - c2, axis=1)
        labels = (d2 < d1).astype(int)

        if labels.sum() == 0 or labels.sum() == len(vecs):
            break

        c1 = vecs[labels==0].mean(axis=0)
        c2 = vecs[labels==1].mean(axis=0)

    return labels, c1, c2


# ============================================================
#  M3 — CALCOLO OMOGRAFIA
# ============================================================
def calculate_homography(all_line_segments, surface_type='CEMENTO'):

    if all_line_segments.size < 4:
        print("Segmenti insufficienti.")
        return None, None

    dx = all_line_segments[:,2] - all_line_segments[:,0]
    dy = all_line_segments[:,3] - all_line_segments[:,1]

    ang = np.arctan2(dy, dx)
    vecs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    labels, c1, c2 = kmeans_2d(vecs)

    cent = np.array([
        np.degrees(np.arctan2(c1[1], c1[0])) % 180,
        np.degrees(np.arctan2(c2[1], c2[0])) % 180
    ])

    h_idx = int(np.argmin(np.minimum(np.abs(cent), 180 - np.abs(cent))))
    v_idx = 1 - h_idx

    is_h = (labels == h_idx)
    is_v = (labels == v_idx)

    H_segments = all_line_segments[is_h]
    V_segments = all_line_segments[is_v]

    if len(H_segments) < 2 or len(V_segments) < 2:
        print(f"{RED}Errore: trovate {len(H_segments)} H e {len(V_segments)} V.{ENDC}")
        return None, None

    # -----------------------------------------------------
    #   TEMPLATE FITTING
    # -----------------------------------------------------
    h_y = (H_segments[:,1] + H_segments[:,3]) / 2
    v_x = (V_segments[:,0] + V_segments[:,2]) / 2

    h_sorted = np.argsort(h_y)[::-1]
    v_sorted = np.argsort(v_x)

    base_line = H_segments[h_sorted[0]]
    service_line = H_segments[h_sorted[1]]

    side_left = V_segments[v_sorted[0]]
    side_right = V_segments[v_sorted[-1]]

    print(f"{RED}--- LINEE CHIAVE RILEVATE ---{ENDC}")
    print(base_line, service_line, side_left, side_right)

    # -----------------------------------------------------
    #   INTERSEZIONI
    # -----------------------------------------------------
    p1 = find_intersection(base_line, side_left)
    p2 = find_intersection(base_line, side_right)
    p3 = find_intersection(service_line, side_left)
    p4 = find_intersection(service_line, side_right)

    if None in [p1[0], p2[0], p3[0], p4[0]]:
        print(f"{RED}Errore intersezioni.{ENDC}")
        return None, None

    points_pix = np.float32([p1, p2, p3, p4])
    print(f"{RED}Punti Pixel:{ENDC}\n{points_pix}")

    points_world = config.POINTS_WORLD_METERS

    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)
    if H is None:
        print(f"{RED}Errore homografia.{ENDC}")
        return None, None

    print(f"{RED}Matrice H calcolata:{ENDC}")
    print(H)
    return H, np.array([base_line, service_line, side_left, side_right])


# ============================================================
#  Utility — mapping pixel → world
# ============================================================
def map_pixel_to_world(H, pixel_coords):
    if H is None:
        return np.array([0,0])

    u,v = pixel_coords
    ph = np.array([u,v,1])
    wh = H @ ph
    X = wh[0] / wh[2]
    Y = wh[1] / wh[2]
    return np.array([X,Y])
