# src/homography_calculator.py
import cv2 as cv
import numpy as np
from . import config

RED = "\033[91m"
ENDC = "\033[0m"


# ============================================================
#  Intersezione tra due linee (robusta)
# ============================================================
def find_intersection(s1, s2):
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2

    D = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    if abs(D) < 1e-4:
        print(f"{RED}[DEBUG] Intersezione impossibile: linee quasi parallele{ENDC}")
        return None, None

    t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / D

    Px = x1 + t*(x2 - x1)
    Py = y1 + t*(y2 - y1)

    return Px, Py


# ============================================================
#  Semplice K-means 2D per angoli
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

        c1 = vecs[labels == 0].mean(axis=0)
        c2 = vecs[labels == 1].mean(axis=0)

    return labels, c1, c2


# ============================================================
#  M3 — CALCOLO OMOGRAFIA (con DEBUG completo)
# ============================================================
def calculate_homography(all_line_segments, surface_type='CEMENTO'):

    print("\n\n========================")
    print(f"🏁 INIZIO CALCOLO OMOGRAFIA PER {surface_type}")
    print("========================\n")

    if all_line_segments.size < 4:
        print(f"{RED}Errore: meno di 4 segmenti totali (={all_line_segments.size}).{ENDC}")
        return None, None

    # ---------------------------------------------------------
    # DEBUG INIZIALE
    # ---------------------------------------------------------
    print(f"[DEBUG] Segmenti ricevuti da M1: {len(all_line_segments)}")

    dx = all_line_segments[:,2] - all_line_segments[:,0]
    dy = all_line_segments[:,3] - all_line_segments[:,1]

    ang = np.arctan2(dy, dx)
    degrees = (np.degrees(ang) % 180)

    print("\n[DEBUG] Primi 10 angoli (deg):", degrees[:10])
    print("[DEBUG] Media angoli:", np.mean(degrees))
    print("[DEBUG] Varianza angoli:", np.var(degrees))

    # ---------------------------------------------------------
    # K-means clustering su vettori direzione
    # ---------------------------------------------------------
    vecs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    labels, c1, c2 = kmeans_2d(vecs)

    # angoli dei centroidi
    cent = np.array([
        np.degrees(np.arctan2(c1[1], c1[0])) % 180,
        np.degrees(np.arctan2(c2[1], c2[0])) % 180
    ])

    print("\n[DEBUG] Centroidi cluster (deg):", cent)

    # cluster più vicino a 0° = orizzontale
    h_idx = int(np.argmin(np.minimum(np.abs(cent), 180 - np.abs(cent))))
    v_idx = 1 - h_idx

    print(f"[DEBUG] Cluster H = {h_idx}, Cluster V = {v_idx}")

    # separa segmenti
    H_segments = all_line_segments[labels == h_idx]
    V_segments = all_line_segments[labels == v_idx]

    print(f"[DEBUG] Segmenti orizzontali trovati: {len(H_segments)}")
    print(f"[DEBUG] Segmenti verticali trovati:   {len(V_segments)}")

    if len(H_segments) < 2 or len(V_segments) < 2:
        print(f"{RED}Errore: trovate troppo poche linee H o V dopo clustering.{ENDC}")
        return None, None

    # ---------------------------------------------------------
    # TEMPLATE FITTING
    # ---------------------------------------------------------
    print("\n[DEBUG] --- TEMPLATE FITTING ---")

    h_y = (H_segments[:,1] + H_segments[:,3]) / 2
    v_x = (V_segments[:,0] + V_segments[:,2]) / 2

    print("[DEBUG] Y dei segmenti H:", h_y)
    print("[DEBUG] X dei segmenti V:", v_x)

    h_sorted = np.argsort(h_y)[::-1]
    v_sorted = np.argsort(v_x)

    base_line = H_segments[h_sorted[0]]
    service_line = H_segments[h_sorted[1]]
    side_left = V_segments[v_sorted[0]]
    side_right = V_segments[v_sorted[-1]]

    print("\n[DEBUG] Linee scelte:")
    print("  Base Line    :", base_line)
    print("  Service Line :", service_line)
    print("  Left Line    :", side_left)
    print("  Right Line   :", side_right)

    # ---------------------------------------------------------
    # INTERSEZIONI (P1 P2 P3 P4)
    # ---------------------------------------------------------
    print("\n[DEBUG] --- CALCOLO INTERSEZIONI ---")

    p1 = find_intersection(base_line, side_left)
    p2 = find_intersection(base_line, side_right)
    p3 = find_intersection(service_line, side_left)
    p4 = find_intersection(service_line, side_right)

    print("[DEBUG] p1:", p1)
    print("[DEBUG] p2:", p2)
    print("[DEBUG] p3:", p3)
    print("[DEBUG] p4:", p4)

    if None in [p1[0], p2[0], p3[0], p4[0]]:
        print(f"{RED}Errore: non è stato possibile trovare tutte le intersezioni.{ENDC}")
        return None, None

    points_pix = np.float32([p1, p2, p3, p4])
    print("\n[DEBUG] Punti pixel finali:")
    print(points_pix)

    # ---------------------------------------------------------
    # CALCOLO OMOGRAFIA
    # ---------------------------------------------------------
    print("\n[DEBUG] --- CALCOLO MATRICE H ---")

    points_world = config.POINTS_WORLD_METERS

    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)

    if H is None:
        print(f"{RED}Errore: cv.findHomography ha fallito (punti non coerenti).{ENDC}")
        return None, None

    print(f"{RED}\n=== MATRICE H CALCOLATA ==={ENDC}")
    print(H)

    return H, np.array([base_line, service_line, side_left, side_right])


# ============================================================
#  Utility — mapping pixel → world
# ============================================================
def map_pixel_to_world(H, pixel_coords):
    if H is None:
        return np.array([0, 0])

    u, v = pixel_coords
    ph = np.array([u, v, 1])
    wh = H @ ph
    X = wh[0] / wh[2]
    Y = wh[1] / wh[2]
    return np.array([X, Y])
