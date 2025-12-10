# src/homography_calculator.py
import cv2 as cv
import numpy as np
from . import config

RED = "\033[91m"
ENDC = "\033[0m"


# ============================================================
#  Intersezione robusta tra due linee
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
#  M3 — CALCOLO OMOGRAFIA (versione corretta)
# ============================================================
def calculate_homography(all_line_segments, surface_type='CEMENTO'):

    print("\n\n========================")
    print(f"🏁 INIZIO CALCOLO OMOGRAFIA PER {surface_type}")
    print("========================\n")

    if all_line_segments.size < 4:
        print(f"{RED}Errore: meno di 4 segmenti totali.{ENDC}")
        return None, None

    print(f"[DEBUG] Segmenti ricevuti da M1: {len(all_line_segments)}")

    # ---------------------------------------------------------
    # 1) Calcolo angoli di tutti i segmenti
    # ---------------------------------------------------------
    dx = all_line_segments[:, 2] - all_line_segments[:, 0]
    dy = all_line_segments[:, 3] - all_line_segments[:, 1]

    angles = (np.degrees(np.arctan2(dy, dx)) % 180)

    print("[DEBUG] Primi 10 angoli :", angles[:10])
    print("[DEBUG] Angolo medio    :", np.mean(angles))

    # ---------------------------------------------------------
    # 2) TROVIAMO LA DIREZIONE ORIZZONTALE DOMINANTE
    #    (la mediana è molto robusta)
    # ---------------------------------------------------------
    theta_h = np.median(angles)
    print(f"[DEBUG] Angolo orizzontale dominante (theta_h): {theta_h:.2f}°")

    # ---------------------------------------------------------
    # 3) LA VERTICALE È PERPENDICOLARE ALL’ORIZZONTALE
    # ---------------------------------------------------------
    theta_v = (theta_h + 90) % 180
    print(f"[DEBUG] Angolo verticale atteso (theta_v): {theta_v:.2f}°")

    # ---------------------------------------------------------
    # 4) CLASSIFICAZIONE SEGMENTI (basata su distanza angolare)
    # ---------------------------------------------------------
    def angular_dist(a, b):
        d = abs(a - b)
        return min(d, 180 - d)

    H_segments = []
    V_segments = []

    for seg, ang in zip(all_line_segments, angles):

        dist_h = angular_dist(ang, theta_h)
        dist_v = angular_dist(ang, theta_v)

        if dist_h < dist_v:
            H_segments.append(seg)
        else:
            V_segments.append(seg)

    H_segments = np.array(H_segments)
    V_segments = np.array(V_segments)

    print(f"[DEBUG] Segmenti orizzontali classificati : {len(H_segments)}")
    print(f"[DEBUG] Segmenti verticali classificati   : {len(V_segments)}")

    if len(H_segments) < 2 or len(V_segments) < 2:
        print(f"{RED}Errore: servono almeno 2 H e 2 V dopo la classificazione.{ENDC}")
        return None, None

    # ---------------------------------------------------------
    # 5) TEMPLATE FITTING
    # ---------------------------------------------------------
    print("\n[DEBUG] --- TEMPLATE FITTING ---")

    h_y = (H_segments[:, 1] + H_segments[:, 3]) / 2
    v_x = (V_segments[:, 0] + V_segments[:, 2]) / 2

    print("[DEBUG] Y orizzontali:", h_y)
    print("[DEBUG] X verticali   :", v_x)

    # Le due orizzontali più basse (base e servizio)
    h_sorted = np.argsort(h_y)[::-1]
    base_line = H_segments[h_sorted[0]]
    service_line = H_segments[h_sorted[1]]

    # Le due verticali più a sinistra e più a destra
    v_sorted = np.argsort(v_x)
    side_left = V_segments[v_sorted[0]]
    side_right = V_segments[v_sorted[-1]]

    print("\n[DEBUG] Linee scelte per omografia:")
    print("  Base      :", base_line)
    print("  Servizio  :", service_line)
    print("  Sinistra  :", side_left)
    print("  Destra    :", side_right)

    # ---------------------------------------------------------
    # 6) INTERSEZIONI
    # ---------------------------------------------------------
    print("\n[DEBUG] --- INTERSEZIONI ---")

    p1 = find_intersection(base_line, side_left)
    p2 = find_intersection(base_line, side_right)
    p3 = find_intersection(service_line, side_left)
    p4 = find_intersection(service_line, side_right)

    print("  p1:", p1)
    print("  p2:", p2)
    print("  p3:", p3)
    print("  p4:", p4)

    if None in [p1[0], p2[0], p3[0], p4[0]]:
        print(f"{RED}Errore: intersezioni non valide.{ENDC}")
        return None, None

    points_pix = np.float32([p1, p2, p3, p4])
    print("\n[DEBUG] Punti pixel usati:")
    print(points_pix)

    # ---------------------------------------------------------
    # 7) CALCOLO OMOGRAFIA
    # ---------------------------------------------------------
    points_world = config.POINTS_WORLD_METERS

    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)

    if H is None:
        print(f"{RED}Errore: cv.findHomography ha fallito.{ENDC}")
        return None, None

    print(f"{RED}\n=== MATRICE H CALCOLATA ==={ENDC}")
    print(H)

    return H, np.array([base_line, service_line, side_left, side_right])



# ============================================================
#  Utility — pixel → world
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
