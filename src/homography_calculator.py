# src/homography_calculator.py
import cv2 as cv
import numpy as np
from src import config

RED = "\033[91m"
GREEN = "\033[92m"
ENDC = "\033[0m"

# ============================================================
#  Intersezione robusta tra due linee
# ============================================================
def find_intersection(s1, s2):
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(D) < 1e-4:
        return None, None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / D
    Px = x1 + t * (x2 - x1)
    Py = y1 + t * (y2 - y1)
    return Px, Py

# ============================================================
#  Calcolo Omografia M3 per area di servizio singolo
# ============================================================
def calculate_homography(all_line_segments, surface_type='CEMENTO'):
    if all_line_segments is None or len(all_line_segments) < 4:
        print(f"{RED}Errore: meno di 4 segmenti validi.{ENDC}")
        return None, None

    # -------------------------------
    # 1) Calcolo angoli e classificazione H/V
    # -------------------------------
    dx = all_line_segments[:, 2] - all_line_segments[:, 0]
    dy = all_line_segments[:, 3] - all_line_segments[:, 1]
    angles = (np.degrees(np.arctan2(dy, dx)) % 180)
    lengths = np.sqrt(dx**2 + dy**2)

    # Istogramma e picchi
    hist, bins = np.histogram(angles, bins=180, range=(0, 180))
    kernel = np.array([1.0, 1.0, 1.0])
    hist_smooth = np.convolve(hist.astype(float), kernel / kernel.sum(), mode='same')
    peak_bins = np.argsort(hist_smooth)[-2:]
    peak_angles = bins[peak_bins] + (bins[1]-bins[0])/2.0
    peak_angles = np.sort(peak_angles)

    # Se picchi non validi fallback segmento più lungo
    if len(peak_angles) < 2:
        longest_index = np.argmax(lengths)
        theta_h = angles[longest_index]
        theta_v = (theta_h + 90.0) % 180
    else:
        dist0 = [min(abs(pa), abs(pa-180)) for pa in peak_angles]
        h_idx = np.argmin(dist0)
        v_idx = 1 - h_idx
        theta_h = peak_angles[h_idx]
        theta_v = peak_angles[v_idx]

    # Classificazione H/V
    H_segments, V_segments = [], []
    for seg, ang in zip(all_line_segments, angles):
        if abs((ang - theta_h + 180) % 180) < abs((ang - theta_v + 180) % 180):
            H_segments.append(seg)
        else:
            V_segments.append(seg)

    H_segments = np.array(H_segments) if H_segments else np.empty((0, 4))
    V_segments = np.array(V_segments) if V_segments else np.empty((0, 4))

    if len(H_segments) < 2 or len(V_segments) < 2:
        print(f"{RED}Errore: servono almeno 2 H e 2 V.{ENDC}")
        return None, None

    # -------------------------------
    # 2) Selezione linee principali: base, servizio, sinistra, destra
    # -------------------------------
    h_y = (H_segments[:,1] + H_segments[:,3]) / 2.0
    v_x = (V_segments[:,0] + V_segments[:,2]) / 2.0

    # Orizzontali: base = più bassa (maggiore Y), servizio = seconda più bassa
    h_sorted = np.argsort(h_y)[::-1]
    base_line = H_segments[h_sorted[0]]
    service_line = H_segments[h_sorted[1]]

    # Verticali: sinistra = più piccola X, destra = più grande X
    v_sorted = np.argsort(v_x)
    side_left = V_segments[v_sorted[0]]
    side_right = V_segments[v_sorted[-1]]

    selected_segments = np.array([base_line, service_line, side_left, side_right], dtype=float)

    # -------------------------------
    # 3) Calcolo intersezioni
    # -------------------------------
    p1 = find_intersection(base_line, side_left)
    p2 = find_intersection(base_line, side_right)
    p3 = find_intersection(service_line, side_left)
    p4 = find_intersection(service_line, side_right)

    points_pix = np.float32([p1, p2, p3, p4])

    # Verifica punti
    for i, p in enumerate(points_pix, 1):
        if p[0] is None or p[1] is None or not np.isfinite(p[0]) or not np.isfinite(p[1]):
            print(f"{RED}Errore: intersezione p{i} non valida.{ENDC}")
            return None, None

    # -------------------------------
    # 4) Calcolo omografia
    # -------------------------------
    points_world = config.POINTS_WORLD_METERS
    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)

    if H is None:
        print(f"{RED}Errore: cv.findHomography ha fallito.{ENDC}")
        return None, None

    return H, selected_segments.astype(np.int32), points_pix.astype(np.int32)

# ============================================================
#  Utility: Pixel → World
# ============================================================
def map_pixel_to_world(H, pixel_coords):
    if H is None:
        return np.array([0.0, 0.0])
    u, v = pixel_coords
    ph = np.array([u, v, 1.0])
    wh = H @ ph
    X = wh[0]/wh[2]
    Y = wh[1]/wh[2]
    return np.array([X, Y])
