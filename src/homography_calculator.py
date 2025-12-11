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

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(D) < 1e-4:
        return None, None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / D

    Px = x1 + t * (x2 - x1)
    Py = y1 + t * (y2 - y1)

    return Px, Py


# ============================================================
#  Helper: distanza angolare in spazio modulo 180°
# ============================================================
def angular_dist(a, b):
    d = abs(a - b)
    return min(d, 180.0 - d)


# ============================================================
#        M2/M3 — CALCOLO OMOGRAFIA (ANGLE CLUSTERING FIX)
# ============================================================
def calculate_homography(all_line_segments, surface_type='CEMENTO'):

    print("\n\n========================")
    print(f"🏁 INIZIO CALCOLO OMOGRAFIA PER {surface_type}")
    print("========================\n")

    if all_line_segments is None or all_line_segments.size < 4:
        print(f"{RED}Errore: meno di 4 segmenti totali o segmenti non validi.{ENDC}")
        return None, None

    print(f"[DEBUG] Segmenti ricevuti da M1: {len(all_line_segments)}")

    # ---------------------------------------------------------
    # 1) Calcolo angoli segmenti
    # ---------------------------------------------------------
    dx = all_line_segments[:, 2] - all_line_segments[:, 0]
    dy = all_line_segments[:, 3] - all_line_segments[:, 1]
    angles = (np.degrees(np.arctan2(dy, dx)) % 180.0)

    # ---------------------------------------------------------
    # 2) TROVA LE DUE DIREZIONI DOMINANTI (ISTOGRAMMA + SMOOTH)
    # ---------------------------------------------------------
    hist_bins = 180
    hist, bin_edges = np.histogram(angles, bins=hist_bins, range=(0.0, 180.0))

    # smoothing semplice (3-sample moving average) per ridurre rumore
    kernel = np.array([1.0, 1.0, 1.0])
    hist_smooth = np.convolve(hist.astype(float), kernel / kernel.sum(), mode='same')

    if np.all(hist_smooth == 0):
        # fallback: usa la mediana se non ci sono bin efficaci
        theta_h = np.median(angles)
        theta_v = (theta_h + 90.0) % 180.0
        print("[WARN] Istogramma vuoto, uso fallback median-based angles.")
    else:
        # trova i due picchi principali (indici dei bin con maggior conteggio)
        peak_bins = np.argsort(hist_smooth)[-2:]
        peak_angles = bin_edges[peak_bins] + (bin_edges[1] - bin_edges[0]) / 2.0
        peak_angles = np.sort(peak_angles)  # ordine crescente

        # Se i due peak sono troppo vicini (ad es. entrambi dovuti a rumore),
        # fallback alla mediana
        if angular_dist(peak_angles[0], peak_angles[1]) < 8.0:
            theta_h = np.median(angles)
            theta_v = (theta_h + 90.0) % 180.0
            print(f"[WARN] I due picchi molto vicini ({peak_angles}), uso fallback median-based.")
        else:
            # Decidere quale dei due picchi è "orizzontale" (vicino 0/180)
            dist0 = [min(angular_dist(pa, 0.0), angular_dist(pa, 180.0)) for pa in peak_angles]
            if dist0[0] <= dist0[1]:
                theta_h = peak_angles[0]
                theta_v = peak_angles[1]
            else:
                theta_h = peak_angles[1]
                theta_v = peak_angles[0]

    print(f"[DEBUG] Peak/Chosen angles -> theta_h: {theta_h:.2f}°, theta_v: {theta_v:.2f}°")

    # ---------------------------------------------------------
    # 3) Classificazione segmenti in H e V
    # ---------------------------------------------------------
    H_segments = []
    V_segments = []

    for seg, ang in zip(all_line_segments, angles):
        if angular_dist(ang, theta_h) <= angular_dist(ang, theta_v):
            H_segments.append(seg)
        else:
            V_segments.append(seg)

    H_segments = np.array(H_segments) if len(H_segments) else np.empty((0, 4), dtype=float)
    V_segments = np.array(V_segments) if len(V_segments) else np.empty((0, 4), dtype=float)

    print(f"[DEBUG] Segmenti classificati H: {len(H_segments)}")
    print(f"[DEBUG] Segmenti classificati V: {len(V_segments)}")

    if len(H_segments) < 2 or len(V_segments) < 2:
        print(f"{RED}Errore: servono almeno 2 H e 2 V dopo la classificazione.{ENDC}")
        return None, None

    # ---------------------------------------------------------
    # 4) SELEZIONE DELLE LINEE (TEMPLATE M3)
    # ---------------------------------------------------------
    print("\n[DEBUG] --- TEMPLATE FITTING ---")

    h_y = (H_segments[:, 1] + H_segments[:, 3]) / 2.0
    v_x = (V_segments[:, 0] + V_segments[:, 2]) / 2.0

    # Orizzontali: ordina per Y discendente (basso → alto)
    h_sorted = np.argsort(h_y)[::-1]
    base_line = H_segments[h_sorted[0]]
    service_line = H_segments[h_sorted[1]]

    # Verticali: ordina per X ascendente (sx → dx)
    v_sorted = np.argsort(v_x)
    side_left = V_segments[v_sorted[0]]
    side_right = V_segments[v_sorted[-1]]

    print("\n[DEBUG] Linee selezionate:")
    print("  Base      :", base_line)
    print("  Servizio  :", service_line)
    print("  Sinistra  :", side_left)
    print("  Destra    :", side_right)

    # ---------------------------------------------------------
    # 5) Calcolo intersezioni
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

    # Controllo validità: ogni punto è una coppia di float non-None e finita
    pts = [p1, p2, p3, p4]
    for i, p in enumerate(pts, start=1):
        if p is None or p[0] is None or p[1] is None or not np.isfinite(p[0]) or not np.isfinite(p[1]):
            print(f"{RED}Errore: intersezione p{i} non valida.{ENDC}")
            return None, None

    points_pix = np.float32(pts)
    print("\n[DEBUG] Punti PIXEL selezionati:")
    print(points_pix)

    # Optional: sanity check geometric (assicurarsi che i punti non siano quasi collineari)
    # Calcolo area del quadrilatero p1,p2,p4,p3 (ordine scelto coerente con POINTS_WORLD_METERS)
    def quad_area(quad):
        # shoelace formula
        x = quad[:, 0]
        y = quad[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area = quad_area(points_pix)
    if area < 1.0:
        print(f"{RED}Errore: area quadrilatero troppo piccola ({area}), punti non validi.{ENDC}")
        return None, None

    # ---------------------------------------------------------
    # 6) Omografia
    # ---------------------------------------------------------
    points_world = config.POINTS_WORLD_METERS

    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)

    if H is None:
        print(f"{RED}Errore: cv.findHomography ha fallito.{ENDC}")
        return None, None

    print(f"{RED}\n=== MATRICE H CALCOLATA ==={ENDC}")
    print(H)

    return H, np.array([base_line, service_line, side_left, side_right], dtype=float)


# ============================================================
#  Utility — pixel → world
# ============================================================
def map_pixel_to_world(H, pixel_coords):
    if H is None:
        return np.array([0.0, 0.0])

    u, v = pixel_coords
    ph = np.array([u, v, 1.0])
    wh = H @ ph
    X = wh[0] / wh[2]
    Y = wh[1] / wh[2]
    return np.array([X, Y])
