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

    # Se D è quasi zero (linee parallele o quasi), non interseca
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
#        M2/M3 — CALCOLO OMOGRAFIA (FIX DI ANCORAGGIO ANGOLARE)
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
    # 1) Calcolo angoli e lunghezza dei segmenti
    # ---------------------------------------------------------
    dx = all_line_segments[:, 2] - all_line_segments[:, 0]
    dy = all_line_segments[:, 3] - all_line_segments[:, 1]
    angles = (np.degrees(np.arctan2(dy, dx)) % 180.0)
    lengths = np.sqrt(dx**2 + dy**2) # Aggiunto calcolo lunghezza

    # ---------------------------------------------------------
    # 2) TROVA LE DUE DIREZIONI DOMINANTI
    # ---------------------------------------------------------
    hist_bins = 180
    hist, bin_edges = np.histogram(angles, bins=hist_bins, range=(0.0, 180.0))
    kernel = np.array([1.0, 1.0, 1.0])
    hist_smooth = np.convolve(hist.astype(float), kernel / kernel.sum(), mode='same')
    
    # Fallback/Debug setup
    peak_angles = []
    if not np.all(hist_smooth == 0):
        peak_bins = np.argsort(hist_smooth)[-2:]
        peak_angles = bin_edges[peak_bins] + (bin_edges[1] - bin_edges[0]) / 2.0
        peak_angles = np.sort(peak_angles)

    # ---------------------------------------------------------
    # 3) CLASSIFICAZIONE ANGOLARE ROBUSATA (FIX)
    # ---------------------------------------------------------
    
    theta_h = 0.0
    theta_v = 90.0
    
    if len(peak_angles) >= 2 and angular_dist(peak_angles[0], peak_angles[1]) >= 8.0:
        # Se l'istogramma è valido, usiamo i picchi.
        # Assegna theta_h al picco più vicino a 0/180
        dist0 = [min(angular_dist(pa, 0.0), angular_dist(pa, 180.0)) for pa in peak_angles]
        h_idx = np.argmin(dist0)
        v_idx = 1 - h_idx
        
        theta_h = peak_angles[h_idx]
        theta_v = peak_angles[v_idx]
        print(f"[DEBUG] Peak/Chosen angles -> theta_h: {theta_h:.2f}°, theta_v: {theta_v:.2f}° (Istogramma)")

        # ** FIX ROBUSTO **: Se theta_h è troppo lontano da 0/180, l'istogramma è fallito.
        # Forza theta_h all'angolo del segmento PIÙ LUNGO.
        if angular_dist(theta_h, 0.0) > 15.0:
            longest_index = np.argmax(lengths)
            theta_h_fallback = angles[longest_index]
            theta_v_fallback = (theta_h_fallback + 90.0) % 180.0
            
            print(f"{RED}[WARN] Istogramma fallito ({theta_h:.2f}°). Forzo theta_h all'angolo del segmento più lungo: {theta_h_fallback:.2f}°{ENDC}")
            theta_h = theta_h_fallback
            theta_v = theta_v_fallback
    else:
        # Fallback: se istogramma non ha picchi validi, usa l'angolo del segmento più lungo.
        longest_index = np.argmax(lengths)
        theta_h = angles[longest_index]
        theta_v = (theta_h + 90.0) % 180.0
        print(f"{RED}[WARN] Istogramma vuoto/fallito. Uso angolo del segmento più lungo come theta_h: {theta_h:.2f}°{ENDC}")


    print(f"[DEBUG] FINAL ANGLES -> theta_h: {theta_h:.2f}°, theta_v: {theta_v:.2f}°")

    H_segments = []
    V_segments = []

    # Classificazione (corretta) in base alla vicinanza all'angolo dominante
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
        print(f"{RED}Errore: servono almeno 2 H e 2 V dopo la classificazione. (H={len(H_segments)}, V={len(V_segments)}){ENDC}")
        return None, None

    # ---------------------------------------------------------
    # 4) SELEZIONE DELLE LINEE (EURISTICA FINALE)
    # ---------------------------------------------------------
    print("\n[DEBUG] --- TEMPLATE FITTING ---")

    h_y = (H_segments[:, 1] + H_segments[:, 3]) / 2.0
    v_x = (V_segments[:, 0] + V_segments[:, 2]) / 2.0

    # Orizzontali: ordina per Y discendente (basso → alto)
    h_sorted = np.argsort(h_y)[::-1]
    base_line = H_segments[h_sorted[0]]
    service_line = H_segments[h_sorted[1]] # Seconda più bassa
    
    # Verticali: ordina per X ascendente (sx → dx)
    v_sorted = np.argsort(v_x)
    side_left = V_segments[v_sorted[0]]
    side_right = V_segments[v_sorted[-1]] # L'ultima è la più a destra

    print("\n[DEBUG] Linee scelte per omografia:")
    print("  Base      :", base_line)
    print("  Servizio  :", service_line)
    print("  Sinistra  :", side_left)
    print("  Destra    :", side_right)

    selected_segments = np.array([base_line, service_line, side_left, side_right], dtype=float)

    # ---------------------------------------------------------
    # 5) Calcolo intersezioni e Omografia
    # ---------------------------------------------------------
    
    # ... (Il resto delle fasi C, D e E rimane invariato)
    print("\n[DEBUG] --- INTERSEZIONI ---")

    p1 = find_intersection(base_line, side_left)
    p2 = find_intersection(base_line, side_right)
    p3 = find_intersection(service_line, side_left)
    p4 = find_intersection(service_line, side_right)

    print("  p1:", p1)
    print("  p2:", p2)
    print("  p3:", p3)
    print("  p4:", p4)

    pts = [p1, p2, p3, p4]
    for i, p in enumerate(pts, start=1):
        if p is None or p[0] is None or p[1] is None or not np.isfinite(p[0]) or not np.isfinite(p[1]):
            print(f"{RED}Errore: intersezione p{i} non valida (Non finita o None).{ENDC}")
            return None, None

    points_pix = np.float32(pts)

    # Sanity check: L'area del quadrilatero deve essere ragionevole
    def quad_area(quad):
        x = quad[:, 0]
        y = quad[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area = quad_area(points_pix[[0, 1, 3, 2]]) # Ordine per l'area
    if area < 500.0: # Aumentato leggermente la soglia minima
        print(f"{RED}Errore: area quadrilatero troppo piccola ({area:.2f}), punti non validi.{ENDC}")
        return None, None

    # 6) Omografia
    points_world = config.POINTS_WORLD_METERS[:4] # Prendiamo solo i primi 4 punti per il box di servizio

    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)

    if H is None:
        print(f"{RED}Errore: cv.findHomography ha fallito.{ENDC}")
        return None, None

    print(f"{RED}\n=== MATRICE H CALCOLATA ==={ENDC}")
    print(H)

    # Restituiamo i segmenti convertiti in int32 per la visualizzazione
    return H, selected_segments.astype(np.int32)


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
