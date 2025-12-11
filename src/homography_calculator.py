import numpy as np
def calculate_homography(all_line_segments, surface_type='CEMENTO'):
    RED = "\033[91m"
    if all_line_segments is None or len(all_line_segments) < 4:
        print(f"{RED}Errore: meno di 4 segmenti validi.{ENDC}")
        return None, None, None

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
        return None, None, None  # <- restituiamo 3 valori anche in errore

    h_y = (H_segments[:,1] + H_segments[:,3]) / 2.0
    v_x = (V_segments[:,0] + V_segments[:,2]) / 2.0

    h_sorted = np.argsort(h_y)[::-1]
    base_line = H_segments[h_sorted[0]]
    service_line = H_segments[h_sorted[1]]

    v_sorted = np.argsort(v_x)
    side_left = V_segments[v_sorted[0]]
    side_right = V_segments[v_sorted[-1]]

    selected_segments = np.array([base_line, service_line, side_left, side_right], dtype=float)

    p1 = find_intersection(base_line, side_left)
    p2 = find_intersection(base_line, side_right)
    p3 = find_intersection(service_line, side_left)
    p4 = find_intersection(service_line, side_right)

    points_pix = np.float32([p1, p2, p3, p4])

    # Verifica punti
    for i, p in enumerate(points_pix, 1):
        if p[0] is None or p[1] is None or not np.isfinite(p[0]) or not np.isfinite(p[1]):
            print(f"{RED}Errore: intersezione p{i} non valida.{ENDC}")
            return None, None, None  # <- restituiamo 3 valori anche in errore

    points_world = config.POINTS_WORLD_METERS
    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)

    if H is None:
        print(f"{RED}Errore: cv.findHomography ha fallito.{ENDC}")
        return None, None, None

    return H, selected_segments.astype(np.int32), points_pix.astype(np.int32)
