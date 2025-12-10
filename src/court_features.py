import cv2
import numpy as np


# ---------------------------------------------------------------------------------------
#   MASCHERA DEL CAMPO
# ---------------------------------------------------------------------------------------

def get_surface_mask(frame, surface):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if surface == "ERBA":
        lower = (35, 35, 40)
        upper = (85, 255, 255)

    elif surface == "CEMENTO":
        lower = (90, 40, 40)
        upper = (130, 255, 255)

    elif surface == "TERRA_BATTUTA":
        lower = (5, 70, 60)
        upper = (20, 255, 255)

    else:  # fallback
        return np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((9, 9), np.uint8), iterations=2)

    return mask


# ---------------------------------------------------------------------------------------
#   LINEE BIANCHE DEL CAMPO
# ---------------------------------------------------------------------------------------

def extract_white_lines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = (0, 0, 180)
    upper_white = (180, 40, 255)

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.GaussianBlur(mask_white, (5, 5), 0)

    return mask_white


# ---------------------------------------------------------------------------------------
#   MERGE LINEE HOUGH (UNIFICA LINEE MOLTO SIMILI)
# ---------------------------------------------------------------------------------------

def merge_similar_lines(lines, rho_thresh=20, theta_thresh=np.deg2rad(3)):
    if lines is None or len(lines) == 0:
        return np.array([])

    merged = []
    used = [False] * len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue

        rho_i, theta_i = lines[i][0]
        group = [(rho_i, theta_i)]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            rho_j, theta_j = lines[j][0]

            if abs(rho_i - rho_j) < rho_thresh and abs(theta_i - theta_j) < theta_thresh:
                group.append((rho_j, theta_j))
                used[j] = True

        g = np.array(group)
        merged.append((float(np.mean(g[:, 0])), float(np.mean(g[:, 1]))))

    return np.array(merged)


# ---------------------------------------------------------------------------------------
#   FUNZIONE FINALE RICHIESTA
# ---------------------------------------------------------------------------------------

def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    """
    Trova le linee reali del campo da tennis.
    Ritorna array di (rho, theta) già ripulite e unite.
    """

    # 1) Mask superficie
    mask_surface = get_surface_mask(image_data, surface_type)

    # 2) Linee bianche
    mask_white = extract_white_lines(image_data)

    # 3) Intersezione
    mask = cv2.bitwise_and(mask_surface, mask_white)

    # 4) Edge detection
    edges = cv2.Canny(mask, 50, 120)

    # 5) Hough
    raw_lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)

    if raw_lines is None:
        return np.array([])

    # 6) Merge linee
    merged = merge_similar_lines(raw_lines)

    return merged
