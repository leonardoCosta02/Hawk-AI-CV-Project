# src/court_features.py
import cv2 as cv
import numpy as np
from src import config


# ===========================================================
#  MERGE DI SEGMENTI COLLINEARI (orizzontali e verticali)
# ===========================================================
def _merge_collinear_segments(segments, orientation, gap_tol_px=45):
    if len(segments) == 0:
        return np.array([])

    merged = []
    used = np.zeros(len(segments), dtype=bool)

    for i in range(len(segments)):
        if used[i]: 
            continue

        a = segments[i]
        used[i] = True

        xs = [a[0], a[2]]
        ys = [a[1], a[3]]

        axc = (a[0] + a[2]) / 2
        ayc = (a[1] + a[3]) / 2

        for j in range(i+1, len(segments)):
            if used[j]:
                continue

            b = segments[j]
            bxc = (b[0] + b[2]) / 2
            byc = (b[1] + b[3]) / 2

            if orientation == "H":
                if abs(byc - ayc) < gap_tol_px:
                    xs += [b[0], b[2]]
                    ys += [b[1], b[3]]
                    used[j] = True

            else:  # verticale
                if abs(bxc - axc) < gap_tol_px:
                    xs += [b[0], b[2]]
                    ys += [b[1], b[3]]
                    used[j] = True

        # costruttore finale
        if orientation == "H":
            merged.append([min(xs), ayc, max(xs), ayc])
        else:
            merged.append([axc, min(ys), axc, max(ys)])

    return np.array(merged, dtype=int)



# ===========================================================
#      M1 — TROVA LINEE (con merge collineare finale)
# ===========================================================
def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:

    if image_data is None:
        return np.array([])

    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common = config.HOUGH_COMMON_PARAMS

    h, w, _ = image_data.shape

    # ---------------------------
    # Preprocessing
    # ---------------------------
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5,5), 1.0)

    edges = cv.Canny(blurred, params['CANNY_LOW'], params['CANNY_HIGH'])

    linesP = cv.HoughLinesP(
        edges,
        rho=common['RHO'],
        theta=common['THETA'],
        threshold=params['HOUGH_THRESHOLD'],
        minLineLength=common['MIN_LENGTH'],
        maxLineGap=common['MAX_GAP']
    )

    if linesP is None:
        return np.array([])

    segments = linesP.reshape(-1,4)

    # -----------------------------------------------------
    #   FILTRI DI CENTRALITÀ (Y → elimina pubblico, X → elimina bordo campo)
    # -----------------------------------------------------
    y_center = (segments[:,1] + segments[:,3]) / 2
    x_center = (segments[:,0] + segments[:,2]) / 2

    valid_y = (y_center > h*0.30) & (y_center < h*0.84)
    valid_x = (x_center > w*0.20) & (x_center < w*0.80)

    segments = segments[valid_y & valid_x]

    if len(segments) == 0:
        return np.array([])

    # -----------------------------------------------------
    #   SEPARA SEGMENTI IN H e V (Semplice)
    # -----------------------------------------------------
    dx = segments[:,2] - segments[:,0]
    dy = segments[:,3] - segments[:,1]
    angles = np.abs(np.degrees(np.arctan2(dy,dx)) % 180)

    is_h = (angles < 45) | (angles > 135)
    is_v = ~is_h

    horiz = segments[is_h]
    vert = segments[is_v]
    print("=== DEBUG: M1 — FILTRI CENTRALITÀ ===")
    print(f"Dopo filtro Y/X: {len(segments)} segmenti")
    print("=== DEBUG: M1 — ANGOLI ===")
    print(f"Orizzontali: {len(horiz)}  Verticali: {len(vert)}")


    # -----------------------------------------------------
    #   MERGE COLLINEARE
    # -----------------------------------------------------
    merged_h = _merge_collinear_segments(horiz, "H")
    merged_v = _merge_collinear_segments(vert, "V")
    print("=== DEBUG: M1 — MERGE ===")
    print(f"Dopo merge H: {len(merged_h)}  Dopo merge V: {len(merged_v)}")
    if len(merged_h) == 0 and len(merged_v) == 0:
        return np.array([])

    return np.vstack([merged_h, merged_v]) if len(merged_h) and len(merged_v) else (merged_h if len(merged_h) else merged_v)
