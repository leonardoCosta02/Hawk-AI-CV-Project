# src/court_features.py
import cv2 as cv
import numpy as np
from src import config


# ===========================================================
#  MERGE DI SEGMENTI COLLINEARI (orizzontali e verticali)
# ===========================================================
# ---------- REPLACEMENT for _merge_collinear_segments ----------
def _merge_collinear_segments(segments, orientation, gap_tol_px=45):
    """
    Merges collinear segments but preserves their slope (perspective).
    For each cluster we fit a line (cv.fitLine) on all endpoints and then
    compute the projected min/max points along the fitted line to build
    the merged segment.
    """
    if len(segments) == 0:
        return np.array([])

    merged = []
    used = np.zeros(len(segments), dtype=bool)

    for i in range(len(segments)):
        if used[i]:
            continue

        # Start a new cluster with segment i
        used[i] = True
        cluster_idxs = [i]

        # center of segment i
        axc = (segments[i][0] + segments[i][2]) / 2.0
        ayc = (segments[i][1] + segments[i][3]) / 2.0

        # collect segments whose center is near (within gap_tol_px) on the main axis
        for j in range(i+1, len(segments)):
            if used[j]:
                continue
            bxc = (segments[j][0] + segments[j][2]) / 2.0
            byc = (segments[j][1] + segments[j][3]) / 2.0

            if orientation == "H":
                if abs(byc - ayc) < gap_tol_px:
                    cluster_idxs.append(j)
                    used[j] = True
            else:  # "V"
                if abs(bxc - axc) < gap_tol_px:
                    cluster_idxs.append(j)
                    used[j] = True

        # gather all endpoints of the cluster
        pts = []
        for idx in cluster_idxs:
            x1, y1, x2, y2 = segments[idx]
            pts.append([x1, y1])
            pts.append([x2, y2])
        pts = np.array(pts, dtype=np.float32)

        # Fit a line using OpenCV (returns vx,vy, x0,y0)
        # handle small clusters robustly: if only two points, just use them directly
        if pts.shape[0] < 3:
            # simple bounding segment along dominant axis
            xs = pts[:,0]
            ys = pts[:,1]
            if orientation == "H":
                y_mean = np.mean(ys)
                merged.append([int(np.min(xs)), int(y_mean), int(np.max(xs)), int(y_mean)])
            else:
                x_mean = np.mean(xs)
                merged.append([int(x_mean), int(np.min(ys)), int(x_mean), int(np.max(ys))])
            continue

        vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)

        # parameterize line: P(t) = (x0, y0) + t*(vx, vy)
        # compute t for each point projection: t = ( (p - p0) · v ) / (v·v)
        v_norm2 = vx*vx + vy*vy
        ts = []
        for (px, py) in pts:
            t = ((px - x0)*vx + (py - y0)*vy) / v_norm2
            ts.append(t)
        tmin = min(ts)
        tmax = max(ts)

        p_min = (x0 + tmin*vx, y0 + tmin*vy)
        p_max = (x0 + tmax*vx, y0 + tmax*vy)
        merged.append([int(round(p_min[0])), int(round(p_min[1])),
                       int(round(p_max[0])), int(round(p_max[1]))])

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
