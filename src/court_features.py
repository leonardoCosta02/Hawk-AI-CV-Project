# src/court_features.py
"""
Feature Extraction Module for Tennis Court Analysis.

This module implements the core Computer Vision pipeline to extract 
structural lines from raw video frames. It addresses two main challenges:
1. Signal Processing: Detecting edges in noisy environments (Clay/Grass) using Canny/Hough.
2. Geometric Reconstruction: Merging fragmented line segments caused by occlusion or 
   lighting variations into coherent structural vectors.
"""

import cv2 as cv
import numpy as np
from src import config

# =============================================================================
# SEGMENT MERGING LOGIC (Collinear Clustering)
# =============================================================================
def _merge_collinear_segments(segments, orientation, gap_tol_px=45):
    """
    Consolidates fragmented Hough segments into single coherent lines.

    The Standard Hough Transform often breaks a single physical line (like a service line)
    into multiple segments due to noise, occlusions (players), or worn paint.
    This function clusters segments that are spatially collinear and geometrically 
    reconstructs the 'true' line using a parametric fit.

    Algorithm:
    1. Cluster segments based on proximity of their centroids along the orthogonal axis.
    2. Collect all endpoints from segments in a cluster.
    3. Perform a linear regression (cv.fitLine) on the point cloud to find the optimal slope.
    4. Project original points onto this new vector to find the true start/end coordinates.

    Args:
        segments: Array of line segments [x1, y1, x2, y2].
        orientation: 'H' (Horizontal) or 'V' (Vertical) guiding the clustering axis.
        gap_tol_px: Maximum pixel gap to consider two segments as part of the same line.

    Returns:
        np.ndarray: Merged segments.
    """
    if len(segments) == 0:
        return np.array([])

    merged = []
    used = np.zeros(len(segments), dtype=bool)

    for i in range(len(segments)):
        if used[i]:
            continue

        # Initialize a new cluster with the current segment
        used[i] = True
        cluster_idxs = [i]

        # Calculate Centroid of the reference segment
        axc = (segments[i][0] + segments[i][2]) / 2.0
        ayc = (segments[i][1] + segments[i][3]) / 2.0

        # Greedy Search: Find other segments belonging to this linear structure
        for j in range(i+1, len(segments)):
            if used[j]:
                continue
            
            # Centroid of the candidate segment
            bxc = (segments[j][0] + segments[j][2]) / 2.0
            byc = (segments[j][1] + segments[j][3]) / 2.0

            # Check alignment based on orientation
            # If Horizontal: Check vertical distance (dy) between centroids
            # If Vertical: Check horizontal distance (dx) between centroids
            if orientation == "H":
                if abs(byc - ayc) < gap_tol_px:
                    cluster_idxs.append(j)
                    used[j] = True
            else:  # "V"
                if abs(bxc - axc) < gap_tol_px:
                    cluster_idxs.append(j)
                    used[j] = True

        # Extract all endpoints (point cloud) from the identified cluster
        pts = []
        for idx in cluster_idxs:
            x1, y1, x2, y2 = segments[idx]
            pts.append([x1, y1])
            pts.append([x2, y2])
        pts = np.array(pts, dtype=np.float32)

        # Geometric Fitting
        # If the cluster is trivial (only 2 points), we skip regression to avoid overfitting
        if pts.shape[0] < 3:
            xs = pts[:,0]
            ys = pts[:,1]
            if orientation == "H":
                y_mean = np.mean(ys)
                merged.append([int(np.min(xs)), int(y_mean), int(np.max(xs)), int(y_mean)])
            else:
                x_mean = np.mean(xs)
                merged.append([int(x_mean), int(np.min(ys)), int(x_mean), int(np.max(ys))])
            continue

        # Robust Line Fitting
        # cv.DIST_L2 = Standard Least Squares. 
        # Returns normalized vector (vx, vy) and a point on the line (x0, y0).
        vx, vy, x0, y0 = cv.fitLine(pts, cv.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)

        # Vector Projection
        # We parameterize the line as P(t) = P0 + t * V
        # We calculate 't' for every point in the cloud to find the span of the line.
        # t = DotProduct( (P - P0), V ) assuming V is normalized.
        v_norm2 = vx*vx + vy*vy
        ts = []
        for (px, py) in pts:
            t = ((px - x0)*vx + (py - y0)*vy) / v_norm2
            ts.append(t)
        
        # Identify the extremes (min/max t) to define the merged segment length
        tmin = min(ts)
        tmax = max(ts)

        p_min = (x0 + tmin*vx, y0 + tmin*vy)
        p_max = (x0 + tmax*vx, y0 + tmax*vy)
        
        merged.append([int(round(p_min[0])), int(round(p_min[1])),
                       int(round(p_max[0])), int(round(p_max[1]))])

    return np.array(merged, dtype=int)


# =============================================================================
# MODULE M1 — PIPELINE ENTRY POINT
# =============================================================================
def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    """
    Main pipeline for Line Detection.
    
    Steps:
    1. Preprocessing (Grayscale + Gaussian Blur).
    2. Edge Detection (Canny) with surface-adaptive thresholds.
    3. Probabilistic Hough Transform to find candidate segments.
    4. Spatial Filtering (ROI) to remove stadium noise.
    5. Orientation Split (Horizontal/Vertical).
    6. Collinear Merging to consolidate segments.
    """

    if image_data is None:
        return np.array([])

    # Load Hyperparameters (SNR optimization per surface)
    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common = config.HOUGH_COMMON_PARAMS
    
    # Load Spatial Filters (Region of Interest)
    centrality_params = config.CENTRALITY_PARAMS.get(surface_type.upper(), 
                                                     config.CENTRALITY_PARAMS['CEMENTO'])

    h, w, _ = image_data.shape

    # ---------------------------
    # Step 1 & 2: Preprocessing
    # ---------------------------
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    
    # Gaussian Blur is critical here to suppress high-frequency texture noise (e.g., grass blades)
    # before calculating gradients in Canny.
    blurred = cv.GaussianBlur(gray, (5,5), 1.0)
    
    # Hysteresis Thresholding via Canny
    edges = cv.Canny(blurred, params['CANNY_LOW'], params['CANNY_HIGH'])

    # ---------------------------
    # Step 3: Hough Transform
    # ---------------------------
    linesP = cv.HoughLinesP(
        edges,
        rho=common['RHO'],              # Accumulator resolution (distance)
        theta=common['THETA'],          # Accumulator resolution (angle)
        threshold=params['HOUGH_THRESHOLD'],
        minLineLength=common['MIN_LENGTH'],
        maxLineGap=common['MAX_GAP']
    )

    if linesP is None:
        return np.array([])

    segments = linesP.reshape(-1,4)

    # -----------------------------------------------------
    # Step 4: Spatial Filtering (Centrality / ROI)
    # -----------------------------------------------------
    # We calculate the geometric center of each detected segment.
    # If the center lies outside the configured percentage of the screen,
    # it is likely a grandstand, scoreboard, or barrier -> Discard.
    y_center = (segments[:,1] + segments[:,3]) / 2
    x_center = (segments[:,0] + segments[:,2]) / 2

    valid_y = (y_center > h * centrality_params['Y_MIN_PCT']) & \
              (y_center < h * centrality_params['Y_MAX_PCT'])
              
    valid_x = (x_center > w * centrality_params['X_MIN_PCT']) & \
              (x_center < w * centrality_params['X_MAX_PCT'])

    segments = segments[valid_y & valid_x]

    if len(segments) == 0:
        return np.array([])

    # -----------------------------------------------------
    # Step 5: Orientation Segmentation
    # -----------------------------------------------------
    # Calculate angle of segments to classify as Horizontal or Vertical.
    # We use arctan2 for correct quadrant handling.
    dx = segments[:,2] - segments[:,0]
    dy = segments[:,3] - segments[:,1]
    angles = np.abs(np.degrees(np.arctan2(dy,dx)) % 180)

    # Thresholds: < 45 or > 135 degrees = Horizontal
    # Everything else = Vertical
    is_h = (angles < 45) | (angles > 135)
    is_v = ~is_h

    horiz = segments[is_h]
    vert = segments[is_v]
    
    # TELEMETRY LOGGING
    print("=== DEBUG: M1 — SPATIAL FILTERS ===")
    print(f"Post-ROI Filter count: {len(segments)}")
    print("=== DEBUG: M1 — ORIENTATION SPLIT ===")
    print(f"Horizontal: {len(horiz)}  Vertical: {len(vert)}")


    # -----------------------------------------------------
    # Step 6: Collinear Merging
    # -----------------------------------------------------
    # Apply the clustering and fitting logic defined above.
    merged_h = _merge_collinear_segments(horiz, "H")
    merged_v = _merge_collinear_segments(vert, "V")
    
    print("=== DEBUG: M1 — MERGE RESULTS ===")
    print(f"Merged H: {len(merged_h)}  Merged V: {len(merged_v)}")
    
    if len(merged_h) == 0 and len(merged_v) == 0:
        return np.array([])

    # Stack results into a single array for downstream processing (Homography)
    return np.vstack([merged_h, merged_v]) if len(merged_h) and len(merged_v) else (merged_h if len(merged_h) else merged_v)
