# src/homography_calculator.py
"""
Homography Estimation Module (M3).

This module is responsible for bridging 2D Pixel Space and 3D World Space.
It performs the following critical tasks:
1. Dominant Orientation Analysis: Uses histograms to find the court's main axes (H/V).
2. Semantic Line Classification: Categorizes segments into 'Horizontal' (Base/Service) 
   and 'Vertical' (Sidelines) based on angular proximity.
3. Keypoint Extraction: Computes intersections of these semantic lines.
4. Homography Calculation: Computes the 3x3 transformation matrix (H) to map 
   pixels to real-world meters.
"""

import cv2 as cv
import numpy as np
from src import config

# ANSI Colors for terminal telemetry
RED = "\033[91m"
GREEN = "\033[92m"
ENDC = "\033[0m"

# ============================================================
#  GEOMETRIC UTILITIES
# ============================================================
def find_intersection(s1, s2):
    """
    Computes the intersection point of two infinite lines defined by segments.
    
    Uses Determinants (Cramer's Rule) for linear system solution.
    P = P1 + t(P2-P1).
    
    Args:
        s1, s2: Line segments [x1, y1, x2, y2]
        
    Returns:
        (Px, Py): Intersection coordinates, or (None, None) if parallel.
    """
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2

    # Determinant of the coefficient matrix
    # Represents the cross product of the two direction vectors.
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Check for parallel lines (D near 0) to avoid division by zero
    if abs(D) < 1e-4:
        return None, None

    # Solve for parameter t (intersection along first line)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / D
    
    Px = x1 + t * (x2 - x1)
    Py = y1 + t * (y2 - y1)
    return Px, Py

def angular_dist(a, b):
    """
    Computes the shortest distance between two angles in [0, 180) space.
    Essential for line orientation clustering where 179° is close to 1°.
    """
    d = abs(a - b)
    return min(d, 180.0 - d)


# ============================================================
#  M3 — HOMOGRAPHY PIPELINE
# ============================================================
def calculate_homography(all_line_segments, surface_type='CEMENTO'):
    """
    Derives the Homography Matrix from a soup of line segments.
    
    Algorithm:
    1. Angle Histogram: Identify dominant court orientations (Main H & V axes).
    2. Classification: Assign every segment to 'Horizontal' or 'Vertical' groups.
    3. Template Matching: Identify specific lines (Baseline, Service, Sidelines) 
       using spatial heuristics (e.g., Baseline is usually the lowest line in Y).
    4. Intersection: Compute 4 corners of the court area.
    5. Calibration: Compute H matrix mapping these corners to known metric dimensions.
    """
    
    print("\n\n========================")
    print(f"🏁 HOMOGRAPHY CALCULATION START: {surface_type}")
    print("========================\n")

    if all_line_segments is None or len(all_line_segments) < 4:
        print(f"{RED}Error: Insufficient segments ({len(all_line_segments) if all_line_segments is not None else 0}) for calibration.{ENDC}")
        return None, None, None

    print(f"[DEBUG] Input Segments: {len(all_line_segments)}")
    
    # ---------------------------------------------------------
    # 1) Feature Extraction (Angles & Lengths)
    # ---------------------------------------------------------
    # Calculate orientation for every segment. 
    # Tennis courts have strong orthogonality, so we expect bimodal distribution.
    dx = all_line_segments[:, 2] - all_line_segments[:, 0]
    dy = all_line_segments[:, 3] - all_line_segments[:, 1]
    angles = (np.degrees(np.arctan2(dy, dx)) % 180.0)
    lengths = np.sqrt(dx**2 + dy**2)


    # ---------------------------------------------------------
    # 2) Dominant Axis Estimation (Histogram Analysis)
    # ---------------------------------------------------------
    # We construct a histogram of angles to find the "Vanishing Points" (sort of).
    hist_bins = 180
    hist, bin_edges = np.histogram(angles, bins=hist_bins, range=(0.0, 180.0))
    
    # Smooth the histogram to suppress noise and find robust peaks.
    kernel = np.array([1.0, 1.0, 1.0])
    hist_smooth = np.convolve(hist.astype(float), kernel / kernel.sum(), mode='same')
    
    peak_angles = []
    if not np.all(hist_smooth == 0):
        # Find the top 2 peaks in the angular distribution
        peak_bins = np.argsort(hist_smooth)[-2:]
        peak_angles = bin_edges[peak_bins] + (bin_edges[1] - bin_edges[0]) / 2.0
        peak_angles = np.sort(peak_angles)

    
    # ---------------------------------------------------------
    # 3) Robust Orientation Classification
    # ---------------------------------------------------------
    # Initialize defaults (Horizontal=0°, Vertical=90°)
    theta_h = 0.0
    theta_v = 90.0
    
    if len(peak_angles) >= 2 and angular_dist(peak_angles[0], peak_angles[1]) >= 8.0:
        
        # Identify which peak corresponds to Horizontal lines (closer to 0° or 180°)
        # In broadcast view, baselines are nearly horizontal.
        dist0 = [min(angular_dist(pa, 0.0), angular_dist(pa, 180.0)) for pa in peak_angles]
        h_idx = np.argmin(dist0)
        v_idx = 1 - h_idx
        
        theta_h = peak_angles[h_idx]
        theta_v = peak_angles[v_idx]

        # ** CRITICAL FIX: SWAP CHECK **
        # Sometimes, perspective distortion or noise makes a vertical line (e.g., 75°)
        # appear as the "most horizontal" candidate if true horizontals are weak.
        # Heuristic: If "horizontal" angle is too steep (>15°), the classification likely flipped.
        if angular_dist(theta_h, 0.0) > 15.0:
            print(f"{RED}[FIX] SWAP DETECTED: theta_h ({theta_h:.2f}°) is too steep. Swapping axes.{ENDC}")
            theta_h, theta_v = theta_v, theta_h 

    else:
        # Fallback: Assume the longest segment found dictates the primary horizontal axis.
        longest_index = np.argmax(lengths)
        theta_h = angles[longest_index]
        theta_v = (theta_h + 90.0) % 180.0
        print(f"{RED}[WARN] Histogram unreliable. Fallback to longest segment: {theta_h:.2f}°{ENDC}")

    print(f"[DEBUG] FINAL ANGLES -> theta_h: {theta_h:.2f}°, theta_v: {theta_v:.2f}°")

    H_segments = []
    V_segments = []

    # Cluster segments based on proximity to the determined dominant axes
    for seg, ang in zip(all_line_segments, angles):
        if angular_dist(ang, theta_h) <= angular_dist(ang, theta_v):
            H_segments.append(seg)
        else:
            V_segments.append(seg)

    H_segments = np.array(H_segments, dtype=float) if len(H_segments) else np.empty((0, 4), dtype=float)
    V_segments = np.array(V_segments, dtype=float) if len(V_segments) else np.empty((0, 4), dtype=float)

    print(f"[DEBUG] Classified Horizontal: {len(H_segments)}")
    print(f"[DEBUG] Classified Vertical: {len(V_segments)}")

    if len(H_segments) < 2 or len(V_segments) < 2:
        print(f"{RED}Error: Need at least 2 Horizontal and 2 Vertical lines.{ENDC}")
        return None, None, None 


    # ---------------------------------------------------------
    # 4) Semantic Line Selection (Heuristics)
    # ---------------------------------------------------------
    print("\n[DEBUG] --- TEMPLATE FITTING ---")

    # Compute centroids to sort lines spatially
    h_y = (H_segments[:, 1] + H_segments[:, 3]) / 2.0
    v_x = (V_segments[:, 0] + V_segments[:, 2]) / 2.0

    # Heuristic: The Baseline is the "Lowest" horizontal line (Max Y) in the image
    # Heuristic: The Service Line is the "Second Lowest" horizontal line
    h_sorted = np.argsort(h_y)[::-1]
    base_line = H_segments[h_sorted[0]]     # Max Y
    service_line = H_segments[h_sorted[1]]  # 2nd Max Y
    
    # Heuristic: Left Sideline is the left-most vertical line (Min X)
    # Heuristic: Right Sideline is the right-most vertical line (Max X)
    v_sorted = np.argsort(v_x)
    side_left = V_segments[v_sorted[0]]     # Min X
    side_right = V_segments[v_sorted[-1]]   # Max X

    print("\n[DEBUG] Semantic Lines Identified:")
    print("  Base      :", base_line)
    print("  Service   :", service_line)
    print("  Left Side :", side_left)
    print("  Right Side:", side_right)

    selected_segments = np.array([base_line, service_line, side_left, side_right], dtype=float)

    # ---------------------------------------------------------
    # 5) Intersection & Homography Matrix Calculation
    # ---------------------------------------------------------
    print("\n[DEBUG] --- INTERSECTION COMPUTATION ---")

    # 
    # We are reconstructing the "Near Service Box + Baseline Area" quadrilateral.
    p1 = find_intersection(base_line, side_left)    # Bottom-Left
    p2 = find_intersection(base_line, side_right)   # Bottom-Right
    p3 = find_intersection(service_line, side_left) # Top-Left (Net side)
    p4 = find_intersection(service_line, side_right)# Top-Right (Net side)

    print("  p1 (BL):", p1)
    print("  p2 (BR):", p2)
    print("  p3 (TL):", p3)
    print("  p4 (TR):", p4)

    pts = [p1, p2, p3, p4]
    
    # Validity Checks
    for i, p in enumerate(pts, start=1):
        if p is None or p[0] is None or p[1] is None or not np.isfinite(p[0]) or not np.isfinite(p[1]):
            print(f"{RED}Error: Invalid intersection at point p{i}.{ENDC}")
            return None, None, None

    points_pix = np.float32(pts)

    # Sanity Check: Quadrilateral Area
    # If the area is too small, the lines collapsed or intersected at infinity.
    def quad_area(quad):
        x = quad[:, 0]
        y = quad[:, 1]
        # Shoelace formula for polygon area
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # Note: Point order for Shoelace must be sequential (perimeter order).
    # p1(BL) -> p2(BR) -> p4(TR) -> p3(TL)
    area = quad_area(points_pix[[0, 1, 3, 2]]) 
    
    if area < 10000.0:
        print(f"{RED}Error: Degenerate quadrilateral area ({area:.2f}). Calibration failed.{ENDC}")
        return None, None, None

    # 6) Compute Homography Matrix (H)
    # We map the detected pixel points (src) to defined world metric points (dst)
    # defined in config.POINTS_WORLD_METERS.
    # Note: We take only the first 4 points from world config matching our 4 intersections.
    points_world = config.POINTS_WORLD_METERS[:4]

    # cv.RANSAC helps ignore outliers if we had more than 4 points, 
    # but here it adds robustness against slight drift.
    H, mask = cv.findHomography(points_pix, points_world, cv.RANSAC, 5.0)

    if H is None:
        print(f"{RED}Error: cv.findHomography returned None.{ENDC}")
        return None, None, None

    print(f"{GREEN}\n=== HOMOGRAPHY MATRIX COMPUTED SUCCESSFULLY ==={ENDC}")
    print(H)

    # Returning:
    # 1. H Matrix (for projection)
    # 2. Selected Segments (for visualization/debugging)
    # 3. Pixel Points (for checking intersection accuracy)
    return H, selected_segments.astype(np.int32), points_pix.astype(np.int32)


# ============================================================
#  INVERSE MAPPING UTILITY
# ============================================================
def map_pixel_to_world(H, pixel_coords):
    """
    Projects a 2D point (u,v) into 3D World Coordinates (X,Y) using matrix H.
    
    Formula: 
    [x', y', w']^T = H * [u, v, 1]^T
    X = x'/w', Y = y'/w'
    """
    if H is None:
        return np.array([0.0, 0.0])
        
    u, v = pixel_coords
    
    # Homogeneous coordinates
    ph = np.array([u, v, 1.0])
    
    # Matrix multiplication
    wh = H @ ph
    
    # Perspective division (normalization)
    # This divides by the depth factor 'w' to get cartesian coordinates.
    X = wh[0]/wh[2]
    Y = wh[1]/wh[2]
    
    return np.array([X, Y])
