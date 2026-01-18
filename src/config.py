# config.py
"""
Configuration Module for Tennis Court Detection System.

This module centralizes all hyperparameters required for the computer vision pipeline,
specifically targeting:
1. Hough Transform parameters for line detection.
2. Canny Edge Detection thresholds, tuned per surface type (Hard, Grass, Clay).
3. Region of Interest (ROI) filtering to exclude stadium noise.
4. Real-world metric dimensions for Homography and 2D-to-3D mapping.
"""

import numpy as np 

# =============================================================================
# SECTION 1: COMMON HOUGH TRANSFORM PARAMETERS (Module M1)
# =============================================================================
# These parameters control the Probabilistic Hough Line Transform (cv2.HoughLinesP).
# They act as a baseline configuration across all court surfaces.
HOUGH_COMMON_PARAMS = {
    # Distance resolution of the accumulator in pixels.
    'RHO': 1,                
    
    # Angle resolution of the accumulator in radians (1 degree).
    'THETA': np.pi / 180,    
    
    # Minimum line length. Line segments shorter than this are rejected.
    # Optimization Note: Increased to filter out short artifacts (noise) 
    # while retaining main court lines.
    'MIN_LENGTH': 60,        
    
    # Maximum allowed gap between points on the same line to link them.
    # Critical for detecting dashed lines or lines interrupted by occlusion (e.g., players).
    'MAX_GAP': 35,           
    
    # Tolerance for classifying lines as Horizontal or Vertical during post-processing.
    'ANGLE_TOLERANCE_DEG': 180,
}

# =============================================================================
# SECTION 2: DATA LOADING PORTS
# =============================================================================
# File paths for static reference frames used for calibration/testing.
CAMPI_PATH = {
    "CEMENTO": 'data/static_court/static_court_frame_cemento.png',       # Hard Court
    "ERBA": 'data/static_court/static_court_frame_erba.png',             # Grass
    "TERRA_BATTUTA": 'data/static_court/static_court_frame_clay.png',    # Clay
}


# =============================================================================
# SECTION 3: SURFACE-SPECIFIC CANNY & HOUGH OPTIMIZATION
# =============================================================================
# Different surfaces present unique Computer Vision challenges (texture, contrast, reflectivity).
# We define distinct hyperparameter sets to maximize the Signal-to-Noise Ratio (SNR) for each.

# --- HARD COURT (CEMENTO) ---
# Characteristics: High contrast, low texture noise, but potential for reflections.
# Objective: Reduce false positives (previously ~21 lines detected).
PARAMS_CEMENTO = {
    'CANNY_LOW': 15,          # Lower hysteresis threshold for edge linking
    'CANNY_HIGH': 120,        # Upper hysteresis threshold for strong edge initialization
    'HOUGH_THRESHOLD': 60,    # Accumulator threshold: minimum votes to accept a line
    'FRAME_PATH': CAMPI_PATH['CEMENTO'],
}

# --- GRASS COURT (ERBA) ---
# Characteristics: Variable texture due to wear, lower contrast on worn baselines.
# Objective: Detect weaker edge gradients while maintaining low noise floor.
PARAMS_ERBA = {
    'CANNY_LOW': 15,
    'CANNY_HIGH': 150,        # Higher threshold to strictly identify strong edges first
    'HOUGH_THRESHOLD': 65,    # Slightly stricter voting to avoid grass texture artifacts
    'FRAME_PATH': CAMPI_PATH['ERBA'], 
}

# --- CLAY COURT (TERRA_BATTUTA) ---
# Characteristics: High frequency noise (granular surface), foot marks, sliding traces.
# Objective: Aggressive filtering to handle the noisiest environment (previously ~101 lines).
PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,          # Significantly raised low threshold to reject surface texture
    'CANNY_HIGH': 150,        
    'HOUGH_THRESHOLD': 60,    
    'FRAME_PATH': CAMPI_PATH['TERRA_BATTUTA'],
}

# Lookup dictionary for dynamic parameter injection based on selected surface.
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
}


# =============================================================================
# SECTION 4: SPATIAL FILTERING (CENTRALITY / ROI)
# =============================================================================
# Defines the Region of Interest (ROI) using normalized coordinates (0.0 to 1.0).
# This creates a logical mask to exclude peripheral noise such as:
# crowds, stadium architecture, and scoreboard graphics.

CENTRALITY_PARAMS = {
    # HARD COURT: Optimized for standard broadcast camera angles.
    'CEMENTO': {
        'Y_MIN_PCT': 0.30, # Top crop (removes audience/stands)
        'Y_MAX_PCT': 0.75, # Bottom crop (focuses on court area)
        'X_MIN_PCT': 0.30, # Left crop
        'X_MAX_PCT': 0.70, # Right crop
    },
    # GRASS: Adjusted to handle perspective distortion common in grass venues.
    # Note: Wider vertical range (0.84) to capture baselines despite perspective.
    'ERBA': {
        'Y_MIN_PCT': 0.30,
        'Y_MAX_PCT': 0.84,
        'X_MIN_PCT': 0.25, # Wider horizontal search area
        'X_MAX_PCT': 0.75,
    },
    # CLAY: Challenging detection environment.
    # ROI constraints help mitigate false edge detection from clay sweepers/drag nets.
    'TERRA_BATTUTA': {
        'Y_MIN_PCT': 0.30,
        'Y_MAX_PCT': 0.84,
        'X_MIN_PCT': 0.30,
        'X_MAX_PCT': 0.70,
    },
}

# =============================================================================
# SECTION 5: WORLD METRICS & HOMOGRAPHY (Module M3)
# =============================================================================
# Standard ITF (International Tennis Federation) dimensions in Meters.
# Essential for computing the Homography matrix to map pixel coordinates to real-world space.

COURT_DIMENSIONS_METERS = {
    'SINGOLO_LARGHEZZA': 8.23,   # Singles width
    'DOPPIO_LARGHEZZA': 10.97,   # Doubles width
    'LUNGHEZZA_TOTALE': 23.77,   # Total length
    'SERVIZIO_RETE': 6.40,       # Distance from Net to Service Line
    'BASE_SERVIZIO': 5.49,       # Distance from Service Line to Baseline
}

# Reference Keypoints in World Coordinates (Meters).
# Used as 'dst_points' in cv2.findHomography().
# Mapping represents the "Near Service Box" area.
POINTS_WORLD_METERS = np.float32([
    [0.0, 0.0], # Angolo fondo-laterale SX
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], 0.0], # Angolo fondo-laterale DX
    [0.0, 5.485], # Punto sulla linea di servizio SX 
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], 5.485], # Punto sulla linea di servizio DX
])
