# src/court_features.py - Versione finale con Filtri Geometrici

import cv2 as cv
import numpy as np
from src import config 

def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    """
    Esegue il preprocessing, l'estrazione delle linee, e i filtri geometrici
    (Lunghezza e Angolo) per pulire il rumore.
    """
    if image_data is None:
        return np.array([])
    
    # 1. RECUPERO PARAMETRI
    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common_hough = config.HOUGH_COMMON_PARAMS
    
    # 2. PREPROCESSING
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 3. EDGE DETECTION (Canny)
    edges = cv.Canny(blurred, params['CANNY_LOW'], params['CANNY_HIGH'])
    
    # 4. LINE DETECTION (Probabilistic Hough Transform)
    raw_lines = cv.HoughLinesP(
        edges,
        rho=common_hough['RHO'],
        theta=common_hough['THETA'],
        threshold=params['HOUGH_THRESHOLD'],
        minLineLength=common_hough['MIN_LENGTH'],
        maxLineGap=common_hough['MAX_GAP']
    )

    # 5. OUTPUT CON FILTRI GEOMETRICI (Lunghezza e Angolo)
    if raw_lines is not None:
        lines = raw_lines.reshape(-1, 4)
        h, w, _ = image_data.shape # Altezza e Larghezza (necessarie per MAX_LENGTH)

        # --- FASE 1: Calcola Lunghezza e Angolo ---
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx**2 + dy**2)
        angles_rad = np.arctan2(dy, dx)
        angles_deg = np.abs(np.degrees(angles_rad) % 180)

        # --- FASE 2: Maschera di Lunghezza ---
        MIN_LENGTH_FILTER = common_hough.get('MIN_LENGTH', 60)
        # Il 90% della larghezza è un buon limite per eliminare rumore molto lungo.
        MAX_LENGTH_FILTER = w * 0.9 
        is_valid_length = (lengths >= MIN_LENGTH_FILTER) & (lengths <= MAX_LENGTH_FILTER)

        # --- FASE 3: Maschera Angolare (Orizzontale / Verticale) ---
        ANGLE_TOLERANCE = common_hough.get('ANGLE_TOLERANCE_DEG', 10) 
        is_valid_angle = (angles_deg < ANGLE_TOLERANCE) | \
                         (angles_deg > 180 - ANGLE_TOLERANCE) | \
                         ((angles_deg > 90 - ANGLE_TOLERANCE) & (angles_deg < 90 + ANGLE_TOLERANCE))

        # --- FASE 4: Applica la maschera finale ---
        final_mask = is_valid_length & is_valid_angle
        filtered_lines = lines[final_mask]
        
        return filtered_lines
    else:
        return np.array([])
