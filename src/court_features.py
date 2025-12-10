# src/court_features.py - Versione finale con Cropping e Filtri

import cv2 as cv
import numpy as np
from src import config 

def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    
    if image_data is None:
        return np.array([])
    
    # 1. RECUPERO PARAMETRI E CROPPING
    
    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common_hough = config.HOUGH_COMMON_PARAMS
    
    # --- APPLICAZIONE CROPPING (Controllo che avvenga qui) ---
    crop_coords = config.CROPPING_PARAMS.get(surface_type.upper())
    offset_x, offset_y = 0, 0 # Inizializziamo gli offset

    if crop_coords:
        x_start, y_start, x_end, y_end = crop_coords
        
        # Salviamo gli offset per i test: il cropping inizia da qui.
        offset_x, offset_y = x_start, y_start 
        
        # Il cropping in NumPy è: [Y_start:Y_end, X_start:X_end]
        image_data = image_data[y_start:y_end, x_start:x_end].copy()
        
        if image_data.size == 0:
            print(f"Errore Cropping: Area di ritaglio non valida per {surface_type}.")
            return np.array([])
            
    # La larghezza (w) è ora quella dell'immagine ritagliata
    w = image_data.shape[1] 

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

    # 5. FILTRO DI LUNGHEZZA E ANGOLO
    if raw_lines is not None:
        lines = raw_lines.reshape(-1, 4)
        
        # 5.1 Calcola Lunghezza e Angolo
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx**2 + dy**2)
        angles_rad = np.arctan2(dy, dx)
        angles_deg = np.abs(np.degrees(angles_rad) % 180)

        # 5.2 Maschera di Lunghezza
        MIN_LENGTH_FILTER = common_hough.get('MIN_LENGTH', 60)
        MAX_LENGTH_FILTER = w * 0.9 # 90% della larghezza immagine
        is_valid_length = (lengths >= MIN_LENGTH_FILTER) & (lengths <= MAX_LENGTH_FILTER)

        # 5.3 Maschera Angolare (prospettiva)
        ANGLE_TOLERANCE = common_hough.get('ANGLE_TOLERANCE_DEG', 10) 
        is_valid_angle = (angles_deg < ANGLE_TOLERANCE) | \
                         (angles_deg > 180 - ANGLE_TOLERANCE) | \
                         ((angles_deg > 90 - ANGLE_TOLERANCE) & (angles_deg < 90 + ANGLE_TOLERANCE))

        
        # Applica la doppia maschera
        final_mask = is_valid_length & is_valid_angle
        filtered_lines = lines[final_mask]
        
        return filtered_lines
    else:
        return np.array([])
