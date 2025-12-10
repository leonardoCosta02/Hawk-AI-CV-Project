# src/court_features.py - Versione finale pulita con filtri geometrici

import cv2 as cv
import numpy as np
from src import config # Importa il file di configurazione (necessario per i parametri)

def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    """
    Esegue il preprocessing, l'estrazione delle linee e l'applicazione di un 
    filtro di Regione di Interesse (ROI), Lunghezza e Angolo.

    Args:
        image_data: Il frame statico del campo da tennis letto da OpenCV.
        surface_type: Tipo di campo ('CEMENTO', 'ERBA', 'TERRA_BATTUTA').

    Returns:
        Un array NumPy contenente i segmenti di linea filtrati in formato [[x1, y1, x2, y2], ...]. 
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

    # 5. OUTPUT CON FILTRI GEOMETRICI (ROI + Lunghezza + Angolo)
    if raw_lines is not None:
        lines = raw_lines.reshape(-1, 4)
        h, w, _ = image_data.shape # Altezza e Larghezza

        # --- FASE A: Filtro di Regione di Interesse (ROI) ---
        X_MIN = int(w * common_hough['ROI_LEFT_PCT'])
        X_MAX = int(w * common_hough['ROI_RIGHT_PCT'])
        Y_MIN = int(h * common_hough['ROI_TOP_PCT'])
        
        # Un segmento è valido se i suoi estremi rientrano nella ROI.
        is_x_valid = (lines[:, 0] >= X_MIN) & (lines[:, 0] <= X_MAX) & \
                     (lines[:, 2] >= X_MIN) & (lines[:, 2] <= X_MAX)
        
        # Filtro verticale: esclude spalti (Y_MIN)
        is_y_valid = (lines[:, 1] >= Y_MIN) & (lines[:, 3] >= Y_MIN)

        roi_mask = is_x_valid & is_y_valid
        lines = lines[roi_mask]
        
        if lines.size == 0:
            return np.array([])

        # --- FASE B: Filtro di Lunghezza e Angolo ---
        
        # 1. Calcola Lunghezza e Angolo
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx**2 + dy**2)
        angles_rad = np.arctan2(dy, dx)
        angles_deg = np.abs(np.degrees(angles_rad) % 180)

        # 2. Maschera di Lunghezza
        MIN_LENGTH_FILTER = common_hough.get('MIN_LENGTH', 60)
        MAX_LENGTH_FILTER = w * 0.7 # 70% della larghezza immagine
        is_valid_length = (lengths >= MIN_LENGTH_FILTER) & (lengths <= MAX_LENGTH_FILTER)

        # 3. Maschera Angolare (solo Orizzontale o Verticale)
        ANGLE_TOLERANCE = 5 # Usa una tolleranza fissa di 5 gradi
        is_valid_angle = (angles_deg < ANGLE_TOLERANCE) | \
                         (angles_deg > 180 - ANGLE_TOLERANCE) | \
                         ((angles_deg > 90 - ANGLE_TOLERANCE) & (angles_deg < 90 + ANGLE_TOLERANCE))

        # --- FASE C: Applica la maschera finale ---
        final_mask = is_valid_length & is_valid_angle
        filtered_lines = lines[final_mask]
        
        return filtered_lines
    else:
        return np.array([])
