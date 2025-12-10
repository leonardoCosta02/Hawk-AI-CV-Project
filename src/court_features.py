# src/court_features.py
import cv2 as cv
import numpy as np
from src import config

# ---------------------------------------------------------
# 1) & 2) Funzioni di Maschera RIMOSSE per semplificazione
# ---------------------------------------------------------

# ---------------------------------------------------------
# 3) merge_collinear_segments - RIMOSSA (Correzione Geometria)
# ---------------------------------------------------------
# NOTA: La funzione merge_collinear_segments è stata rimossa.


# ---------------------------------------------------------
# 4) Funzione principale — RESTITUISCE SEGMENTI (x1,y1,x2,y2)
# ---------------------------------------------------------
def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    if image_data is None:
        return np.array([])

    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common = config.HOUGH_COMMON_PARAMS

    # --- Preprocessing ---
    # Converti direttamente in scala di grigi
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY) 
    # Sfoca per ridurre il rumore
    blurred_gray = cv.GaussianBlur(gray, (5, 5), 1.0) 

    # --- Canny ---
    # Applicazione diretta dei soli parametri Canny
    edges = cv.Canny(blurred_gray, params['CANNY_LOW'], params['CANNY_HIGH'])

    # --- Hough Probabilistico ---
    linesP = cv.HoughLinesP(
        edges, # Input diretto da Canny (non mascherato)
        rho=common['RHO'],
        theta=common['THETA'],
        threshold=params['HOUGH_THRESHOLD'], 
        minLineLength=common['MIN_LENGTH'], 
        maxLineGap=common['MAX_GAP']
    )

    if linesP is None:
        return np.array([])

    segments = linesP.reshape(-1, 4)

    # -----------------------------------------------------
    #  Filtri su angolo (verticali/orizzontali) - ESSENZIALE
    # -----------------------------------------------------
    
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    angles = np.abs(np.degrees(np.arctan2(dy, dx)) % 180)

    tol = common['ANGLE_TOLERANCE_DEG']
    valid_angle = (
        (angles < tol) |                       
        (angles > 180 - tol) |                 
        ((angles > 90 - tol) & (angles < 90 + tol)) 
    )

    segments = segments[valid_angle]
    
    # -----------------------------------------------------
    #  Restituzione
    # -----------------------------------------------------
    return segments
