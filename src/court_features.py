# src/court_features.py
import cv2 as cv
import numpy as np
from src import config

# ---------------------------------------------------------
# 4) Funzione principale — RESTITUISCE SEGMENTI (x1,y1,x2,y2)
# ---------------------------------------------------------
def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    if image_data is None:
        return np.array([])

    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    common = config.HOUGH_COMMON_PARAMS
    h, w, _ = image_data.shape # Altezza e larghezza dell'immagine

    # --- Preprocessing ---
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY) 
    blurred_gray = cv.GaussianBlur(gray, (5, 5), 1.0) 

    # --- Canny & Hough ---
    edges = cv.Canny(blurred_gray, params['CANNY_LOW'], params['CANNY_HIGH'])
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

    segments = linesP.reshape(-1, 4)

    # -----------------------------------------------------
    # NUOVO FILTRO 1: FILTRO DI CENTRALITÀ (POSIZIONE Y)
    # Rimuove i segmenti che sono troppo alti nell'immagine (dove non c'è campo)
    # Assumiamo che il campo inizi al 20% della parte alta dell'immagine (0.2 * H)
    # e che la linea di fondo sia sotto il 95% dell'immagine (0.95 * H).
    # -----------------------------------------------------
    
    # Calcola il punto medio Y di ogni segmento
    y_center = (segments[:, 1] + segments[:, 3]) / 2
    
    # Soglie in base alla dimensione dell'immagine (in pixel)
    Y_MIN_PIXEL = h * 0.30 # Minimo 20% dall'alto (esclude il pubblico)
    Y_MAX_PIXEL = h * 0.95 # Massimo 95% dall'alto (esclude il rumore del fondo)
    
    valid_y = (y_center > Y_MIN_PIXEL) & (y_center < Y_MAX_PIXEL)
    segments = segments[valid_y]
    
    # Se dopo il filtro Y non ci sono segmenti, esci
    if len(segments) == 0:
        return np.array([])
    
    # -----------------------------------------------------
    # NUOVO FILTRO 2: FILTRO DI CENTRALITÀ (POSIZIONE X)
    # Rimuove i segmenti troppo ai lati (che non fanno parte del campo)
    # Assumiamo che il campo finisca al 95% della larghezza (0.95 * W).
    # -----------------------------------------------------
    
    x_center = (segments[:, 0] + segments[:, 2]) / 2
    
    X_MIN_PIXEL = w * 0.20
    X_MAX_PIXEL = w * 0.80
    
    valid_x = (x_center > X_MIN_PIXEL) & (x_center < X_MAX_PIXEL)
    segments = segments[valid_x]

    # -----------------------------------------------------
    # Filtri su angolo (RIATTIVATO per eliminare il rumore di sfondo)
    # Non possiamo eliminarlo, altrimenti il Modulo M3 non funzionerà!
    # -----------------------------------------------------
    """
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
    """
    # -----------------------------------------------------
    # Restituzione
    # -----------------------------------------------------
    return segments
