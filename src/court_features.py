# src/court_features.py
import cv2 as cv
import numpy as np
from src import config


# ---------------------------------------------------------
# 1) ROI MASK (Invariata)
# ---------------------------------------------------------
def build_roi_mask(gray):
    h, w = gray.shape
    sobel_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    proj = np.sum(np.abs(sobel_y), axis=1)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)

    thresh = proj.mean() * 1.5
    rows = np.where(proj > thresh)[0]

    mask = np.zeros_like(gray, dtype=np.uint8)
    if len(rows) > 0:
        top = max(0, rows[0] - 20)
        bottom = min(h, rows[-1] + 20)
        mask[top:bottom, :] = 255

    return mask


# ---------------------------------------------------------
# 2) Filtro colore bianco (Invariato)
# ---------------------------------------------------------
def extract_white_pixels(image_bgr):
    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)
    return mask


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
    # Sfoca l'immagine BGR prima di usare la White Mask
    blurred_bgr = cv.GaussianBlur(image_data, (5, 5), 1.0)
    gray = cv.cvtColor(blurred_bgr, cv.COLOR_BGR2GRAY) 
    blurred_gray = cv.GaussianBlur(gray, (5, 5), 1.0) 

    # ------------------------------------------------
    # 1. RILEVAMENTO BORDI (CANNY) su IMMAGINE IN SCALA DI GRIGI INTERA
    # Questo è il punto chiave: Canny vede tutti i bordi prima del filtraggio
    edges = cv.Canny(blurred_gray, params['CANNY_LOW'], params['CANNY_HIGH'])
    # ------------------------------------------------

    # --- 2. CREAZIONE MASCHERA DI FILTRAGGIO ---
    
    # Maschera ROI (geometrica)
    roi_mask = build_roi_mask(gray)
    
    # Maschera Colore (bianco)
    white_mask = extract_white_pixels(blurred_bgr) 

    # OTTIMIZZAZIONE: Combina le due maschere (solo AND)
    combined_mask = cv.bitwise_and(roi_mask, white_mask) 
    
    # --- 3. APPLICAZIONE FILTRI/MASCHERE all'output di Canny ---
    # Applica la maschera all'output di Canny (Filtra i bordi non bianchi e fuori ROI)
    masked_edges = cv.bitwise_and(edges, edges, mask=combined_mask)

    # --- Hough Probabilistico ---
    linesP = cv.HoughLinesP(
        masked_edges, # USA l'output di Canny filtrato (masked_edges)
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
    #  Filtri su angolo (verticali/orizzontali)
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
