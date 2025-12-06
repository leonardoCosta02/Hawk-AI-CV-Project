# src/court_features.py - Versione finale pulita

import cv2 as cv
import numpy as np
from src import config # Importa il file di configurazione (necessario per i parametri)

def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    """
    Esegue il preprocessing e l'estrazione delle linee, usando i parametri
    ottimali specifici per la superficie definiti in config.py.

    Args:
        image_data: Il frame statico del campo da tennis letto da OpenCV.
        surface_type: Tipo di campo ('CEMENTO', 'ERBA', 'TERRA_BATTUTA').

    Returns:
        Un array NumPy contenente i segmenti di linea raw in formato [[x1, y1, x2, y2], ...]. 
    """
    if image_data is None:
        return np.array([])
    
    # 1. RECUPERO PARAMETRI (Legge i parametri specifici per la superficie)
    
    # Prende i parametri specifici per Canny/Hough
    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    # Prende i parametri di Hough comuni
    common_hough = config.HOUGH_COMMON_PARAMS

    # 2. PREPROCESSING
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 3. EDGE DETECTION (Canny) - Usa i parametri specifici della superficie
    edges = cv.Canny(blurred, params['CANNY_LOW'], params['CANNY_HIGH'])
    
    # 4. LINE DETECTION (Probabilistic Hough Transform) - Usa i parametri specifici
    raw_lines = cv.HoughLinesP(
        edges,
        rho=common_hough['RHO'],
        theta=common_hough['THETA'],
        threshold=params['HOUGH_THRESHOLD'], # Usa la soglia specifica
        minLineLength=common_hough['MIN_LENGTH'],
        maxLineGap=common_hough['MAX_GAP']
    )

    # 5. OUTPUT
    
    # All'interno di src/court_features.py, modifica la sezione 5. OUTPUT

    # 1. Recupera il parametro MAX_LENGTH
    common_hough = config.HOUGH_COMMON_PARAMS
    MAX_PIXEL_LENGTH = common_hough['MAX_LENGTH'] # Nuovo parametro

    # 5. OUTPUT
    if raw_lines is not None:
        # Riformatta raw_lines in un array 2D
        line_segments = raw_lines.reshape(-1, 4)
        return line_segments
    else:
        return np.array([])
