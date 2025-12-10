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
        Un array NumPy contenente i segmenti di linea filtrati in formato [[x1, y1, x2, y2], ...]. 
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
    # Nota: MIN_LENGTH è qui usato per il filtro iniziale di OpenCV
    raw_lines = cv.HoughLinesP(
        edges,
        rho=common_hough['RHO'],
        theta=common_hough['THETA'],
        threshold=params['HOUGH_THRESHOLD'], # Usa la soglia specifica
        minLineLength=common_hough['MIN_LENGTH'],
        maxLineGap=common_hough['MAX_GAP']
    )

    # 5. OUTPUT CON FILTRO DI LUNGHEZZA AVANZATO (Minimo e Massimo)
    if raw_lines is not None:
        lines = raw_lines.reshape(-1, 4)
        
        # 5.1 Calcola la lunghezza effettiva di ogni segmento (Teorema di Pitagora)
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        lengths = np.sqrt(dx**2 + dy**2)
        
        # 5.2 Definisce i parametri di filtro Min e Max Length
        
        # Usa il valore MIN_LENGTH dal tuo config.py (che è 60 pixel)
        MIN_LENGTH_FILTER = common_hough.get('MIN_LENGTH', 60) 
        
        # MAX_LENGTH_FILTER: Imposta un limite superiore (70% della larghezza dell'immagine)
        # per rimuovere linee extra lunghe e rumorose (es. bordi lontani).
        MAX_LENGTH_FILTER = image_data.shape[1] * 0.7 
        
        # 5.3 Crea la maschera di filtro
        # Accetta solo segmenti che rientrano nell'intervallo di lunghezza
        is_valid_length = (lengths >= MIN_LENGTH_FILTER) & (lengths <= MAX_LENGTH_FILTER)
                         
        # Applica la maschera e restituisce i segmenti filtrati
        filtered_lines = lines[is_valid_length]
        
        return filtered_lines
    else:
        return np.array([])
