# config.py
import numpy as np # Necessario per definire la costante matematica 'np.pi'

# --- PARAMETRI DI HOUGH COMUNI (M1) ---
HOUGH_COMMON_PARAMS = {
    'RHO': 1,               
    'THETA': np.pi / 180,   
    'MIN_LENGTH': 60,       
    'MAX_GAP': 15,          
    'ANGLE_TOLERANCE_DEG': 10, # Tolleranza angolare per il filtro in Omografia (FASE A)
}

# --- LISTA DEI PERCORSI DEI FRAME PER IL CARICAMENTO ---
CAMPI_PATH = {
    "CEMENTO": 'data/static_court/static_court_frame_cemento.png',
    "ERBA": 'data/static_court/static_court_frame_erba.png',
    "TERRA_BATTUTA": 'data/static_court/static_court_frame_clay.png',
}

# --- PARAMETRI DI CROPPING FISSI (M1) (Forniti dall'utente) ---
# Queste coordinate (x_start, y_start, x_end, y_end) isolano il campo da tennis
CROPPING_PARAMS = {
    'CEMENTO': [100, 200, 1100, 800],    # Ritaglio per CEMENTO
    'ERBA': [50, 150, 1200, 900],        # Ritaglio per ERBA
    'TERRA_BATTUTA': [200, 100, 1000, 750], # Ritaglio per TERRA_BATTUTA
}


# --- PARAMETRI OTTIMALI CANNY (M1) PER SUPERFICIE ---

PARAMS_CEMENTO = {
    'CANNY_LOW': 25,        
    'CANNY_HIGH': 70,       
    'HOUGH_THRESHOLD': 40,   
    'FRAME_PATH': CAMPI_PATH['CEMENTO'],
}

PARAMS_ERBA = {
    'CANNY_LOW': 30,
    'CANNY_HIGH': 100,      
    'HOUGH_THRESHOLD': 50,   
    'FRAME_PATH': CAMPI_PATH['ERBA'], 
}

PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,
    'CANNY_HIGH': 120,      
    'HOUGH_THRESHOLD': 40,   
    'FRAME_PATH': CAMPI_PATH['TERRA_BATTUTA'],
}

# Mappa per accedere rapidamente
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
}


# --- DIMENSIONI METRICHE PER L'OMOGRAFIA (M3) ---

COURT_DIMENSIONS_METERS = {
    'SINGOLO_LARGHEZZA': 8.23,
    'DOPPIO_LARGHEZZA': 10.97,
    'LUNGHEZZA_TOTALE': 23.77,
    'SERVIZIO_RETE': 6.40,
    'BASE_SERVIZIO': 5.49,
}

# Punti di Riferimento in Metri (World Coordinates) per l'area di servizio vicina
POINTS_WORLD_METERS = np.float32([
    [0.0, 0.0],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], 0.0],
    [0.0, COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
    # ... altri punti non usati direttamente in questa omografia
])
