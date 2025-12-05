# config.py
import numpy as np

# --- PARAMETRI GENERALI DEL PROGETTO ---
# Utili per M2, M3 e M4
VIDEO_FPS = 60           
COURT_WIDTH_METERS = 10.97 
COURT_LENGTH_METERS = 23.77 
FRAME_WIDTH = 1280       
FRAME_HEIGHT = 720       

# --- PARAMETRI DI HOUGH COMUNI (M1) ---
# Questi possono restare fissi o essere ottimizzati separatamente
HOUGH_COMMON_PARAMS = {
    'RHO': 1,              
    'THETA': np.pi / 180,  
    'MIN_LENGTH': 40,      
    'MAX_GAP': 15,         
}

# --- LISTA DEI PERCORSI DEI FRAME PER IL CARICAMENTO (NUOVA VARIABILE) ---
# Usa questi percorsi per caricare le immagini statiche in modo pulito
CAMPI_PATH = {
    "CEMENTO": 'data/static_images/static_court_frame_cemento.jpg',
    "ERBA": 'data/static_images/static_court_frame_erba.jpg',
    "TERRA_BATTUTA": 'data/static_images/static_court_frame_clay.jpg',
}

# --- PARAMETRI OTTIMALI CANNY (M1) PER SUPERFICIE ---
# I valori che hai ottimizzato per la robustezza
PARAMS_CEMENTO = {
    'CANNY_LOW': 25,
    'CANNY_HIGH': 100,
    'HOUGH_THRESHOLD': 70, 
    'FRAME_PATH': CAMPI_PATH['CEMENTO'], # Usa la variabile definita sopra
}

PARAMS_ERBA = {
    'CANNY_LOW': 30,
    'CANNY_HIGH': 120,
    'HOUGH_THRESHOLD': 65,
    'FRAME_PATH': CAMPI_PATH['ERBA'], # Usa la variabile definita sopra
}

PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,
    'CANNY_HIGH': 180,
    'HOUGH_THRESHOLD': 30, 
    'FRAME_PATH': CAMPI_PATH['TERRA_BATTUTA'], # Usa la variabile definita sopra
}

# Mappa per accedere rapidamente
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
}
