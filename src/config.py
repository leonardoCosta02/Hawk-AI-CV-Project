# config.py

# --- PARAMETRI GENERALI DEL PROGETTO ---
# Utili per M2, M3 e M4
VIDEO_FPS = 60           # Fotogrammi al secondo (se usi il dataset di Kaggle)
COURT_WIDTH_METERS = 10.97 # Larghezza del campo da tennis (regola base)
COURT_LENGTH_METERS = 23.77 # Lunghezza del campo
FRAME_WIDTH = 1280       # Larghezza in pixel del frame che usi (ad esempio)
FRAME_HEIGHT = 720       # Altezza in pixel del frame che usi (ad esempio)

# --- PARAMETRI DI HOUGH COMUNI (M1) ---
# Questi possono restare fissi o essere ottimizzati separatamente
HOUGH_COMMON_PARAMS = {
    'RHO': 1,              # Risoluzione della distanza in pixel
    'THETA': np.pi / 180,  # Risoluzione dell'angolo in radianti
    'MIN_LENGTH': 40,      # Lunghezza minima in pixel di un segmento di linea
    'MAX_GAP': 15,         # Distanza massima per unire due segmenti vicini
}

# --- PARAMETRI OTTIMALI CANNY (M1) PER SUPERFICIE ---
# I valori che hai ottimizzato per la robustezza
PARAMS_CEMENTO = {
    'CANNY_LOW': 25,
    'CANNY_HIGH': 100,
    'HOUGH_THRESHOLD': 70, # Più alto per linee lunghe e sicure
    'FRAME_PATH': 'data/static_court_frame_cemento.jpg',
}

PARAMS_ERBA = {
    'CANNY_LOW': 30,
    'CANNY_HIGH': 120,
    'HOUGH_THRESHOLD': 65,
    'FRAME_PATH': 'data/static_court_frame_erba.jpg',
}

PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,
    'CANNY_HIGH': 180,
    'HOUGH_THRESHOLD': 30, # Più basso a causa delle linee spezzate
    'FRAME_PATH': 'data/static_court_frame_clay.jpg',
}

# Mappa per accedere rapidamente
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
}
