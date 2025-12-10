# config.py
import numpy as np # Necessario per definire la costante matematica 'np.pi'

# --- PARAMETRI DI HOUGH COMUNI (M1) ---
HOUGH_COMMON_PARAMS = {
    'RHO': 1,
    'THETA': np.pi / 180,
    'MIN_LENGTH': 60,
    'MAX_GAP': 15,
    
    # --- NUOVI PARAMETRI ROI (Percentuali dello schermo) ---
    # Questi filtrano segmenti fuori dal campo (spalti, pubblicità, ecc.)
    'ROI_LEFT_PCT': 0.05,    # Esclude il 5% più a sinistra
    'ROI_RIGHT_PCT': 0.95,   # Esclude il 5% più a destra
    'ROI_TOP_PCT': 0.15,     # Esclude il 15% in alto (spalti lontani)
    'ANGLE_TOLERANCE_DEG': 10, # Usato in court_features per il filtro angolare
}

# --- LISTA DEI PERCORSI DEI FRAME PER IL CARICAMENTO ---
CAMPI_PATH = {
    "CEMENTO": 'data/static_court/static_court_frame_cemento.png',
    "ERBA": 'data/static_court/static_court_frame_erba.png',
    "TERRA_BATTUTA": 'data/static_court/static_court_frame_clay.png',
}

# --- PARAMETRI OTTIMALI CANNY (M1) PER SUPERFICIE (AMMORBIDITI) ---

# -----------------
# Nel tuo file config.py, aggiorna questa sezione:

# -----------------
# CEMENTO: Soglie ulteriormente abbassate.
# -----------------
PARAMS_CEMENTO = {
    'CANNY_LOW': 25,        
    'CANNY_HIGH': 70,       # DA 80 A 70: Molto più facile rilevare i bordi.
    'HOUGH_THRESHOLD': 40,   # DA 50 A 40: Molto facile rilevare una linea.
    'FRAME_PATH': CAMPI_PATH['CEMENTO'],
}

# -----------------
# ERBA: Soglie ulteriormente ammorbidite.
# -----------------
PARAMS_ERBA = {
    'CANNY_LOW': 30,
    'CANNY_HIGH': 100,      # DA 120 A 100: Più facile che un bordo sia considerato 'certo'.
    'HOUGH_THRESHOLD': 50,   # DA 60 A 50: Più facile rilevare.
    'FRAME_PATH': CAMPI_PATH['ERBA'], 
}

# -----------------
# TERRA BATTUTA: Soglie al limite per catturare linee spezzate.
# -----------------
PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,
    'CANNY_HIGH': 120,      # DA 150 A 120: Molto permissivo per i bordi.
    'HOUGH_THRESHOLD': 40,   # DA 50 A 40: Massima facilità per le linee spezzate.
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

# 12 Punti di Riferimento in Metri (World Coordinates)
POINTS_WORLD_METERS = np.float32([
    # ... [LA TUA LISTA DI 12 PUNTI È IMMUTATA E CORRETTA QUI] ...
    [0.0, 0.0],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], 0.0],
    [0.0, COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
    [0.0, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']/2],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']/2],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA']/2, COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA']/2, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE'] - COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
    [0.0, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']],
    [0.0, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE'] - COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE'] - COURT_DIMENSIONS_METERS['SERVIZIO_RETE']],
])
