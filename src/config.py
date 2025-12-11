# config.py
import numpy as np 

# --- PARAMETRI DI HOUGH COMUNI (M1) ---
HOUGH_COMMON_PARAMS = {
    'RHO': 1,               
    'THETA': np.pi / 180,   
    'MIN_LENGTH': 60,       # OTTIMIZZATO: Aumentato da 60 a 80 per ridurre il rumore.
    'MAX_GAP': 35,          
    'ANGLE_TOLERANCE_DEG': 180,
}

# --- LISTA DEI PERCORSI DEI FRAME PER IL CARICAMENTO ---
CAMPI_PATH = {
    "CEMENTO": 'data/static_court/static_court_frame_cemento.png',
    "ERBA": 'data/static_court/static_court_frame_erba.png',
    "TERRA_BATTUTA": 'data/static_court/static_court_frame_clay.png',
}


# --- PARAMETRI OTTIMALI CANNY (M1) PER SUPERFICIE (OTTIMIZZATI) ---

# CEMENTO (Target: ridurre le linee da 21)
PARAMS_CEMENTO = {
    'CANNY_LOW': 15,        
    'CANNY_HIGH': 120,      
    'HOUGH_THRESHOLD': 60,   
    'FRAME_PATH': CAMPI_PATH['CEMENTO'],
}

# ERBA (Target: linee deboli, mantenute basse)
PARAMS_ERBA = {
    'CANNY_LOW': 15,
    'CANNY_HIGH': 50,       
    'HOUGH_THRESHOLD': 30,   
    'FRAME_PATH': CAMPI_PATH['ERBA'], 
}

# TERRA_BATTUTA (Target: ridurre drasticamente le 101 linee. Filtro aggressivo)
PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,
    'CANNY_HIGH': 150,      
    'HOUGH_THRESHOLD': 60,   
    'FRAME_PATH': CAMPI_PATH['TERRA_BATTUTA'],
}

# Mappa per accedere rapidamente
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
}
# Nel tuo file config.py, aggiungi questa sezione:

# --- PARAMETRI DI FILTRO DI CENTRALITÀ PER SUPERFICIE ---
# Questi parametri definiscono la Regione di Interesse percentualmente (0.0 a 1.0)
# per eliminare il rumore periferico (spalti, bordi campo).

CENTRALITY_PARAMS = {
    # CEMENTO (Esempio ottimizzato per inquadratura standard)
    'CEMENTO': {
        'Y_MIN_PCT': 0.30, # Più campo lontano
        'Y_MAX_PCT': 0.75, # Più campo vicino
        'X_MIN_PCT': 0.35, # Margini più larghi
        'X_MAX_PCT': 0.70,
    },
    # ERBA (Meno rumore di texture, ma più problemi di prospettiva)
    'ERBA': {
        'Y_MIN_PCT': 0.30,
        'Y_MAX_PCT': 0.84,
        'X_MIN_PCT': 0.25,
        'X_MAX_PCT': 0.75,
    },
    # TERRA_BATTUTA (Il più difficile, potremmo aver bisogno di un range più stretto)
    'TERRA_BATTUTA': {
        'Y_MIN_PCT': 0.30,
        'Y_MAX_PCT': 0.84,
        'X_MIN_PCT': 0.30,
        'X_MAX_PCT': 0.70,
    },
}

# --- DIMENSIONI METRICHE PER L'OMOGRAFIA (M3) ---
# Sezione mantenuta invariata
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
])
