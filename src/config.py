# config.py
import numpy as np # Necessario per definire la costante matematica 'np.pi'

# --- PARAMETRI DI HOUGH COMUNI (M1) ---
# Questi parametri sono geometrici e sono spesso validi per tutte le superfici,
# a meno che le linee non siano molto spezzate.
HOUGH_COMMON_PARAMS = {
    'RHO': 1,               # Risoluzione della distanza (Rho): 1 pixel. Non si tocca quasi mai.
    'THETA': np.pi / 180,   # Risoluzione dell'angolo (Theta): 1 grado (pi/180). Non si tocca mai.
    'MIN_LENGTH': 60,       # Lunghezza minima (in pixel) che un segmento deve avere per essere rilevato. 
                            # Alto per ignorare frammenti di rumore.
    'MAX_GAP': 15,          # Distanza massima (in pixel) tra due segmenti per unirli in un'unica linea. 
                            # Utile per ricongiungere linee spezzate dalle ombre o dalla terra battuta.
}

# --- LISTA DEI PERCORSI DEI FRAME PER IL CARICAMENTO ---
# Variabile usata nella Cella 2 del notebook per il loop di caricamento delle immagini statiche.
CAMPI_PATH = {
    "CEMENTO": 'data/static_images/static_court_frame_cemento.jpg',
    "ERBA": 'data/static_images/static_court_frame_erba.jpg',
    "TERRA_BATTUTA": 'data/static_images/static_court_frame_clay.jpg',
}

# --- PARAMETRI OTTIMALI CANNY (M1) PER SUPERFICIE ---

# -----------------
# CEMENTO: Superficie a bassa rumorosità e contrasto elevato.
# -----------------
PARAMS_CEMENTO = {
    'CANNY_LOW': 25,         # Soglia Bassa Canny: Bassa per catturare bordi deboli che sono connessi.
    'CANNY_HIGH': 80,       # Soglia Alta Canny: Bassa/Media, sufficiente per definire le linee bianche come 'bordi certi'.
    'HOUGH_THRESHOLD': 65,   # Soglia di Votazione Hough: Alta. Richiediamo molti pixel per linea, perché il contrasto è buono.
    'FRAME_PATH': CAMPI_PATH['CEMENTO'], # Percorso specifico del frame.
}

# -----------------
# ERBA: Superficie ad alta rumorosità/texture (fili d'erba) e contrasto variabile.
# -----------------
PARAMS_ERBA = {
    'CANNY_LOW': 30,         # Soglia Bassa: Leggermente più alta del cemento, per ignorare la granulosità di base.
    'CANNY_HIGH': 120,       # Soglia Alta: Più alta (120) per ignorare l'intera texture dell'erba e accettare solo le linee bianche.
    'HOUGH_THRESHOLD': 65,   # Soglia di Votazione Hough: Media/Alta. Simile al cemento, richiede linee ben definite.
    'FRAME_PATH': CAMPI_PATH['ERBA'], 
}

# -----------------
# TERRA BATTUTA: Superficie con la più alta rumorosità/texture e probabilità di linee spezzate.
# -----------------
PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,         # Soglia Bassa: La più alta (40) per filtrare il rumore evidente della terra battuta.
    'CANNY_HIGH': 180,       # Soglia Alta: Molto alta (180). Solo i salti di intensità fortissimi (le linee bianche) sono considerati 'certi'.
    'HOUGH_THRESHOLD': 30,   # Soglia di Votazione Hough: Bassa. Necessaria perché le linee sono spezzate/usurate; un segmento reale non ha molti pixel continui che votano.
    'FRAME_PATH': CAMPI_PATH['TERRA_BATTUTA'],
}

# Mappa per accedere rapidamente
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
} # Dizionario centrale usato nel loop (Cella 3) per recuperare l'intero set di parametri per una data superficie.
