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
    'CANNY_HIGH': 220,       # Soglia Alta: Più alta (120) per ignorare l'intera texture dell'erba e accettare solo le linee bianche.
    'HOUGH_THRESHOLD': 75,   # Soglia di Votazione Hough: Media/Alta. Simile al cemento, richiede linee ben definite.
    'FRAME_PATH': CAMPI_PATH['ERBA'], 
}

# -----------------
# TERRA BATTUTA: Superficie con la più alta rumorosità/texture e probabilità di linee spezzate.
# -----------------
PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,         # Soglia Bassa: La più alta (40) per filtrare il rumore evidente della terra battuta.
    'CANNY_HIGH': 240,       # Soglia Alta: Molto alta (180). Solo i salti di intensità fortissimi (le linee bianche) sono considerati 'certi'.
    'HOUGH_THRESHOLD': 60,   # Soglia di Votazione Hough: Bassa. Necessaria perché le linee sono spezzate/usurate; un segmento reale non ha molti pixel continui che votano.
    'FRAME_PATH': CAMPI_PATH['TERRA_BATTUTA'],
}

# Mappa per accedere rapidamente
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
} # Dizionario centrale usato nel loop (Cella 3) per recuperare l'intero set di parametri per una data superficie.


# Aggiungi questa nuova sezione in config.py
# (Dimensioni standard del campo da tennis in metri)

COURT_DIMENSIONS_METERS = {
    'SINGOLO_LARGHEZZA': 8.23,  # Larghezza campo singolo (27 ft)
    'DOPPIO_LARGHEZZA': 10.97, # Larghezza campo doppio (36 ft)
    'LUNGHEZZA_TOTALE': 23.77, # Lunghezza totale del campo (78 ft)
    'SERVIZIO_RETE': 6.40,     # Distanza Rete a Linea di Servizio (21 ft)
    'BASE_SERVIZIO': 5.49,     # Distanza Linea Servizio a Linea di Fondo (18 ft)
}

# Definiamo 8 Punti di Riferimento (Corners of the court + T-junctions)
# Usiamo il campo doppio (10.97 x 23.77m)
# Assumiamo l'origine (0, 0) nell'angolo in basso a sinistra della linea di fondo.
#POINTS_WORLD_METERS è: È una matrice con 8 righe E 2 colonne Ogni riga rappresenta un punto del campo da tennis Ogni punto è espresso in coordinate metriche (X, Y)
POINTS_WORLD_METERS = np.float32([
    # X (Larghezza)        Y (Lunghezza)
    # Angoli linea di fondo (Base-line corners)
    [0.0, 0.0],                          # 1. Angolo in basso a sinistra (X0, Y0)
    [COURT_DIMENSIONS_METERS['DOPPIO_LARGHEZZA'], 0.0], # 2. Angolo in basso a destra (Xmax, Y0)
    
    # Intersezioni Linea di Servizio (Service-line intersections)
    [0.0, COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 3. Lato sinistro linea servizio
    [COURT_DIMENSIONS_METERS['DOPPIO_LARGHEZZA'], COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 4. Lato destro linea servizio
    
    # Rete
    [0.0, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']/2], # 5. Lato sinistro della Rete
    [COURT_DIMENSIONS_METERS['DOPPIO_LARGHEZZA'], COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']/2], # 6. Lato destro della Rete
    
    # Centro della Linea di Servizio (T-junctions)
    [COURT_DIMENSIONS_METERS['DOPPIO_LARGHEZZA']/2, COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 7. Linea di Servizio T-junction (inferiore)
    [COURT_DIMENSIONS_METERS['DOPPIO_LARGHEZZA']/2, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE'] - COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 8. Linea di Servizio T-junction (superiore)
])
