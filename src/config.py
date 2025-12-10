# config.py
import numpy as np # Necessario per definire la costante matematica 'np.pi'

# --- PARAMETRI DI HOUGH COMUNI (M1) ---
HOUGH_COMMON_PARAMS = {
    'RHO': 1,               # Risoluzione della distanza (Rho): 1 pixel.
    'THETA': np.pi / 180,   # Risoluzione dell'angolo (Theta): 1 grado.
    'MIN_LENGTH': 60,       # Lunghezza minima (in pixel) che un segmento deve avere per essere rilevato. 
    'MAX_GAP': 15,          # Distanza massima (in pixel) tra due segmenti per unirli.
    
}

# --- LISTA DEI PERCORSI DEI FRAME PER IL CARICAMENTO ---
CAMPI_PATH = {
    "CEMENTO": 'data/static_court/static_court_frame_cemento.png',
    "ERBA": 'data/static_court/static_court_frame_erba.png',
    "TERRA_BATTUTA": 'data/static_court/static_court_frame_clay.png',
}

# --- PARAMETRI OTTIMALI CANNY (M1) PER SUPERFICIE ---

# -----------------
# CEMENTO: Superficie stabile. Soglie bilanciate.
# -----------------
PARAMS_CEMENTO = {
    'CANNY_LOW': 25,        
    'CANNY_HIGH': 80,       
    'HOUGH_THRESHOLD': 65,   # Mantenuta media, il contrasto è buono.
    'FRAME_PATH': CAMPI_PATH['CEMENTO'],
}

# -----------------
# ERBA: Alta rumorosità da texture. Soglia Hough alzata.
# -----------------
PARAMS_ERBA = {
    'CANNY_LOW': 30,
    'CANNY_HIGH': 220,      
    'HOUGH_THRESHOLD': 90,   # AUMENTATO (Da 75 a 90): Rende Hough più selettiva per eliminare il rumore di sfondo.
    'FRAME_PATH': CAMPI_PATH['ERBA'], 
}

# -----------------
# TERRA BATTUTA: Rumorosità massima e linee spezzate. Soglia Hough molto alta.
# -----------------
PARAMS_TERRA_BATTUTA = {
    'CANNY_LOW': 40,
    'CANNY_HIGH': 240,      
    'HOUGH_THRESHOLD': 95,   # AUMENTATO (Da 60 a 95): Rende Hough molto selettiva per eliminare la texture e il rumore della terra battuta.
    'FRAME_PATH': CAMPI_PATH['TERRA_BATTUTA'],
}

# Mappa per accedere rapidamente
ALL_SURFACE_PARAMS = {
    'CEMENTO': PARAMS_CEMENTO,
    'ERBA': PARAMS_ERBA,
    'TERRA_BATTUTA': PARAMS_TERRA_BATTUTA,
} # Dizionario centrale usato nel loop (Cella 3) per recuperare l'intero set di parametri per una data superficie.


# --- DIMENSIONI METRICHE PER L'OMOGRAFIA (M3) ---

COURT_DIMENSIONS_METERS = {
    'SINGOLO_LARGHEZZA': 8.23,  # Larghezza campo singolo (27 ft)
    'DOPPIO_LARGHEZZA': 10.97, # Larghezza campo doppio (36 ft)
    'LUNGHEZZA_TOTALE': 23.77, # Lunghezza totale del campo (78 ft)
    'SERVIZIO_RETE': 6.40,     # Distanza Rete a Linea di Servizio (21 ft)
    'BASE_SERVIZIO': 5.49,     # Distanza Linea Servizio a Linea di Fondo (18 ft)
}

# Definiamo 12 Punti di Riferimento (Punti di Ancoraggio per l'Omografia, Campo Singolo)
# L'origine (0, 0) è nell'angolo in basso a sinistra della linea di fondo.
POINTS_WORLD_METERS = np.float32([
    # X (Larghezza)        Y (Lunghezza)
    # -----------------------------------------------------------------------
    # 1. ANGOLI LINEA DI FONDO VICINA (Y = 0.0m)
    # -----------------------------------------------------------------------
    [0.0, 0.0],                          # 1. Angolo in basso a sinistra (Origine)
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], 0.0], # 2. Angolo in basso a destra
    
    # -----------------------------------------------------------------------
    # 3. INTERSEZIONI LINEA DI SERVIZIO VICINA (Y = 6.40m)
    # -----------------------------------------------------------------------
    [0.0, COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 3. Lato sinistro linea servizio vicina
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 4. Lato destro linea servizio vicina
    
    # -----------------------------------------------------------------------
    # 5. PUNTI SULLA RETE (Y = 11.885m)
    # -----------------------------------------------------------------------
    [0.0, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']/2], # 5. Lato sinistro della Rete
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']/2], # 6. Lato destro della Rete
    
    # -----------------------------------------------------------------------
    # 7. CENTRI DELLE LINEE DI SERVIZIO (T-JUNCTIONS)
    # -----------------------------------------------------------------------
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA']/2, COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 7. Giunzione T sulla Linea di Servizio più vicina
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA']/2, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE'] - COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 8. Giunzione T sulla Linea di Servizio più lontana
    
    # -----------------------------------------------------------------------
    # 9. ANGOLI LINEA DI FONDO AVVERSARIO (LONTANA)
    # -----------------------------------------------------------------------
    [0.0, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']], # 9. Angolo in alto a sinistra della Linea di Fondo Lontana
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE']], # 10. Angolo in alto a destra della Linea di Fondo Lontana
    
    # -----------------------------------------------------------------------
    # 11. INTERSEZIONI LINEA DI SERVIZIO LONTANA (Y = 17.37m)
    # -----------------------------------------------------------------------
    [0.0, COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE'] - COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 11. Intersezione sinistra tra Linea Laterale e Linea di Servizio lontana
    [COURT_DIMENSIONS_METERS['SINGOLO_LARGHEZZA'], COURT_DIMENSIONS_METERS['LUNGHEZZA_TOTALE'] - COURT_DIMENSIONS_METERS['SERVIZIO_RETE']], # 12. Intersezione destra tra Linea Laterale e Linea di Servizio lontana
])
