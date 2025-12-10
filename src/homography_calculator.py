# src/homography_calculator.py
import cv2 as cv
import numpy as np
import math
from . import config 

# Definizioni ANSI per l'output in rosso nel terminale/notebook
RED = "\033[91m"
ENDC = "\033[0m"

# ==============================================================================
# 1. FUNZIONE PER TROVARE L'INTERSEZIONE TRA DUE LINEE
# ==============================================================================
def find_intersection(segment1: np.ndarray, segment2: np.ndarray) -> tuple:
    """
    Trova il punto di intersezione tra due segmenti di linea in forma di retta.
    AGGIUNTA DI ROBUSTEZZA: Controllo per linee quasi parallele (D ~ 0).
    """
    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    epsilon = 1e-4 # Soglia per linee quasi parallele

    if abs(D) < epsilon: # Linee parallele o quasi parallele
        return None, None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / D

    Px = x1 + t * (x2 - x1)
    Py = y1 + t * (y2 - y1)
    
    return Px, Py

# ==============================================================================
# 2. LOGICA PRINCIPALE PER IL CALCOLO DELL'OMOGRAFIA (M3)
# ==============================================================================
def calculate_homography(all_line_segments: np.ndarray, surface_type: str = 'CEMENTO') -> tuple:
    
    if all_line_segments.size < 4: 
        print(f"Errore: Output del Modulo M1 insufficiente per calcolare l'omografia ({all_line_segments.size} segmenti trovati).")
        return None, None
    
    # --- FASE A: Filtraggio Linee e Raggruppamento (Separazione H/V PERMISSIVA) ---
    
    dx = all_line_segments[:, 2] - all_line_segments[:, 0]
    dy = all_line_segments[:, 3] - all_line_segments[:, 1]
    angles = np.degrees(np.arctan2(dy, dx)) % 180

    # USIAMO UNA TOLLERANZA MOLTO ALTA (35 gradi) per SEPARARE i gruppi H e V
    # Questo indirizza il problema della prospettiva estrema.
    separation_tol = 35 
    
    # Segmenti Orizzontali (Angoli vicini a 0°/180°)
    is_horizontal = (angles < separation_tol) | (angles > 180 - separation_tol)
    horizontal_segments = all_line_segments[is_horizontal]
    
    # Segmenti Verticali (Angoli vicini a 90°)
    is_vertical = (angles > 90 - separation_tol) & (angles < 90 + separation_tol)
    vertical_segments = all_line_segments[is_vertical]

    if len(horizontal_segments) < 2 or len(vertical_segments) < 2:
        print(f"{RED}Errore: Trovate solo {len(horizontal_segments)} linee orizzontali e {len(vertical_segments)} linee verticali. Servono almeno 2 H e 2 V.{ENDC}")
        return None, None

    # --- FASE B: Selezione Linee Chiave (Euristica basata su Centroidi Y) ---
    # Questa fase è cruciale e si basa sul presupposto che il Modulo M1 (trova_linee)
    # abbia già rimosso il rumore di fondo indesiderato tramite i filtri di centralità Y/X.
    
    try:
        # Ordina orizzontali per coordinata Y media decrescente (dal basso verso l'alto nell'immagine)
        h_y_center = (horizontal_segments[:, 1] + horizontal_segments[:, 3]) / 2
        h_sorted_indices = np.argsort(h_y_center)[::-1]
        
        # Linea Base (la più in basso)
        base_line = horizontal_segments[h_sorted_indices[0]] 
        
        # Linea Servizio (la seconda in basso)
        service_line = horizontal_segments[h_sorted_indices[1]] 
        
        # Ordina verticali per coordinata X media (da sinistra a destra)
        v_x_center = (vertical_segments[:, 0] + vertical_segments[:, 2]) / 2
        v_sorted_indices = np.argsort(v_x_center)
        
        # Linea Laterale Sinistra (la più a sinistra)
        side_line_left = vertical_segments[v_sorted_indices[0]]
        
        # Linea Laterale Destra (la più a destra)
        side_line_right = vertical_segments[v_sorted_indices[-1]] 
        
        selected_segments = np.array([base_line, service_line, side_line_left, side_line_right])
        
    except IndexError:
        print(f"{RED}Errore: Impossibile trovare le 4 linee chiave (non abbastanza segmenti validi).{ENDC}")
        return None, None
        
    # --- STAMPA DELLE LINEE RILEVATE (ROSSO) ---
    print(f"{RED}--- LINEE CHIAVE RILEVATE (M3 - TEMPLATE FITTING CANDIDATO) ---{ENDC}")
    print(f"{RED}Base Line (H): {base_line}{ENDC}")
    print(f"{RED}Service Line (H): {service_line}{ENDC}")
    print(f"{RED}Side Line Left (V): {side_line_left}{ENDC}")
    print(f"{RED}Side Line Right (V): {side_line_right}{ENDC}")
    

    # --- FASE C: Calcolo delle intersezioni (4 Punti Pixel) ---
    
    # I punti pixel devono corrispondere all'ordine dei POINTS_WORLD_METERS:
    # P1: (0.0, 0.0) -> Base Line e Sinistra (x=0.0)
    p1x, p1y = find_intersection(base_line, side_line_left)
    
    # P2: (LARGHEZZA, 0.0) -> Base Line e Destra 
    p2x, p2y = find_intersection(base_line, side_line_right)

    # P3: (0.0, SERVIZIO_RETE) -> Service Line e Sinistra
    p3x, p3y = find_intersection(service_line, side_line_left)

    # P4: (LARGHEZZA, SERVIZIO_RETE) -> Service Line e Destra
    p4x, p4y = find_intersection(service_line, side_line_right)
    
    # Il controllo sulle intersezioni quasi parallele è già in find_intersection()
    if None in [p1x, p2x, p3x, p4x]:
        print(f"{RED}Errore: Impossibile trovare tutte le 4 intersezioni chiave (le linee selezionate sono quasi parallele).{ENDC}")
        return None, None
    
    points_image_pixel = np.float32([
        [p1x, p1y],  
        [p2x, p2y],  
        [p3x, p3y],  
        [p4x, p4y],  
    ])
    
    # --- STAMPA PUNTI PIXEL USATI (ROSSO) ---
    print(f"\n{RED}Punti Pixel usati per Omografia (corrispondenti a POINTS_WORLD_METERS):{ENDC}")
    print(points_image_pixel)

    # --- FASE D: CALCOLO FINALE OMOGRAFIA ---
    
    points_world_sample = config.POINTS_WORLD_METERS
    
    # Usa RANSAC per robustezza contro outlier
    H, mask = cv.findHomography(points_image_pixel, points_world_sample, cv.RANSAC, 5.0)
    
    if H is not None:
        # --- STAMPA MATRICE H (ROSSO) ---
        print(f"\n{RED}✅ Matrice H calcolata con successo per {surface_type}.{ENDC}")
        print(f"{RED}Matrice H (3x3):{ENDC}")
        print(H)
        return H, selected_segments
    else:
        print(f"{RED}❌ Errore: cv.findHomography ha fallito (probabilmente punti pixel mal allineati).{ENDC}")
        return None, None


# ==============================================================================
# 3. FUNZIONE DI MAPPATURA (Utilità)
# ==============================================================================
def map_pixel_to_world(H: np.ndarray, pixel_coords: tuple) -> np.ndarray:
    """
    Mappa un punto pixel (u, v) a coordinate reali (X, Y) usando la Matrice H.
    """
    if H is None:
        return np.array([0.0, 0.0])
    
    point_homog = np.array([pixel_coords[0], pixel_coords[1], 1])
    world_homog = H @ point_homog
    
    # Normalizza
    X = world_homog[0] / world_homog[2]
    Y = world_homog[1] / world_homog[2]
    
    return np.array([X, Y])
