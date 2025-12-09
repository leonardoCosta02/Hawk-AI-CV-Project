# src/homography_calculator.py
import cv2 as cv
import numpy as np
import math
from . import config # Importa i punti fissi in metri

# ==============================================================================
# 1. FUNZIONE PER TROVARE L'INTERSEZIONE TRA DUE LINEE
# ==============================================================================
def find_intersection(segment1: np.ndarray, segment2: np.ndarray) -> tuple:
    """
    Trova il punto di intersezione tra due segmenti di linea in forma di retta.
    Ritorna (x, y) se c'è un'intersezione, altrimenti (None, None).
    I segmenti sono passati come array [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2

    # Calcola il denominatore D (determinante)
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Se D è zero, le linee sono parallele e non si intersecano.
    if D == 0:
        return None, None

    # Calcola il parametro t
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / D

    # Calcola il punto di intersezione (pixel)
    Px = x1 + t * (x2 - x1)
    Py = y1 + t * (y2 - y1)
    
    return Px, Py

# ==============================================================================
# 2. LOGICA PRINCIPALE PER IL CALCOLO DELL'OMOGRAFIA (M3)
# ==============================================================================
def calculate_homography(all_line_segments: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    """
    Trova un set di punti chiave dall'immagine (pixel) e li mappa ai punti
    corrispondenti nel mondo reale (metri) per calcolare la Matrice H.

    Args:
        all_line_segments: L'output del Membro 1 (array N x 4 di segmenti [x1, y1, x2, y2]).
        surface_type: Tipo di campo (usato per debug o parametri futuri).

    Returns:
        La Matrice di Omografia H (3x3) o None se il calcolo fallisce.
    """
    # Necessitiamo di almeno 4 segmenti per tentare di trovare 4 angoli.
    if all_line_segments.size < 4: 
        print(f"Errore: Output del Membro 1 insufficiente ({all_line_segments.size // 4} segmenti trovati).")
        return None
        
     """
     📐 1. Obiettivo della Fase A: Classificazione GeometricaL'obiettivo di questa fase è prendere l'array di tutti i segmenti trovati da M1
     ($\texttt{all\_line\_segments}$) e dividerli in due gruppi distinti basati sull'angolo:
     Segmenti Orizzontali (Baselines, Service lines): Le linee che attraversano il campo.
     Segmenti Verticali (Sidelines, Center lines): Le linee che corrono per la lunghezza del campo.
     Questo è necessario perché, per trovare l'angolo del campo (ad esempio, l'angolo in basso a sinistra), devi intersecare una linea del gruppo Orizzontale (la linea di fondo) con una linea del gruppo Verticale (la linea laterale)
     
     
     """
    # --- FASE A: SELEZIONE EURISTICA DEI SEGMENTI CHIAVE ---
    
    
    # 1. Calcola l'angolo di ogni segmento
    angles_rad = np.arctan2(all_line_segments[:, 3] - all_line_segments[:, 1], 
                            all_line_segments[:, 2] - all_line_segments[:, 0])
    angles_deg = np.abs(np.degrees(angles_rad) % 180) # Normalizza a [0, 180]

    # 2. Definisci i criteri angolari basati sul parametro di tolleranza
    ANGLE_TOLERANCE = config.HOUGH_COMMON_PARAMS.get('ANGLE_TOLERANCE_DEG', 5)
    
    # Linee Orizzontali (vicino a 0° o 180°)
    is_horizontal = (angles_deg < ANGLE_TOLERANCE) | (angles_deg > 180 - ANGLE_TOLERANCE)
    
    # Linee Verticali (vicino a 90°)
    is_vertical = (angles_deg > 90 - ANGLE_TOLERANCE) & (angles_deg < 90 + ANGLE_TOLERANCE)

    # 3. Filtra i segmenti nei due gruppi
    horizontal_segments = all_line_segments[is_horizontal]
    vertical_segments = all_line_segments[is_vertical]

    if len(horizontal_segments) < 2 or len(vertical_segments) < 2:
        print("Errore: Non sono stati trovati abbastanza segmenti Orizzontali o Verticali (minimo 2 ciascuno).")
        return None

    # --- FASE B: IDENTIFICAZIONE DEI SEGMENTI PIÙ ESTERNI ---
    
    # L'euristica cerca i segmenti che definiscono la metà campo più vicina (i 4 punti di ancoraggio).
    
    # 1. Linea Orizzontale di Fondo (Base Line: la più in basso nel frame)
    h_y_coords = (horizontal_segments[:, 1] + horizontal_segments[:, 3]) / 2
    base_line_index = np.argmax(h_y_coords)
    base_line = horizontal_segments[base_line_index]

    # 2. Linea Verticale Sinistra (Left Line: la più a sinistra nel frame)
    v_x_coords = (vertical_segments[:, 0] + vertical_segments[:, 2]) / 2
    left_line_index = np.argmin(v_x_coords)
    left_line = vertical_segments[left_line_index]

    # 3. Linea Verticale Destra (Right Line: la più a destra nel frame)
    right_line_index = np.argmax(v_x_coords)
    right_line = vertical_segments[right_line_index]
    
    # 4. Linea Orizzontale di Servizio (Service Line: la più in alto tra le restanti orizzontali)
    h_y_coords_filtered = np.delete(h_y_coords, base_line_index)
    h_segments_filtered = np.delete(horizontal_segments, base_line_index, axis=0)
    
    if len(h_segments_filtered) > 0:
        service_line_index = np.argmin(h_y_coords_filtered)
        service_line = h_segments_filtered[service_line_index]
    else:
        print("Meno di 2 linee orizzontali trovate. Il calcolo fallirà.")
        return None


    # --- FASE C: CALCOLO DELLE 4 INTERSEZIONI (PUNTI PIXEL) ---
    
    # 1. Angolo in basso a sinistra (Corrisponde a [0.0, 0.0] metri)
    p1x, p1y = find_intersection(base_line, left_line)
    
    # 2. Angolo in basso a destra (Corrisponde a [Xmax, 0.0] metri)
    p2x, p2y = find_intersection(base_line, right_line)
    
    # 3. Angolo Servizio Sinistra (Corrisponde a [0.0, Yservice] metri)
    p3x, p3y = find_intersection(service_line, left_line)
    
    # 4. Angolo Servizio Destra (Corrisponde a [Xmax, Yservice] metri)
    p4x, p4y = find_intersection(service_line, right_line)
    
    # Controllo di validità
    if None in [p1x, p2x, p3x, p4x]:
        print("Errore: Impossibile trovare tutte le 4 intersezioni chiave (linee parallele o non trovate).")
        return None
        
    # Costruisci l'array dei punti immagine (pixel)
    points_image_pixel = np.float32([
        [p1x, p1y],  
        [p2x, p2y],  
        [p3x, p3y],  
        [p4x, p4y],  
    ])

    # --- FASE D: CALCOLO FINALE OMOGRAFIA ---
    
    # Prende i 4 punti corrispondenti dal mondo reale in config.py
    # L'ordine DEVE corrispondere all'ordine dei punti pixel calcolati sopra.
    points_world_sample = np.float32([
        config.POINTS_WORLD_METERS[0], # 1. Angolo in basso a sinistra (0.0, 0.0)
        config.POINTS_WORLD_METERS[1], # 2. Angolo in basso a destra (Xmax, 0.0)
        config.POINTS_WORLD_METERS[2], # 3. Lato sinistro linea servizio (0.0, Yservice)
        config.POINTS_WORLD_METERS[3]  # 4. Lato destro linea servizio (Xmax, Yservice)
    ])
    
    # Usa cv.RANSAC per tollerare errori nei punti pixel
    H, mask = cv.findHomography(points_image_pixel, points_world_sample, cv.RANSAC, 5.0)

    if H is not None:
        print(f"\nMatrice di Omografia H calcolata con successo.")
    
    return H

# ==============================================================================
# 3. FUNZIONE DI MAPPATURA (Utilità)
# ==============================================================================
def map_pixel_to_world(H: np.ndarray, pixel_coords: tuple) -> np.ndarray:
    """
    Applica la Matrice di Omografia H per mappare un punto pixel (u, v)
    alle coordinate del mondo reale (X, Y) in metri.
    """
    if H is None:
        return None
        
    u, v = pixel_coords
    
    # 1. Trasforma le coordinate pixel in coordinate omogenee (u, v, 1)
    pixel_homogeneous = np.array([u, v, 1], dtype=np.float32)
    
    # 2. Moltiplica H per il vettore di coordinate pixel (H * p)
    world_homogeneous = H @ pixel_homogeneous
    
    # 3. Normalizza (divisione per la terza coordinata 'w')
    X = world_homogeneous[0] / world_homogeneous[2]
    Y = world_homogeneous[1] / world_homogeneous[2]
    
    return np.array([X, Y])
