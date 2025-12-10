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
    """
    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2

    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if D == 0:
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
        print(f"Errore: Output del Membro 1 insufficiente ({all_line_segments.size // 4} segmenti trovati).")
        return None, None
        
    # --- FASE A: SELEZIONE EURISTICA DEI SEGMENTI CHIAVE (Angolo) ---
    
    angles_rad = np.arctan2(all_line_segments[:, 3] - all_line_segments[:, 1], 
                            all_line_segments[:, 2] - all_line_segments[:, 0])
    angles_deg = np.abs(np.degrees(angles_rad) % 180)

    ANGLE_TOLERANCE = config.HOUGH_COMMON_PARAMS.get('ANGLE_TOLERANCE_DEG', 5)
    
    is_horizontal = (angles_deg < ANGLE_TOLERANCE) | (angles_deg > 180 - ANGLE_TOLERANCE)
    is_vertical = (angles_deg > 90 - ANGLE_TOLERANCE) & (angles_deg < 90 + ANGLE_TOLERANCE)

    horizontal_segments = all_line_segments[is_horizontal]
    vertical_segments = all_line_segments[is_vertical]

    if len(horizontal_segments) < 2 or len(vertical_segments) < 2:
        print("Errore: Non sono stati trovati abbastanza segmenti Orizzontali o Verticali (minimo 2 ciascuno).")
        return None, None

    # --- FASE B: IDENTIFICAZIONE DEI SEGMENTI PIÙ ESTERNI (POSIZIONE) ---
    
    # 1. Linea Orizzontale di Fondo (Base Line: la più in basso nel frame, Y max)
    h_y_coords = (horizontal_segments[:, 1] + horizontal_segments[:, 3]) / 2 
    base_line_index = np.argmax(h_y_coords) 
    base_line = horizontal_segments[base_line_index] 
    
    # 2. Linea Verticale Sinistra (Left Line: la più a sinistra nel frame, X min)
    v_x_coords = (vertical_segments[:, 0] + vertical_segments[:, 2]) / 2 
    left_line_index = np.argmin(v_x_coords) 
    left_line = vertical_segments[left_line_index] 
    
    # 3. Linea Verticale Destra (Right Line: la più a destra nel frame, X max)
    right_line_index = np.argmax(v_x_coords) 
    right_line = vertical_segments[right_line_index] 
    
    # 4. Linea Orizzontale di Servizio (Service Line: la seconda più in basso)
    
    temp_mask = np.ones(len(horizontal_segments), dtype=bool)
    temp_mask[base_line_index] = False
    
    h_y_coords_filtered = h_y_coords[temp_mask]
    h_segments_filtered = horizontal_segments[temp_mask]
    
    if len(h_segments_filtered) > 0:
        # Trova il MAX Y rimanente (la seconda linea più in basso)
        second_closest_line_index = np.argmax(h_y_coords_filtered) 
        service_line = h_segments_filtered[second_closest_line_index]
    else:
        print("Errore: Non sono state trovate almeno due linee orizzontali chiave.")
        return None, None

    # ** CORREZIONE: DEFINIZIONE DI selected_segments (RISOLVE IL NAMETERROR) **
    selected_segments = np.array([base_line, left_line, right_line, service_line], dtype=np.int32)


    # --- FASE C: CALCOLO DELLE 4 INTERSEZIONI (PUNTI PIXEL) ---
    
    p1x, p1y = find_intersection(base_line, left_line)
    p2x, p2y = find_intersection(base_line, right_line)
    p3x, p3y = find_intersection(service_line, left_line)
    p4x, p4y = find_intersection(service_line, right_line)
    
    if None in [p1x, p2x, p3x, p4x]:
        print("Errore: Impossibile trovare tutte le 4 intersezioni chiave.")
        return None, None
    
    points_image_pixel = np.float32([
        [p1x, p1y],  # Angolo 1: Base Line Sinistra
        [p2x, p2y],  # Angolo 2: Base Line Destra
        [p3x, p3y],  # Angolo 3: Service Line Sinistra
        [p4x, p4y],  # Angolo 4: Service Line Destra
    ])

    # --- FASE D: CALCOLO FINALE OMOGRAFIA ---
    
    points_world_sample = np.float32([
        config.POINTS_WORLD_METERS[0], 
        config.POINTS_WORLD_METERS[1], 
        config.POINTS_WORLD_METERS[2], 
        config.POINTS_WORLD_METERS[3]  
    ])
    
    H, mask = cv.findHomography(points_image_pixel, points_world_sample, cv.RANSAC, 5.0)
    
    if H is not None:
        print(f"\nMatrice di Omografia H calcolata con successo per {surface_type}.")
        return H, selected_segments
    else:
        return None, None


# ==============================================================================
# 3. FUNZIONE DI MAPPATURA (Utilità)
# ==============================================================================
def map_pixel_to_world(H: np.ndarray, pixel_coords: tuple) -> np.ndarray:
    
    if H is None:
        return None
        
    u, v = pixel_coords
    
    pixel_homogeneous = np.array([u, v, 1], dtype=np.float32)
    world_homogeneous = H @ pixel_homogeneous
    
    X = world_homogeneous[0] / world_homogeneous[2]
    Y = world_homogeneous[1] / world_homogeneous[2]
    
    return np.array([X, Y])
