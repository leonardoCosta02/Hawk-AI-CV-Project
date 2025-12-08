# src/homography_calculator.py
import cv2 as cv
import numpy as np
import math
from . import config # Importa i punti fissi in metri

# --- 1. FUNZIONE PER TROVARE L'INTERSEZIONE TRA DUE LINEE ---
# Basata sulla forma parametrica dei segmenti di linea
def find_intersection(segment1: np.ndarray, segment2: np.ndarray) -> tuple:
    """
    Trova il punto di intersezione tra due segmenti di linea.
    Ritorna (x, y) se c'è un'intersezione, altrimenti (None, None).
    I segmenti sono passati come array [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = segment1
    x3, y3, x4, y4 = segment2

    # Calcola il denominatore D della formula di intersezione
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Se D è zero, le linee sono parallele e non si intersecano.
    if D == 0:
        return None, None

    # Calcola il numeratore
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / D
    # u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / D

    # Calcola il punto di intersezione (pixel)
    Px = x1 + t * (x2 - x1)
    Py = y1 + t * (y2 - y1)
    
    # Opzionale: Aggiungere un controllo per assicurarsi che l'intersezione
    # cada all'interno dei segmenti, ma per le linee del campo usiamo
    # l'intersezione della retta, non solo del segmento.

    return Px, Py

# --- 2. LOGICA PRINCIPALE PER IL CALCOLO DELL'OMOGRAFIA ---

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
    if all_line_segments.size < 4: # Richiede almeno 4 segmenti per iniziare a cercare 4 intersezioni
        print("Errore: Output del Membro 1 insufficiente.")
        return None

    # Vengono selezionati 4 punti per l'esempio (di solito se ne usano di più per RANSAC)
    # Per semplificare, assumiamo che i segmenti puliti del Cemento contengano le 4 linee principali:
    # 2 laterali (quasi verticali) e 2 di fondo (quasi orizzontali).

    # --- FASE A: Ricerca Heuristica dei Punti Immagine (Esempio Semplificato) ---

    # Ordina i segmenti per angolo e posizione per trovare i candidati migliori
    # (Questo è il passo più difficile e di solito richiede l'algoritmo RANSAC)

    # In questo esempio, prenderemo solo i primi 4 segmenti (ASSUMENDO che siano i migliori)
    # e li incroceremo per trovare 4 angoli.
    
    if len(all_line_segments) < 4:
        print("Meno di 4 segmenti disponibili per il calcolo.")
        return None
        
    # Useremo solo 4 coppie di punti per DLT (necessarie 4+)
    # Assumiamo di voler trovare i 4 angoli della metà campo più vicina.
    
    # [Qui andrebbe il codice avanzato RANSAC per selezionare i segmenti corretti]
    # Usiamo un'euristica molto semplice:
    
    # Ad esempio, prendiamo i primi 4 segmenti e calcoliamo l'intersezione tra S1-S3, S2-S4, ecc.
    # Questo è un placeholder e andrebbe migliorato con il tuo output reale!
    
    # In un sistema reale, dovresti trovare:
    # 1. La linea più verticale a sinistra (segmento A)
    # 2. La linea più verticale a destra (segmento B)
    # 3. La linea più orizzontale in basso (segmento C)
    # 4. La linea di servizio orizzontale (segmento D)

    # Per il momento, usiamo i primi 4 segmenti e li incrociamo (risultato NON garantito):
    
    # Si suppone che M3 fornisca un algoritmo per identificare i segmenti chiave (es. i 4 angoli del campo)
    # In questa simulazione, usiamo un placeholder:
    
    points_image_pixel = np.float32([
        [100, 500],  # Angolo 1 (Pixel) - Placeholder
        [800, 500],  # Angolo 2 (Pixel) - Placeholder
        [100, 100],  # Angolo 3 (Pixel) - Placeholder
        [800, 100],  # Angolo 4 (Pixel) - Placeholder
    ])
    
    # --- FASE B: Calcolo dell'Omografia ---

    # Prendi solo i primi 4 punti dal mondo reale che corrispondono ai 4 angoli nell'esempio
    # (Angoli di fondo + Angoli di servizio lato sinistro e destro)
    points_world_sample = config.POINTS_WORLD_METERS[:4]
    
    if points_image_pixel.shape[0] < 4 or points_world_sample.shape[0] < 4:
        print("Errore: Necessari almeno 4 punti corrispondenti per l'omografia.")
        return None

    # cv.findHomography richiede i punti immagine e i punti mondo
    H, mask = cv.findHomography(points_image_pixel, points_world_sample, cv.RANSAC, 5.0)

    # H è la Matrice 3x3 di Omografia
    if H is not None:
        print(f"\nMatrice di Omografia H calcolata con successo.")
    
    # Restituisce la matrice H (3x3)
    return H

# --- ESEMPIO DI UTILIZZO PER MAPPARE UN PUNTO ---
def map_pixel_to_world(H: np.ndarray, pixel_coords: tuple) -> np.ndarray:
    """
    Applica la Matrice di Omografia H per mappare un punto pixel (u, v)
    alle coordinate del mondo reale (X, Y).
    """
    if H is None:
        return None
        
    u, v = pixel_coords
    
    # Trasforma le coordinate pixel in coordinate omogenee (u, v, 1)
    pixel_homogeneous = np.array([u, v, 1], dtype=np.float32)
    
    # Moltiplica H per il vettore di coordinate pixel
    world_homogeneous = H @ pixel_homogeneous
    
    # Normalizza (dividi per la terza coordinata, w)
    X = world_homogeneous[0] / world_homogeneous[2]
    Y = world_homogeneous[1] / world_homogeneous[2]
    
    return np.array([X, Y])

# Esempio per M3: 
# M3, prendendo un punto in pixel, può calcolarne la posizione reale in metri.
