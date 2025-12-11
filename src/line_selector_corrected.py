# line_selector_corrected.py (Nuovo File - Simula la correzione di M2)

import numpy as np

def select_court_lines_corrected(segments):
    """
    Seleziona le 4 linee essenziali (Base, Servizio, Sinistra, Destra) 
    dall'insieme di segmenti consolidati.
    
    Si basa sulla posizione centrale Y per H e sulla posizione centrale X per V.
    
    Risolve il problema dello scambio di ruoli V/H e l'eccessiva rigidità.
    """
    
    if len(segments) < 4:
        raise ValueError("Sono necessari almeno 4 segmenti per l'omografia.")

    # 1. Calcola centri e angoli (come in court_features, ma per riutilizzarli)
    x_center = (segments[:,0] + segments[:,2]) / 2
    y_center = (segments[:,1] + segments[:,3]) / 2
    
    dx = segments[:,2] - segments[:,0]
    dy = segments[:,3] - segments[:,1]
    angles = np.abs(np.degrees(np.arctan2(dy, dx)) % 180)

    # 2. SEPARAZIONE in Orizzontali (H) e Verticali (V)
    # Rilassiamo il filtro per includere linee laterali prospettiche.
    # Usiamo un approccio che favorisce la H se l'angolo è vicino a 0/180.
    
    # Un segmento è Orizzontale (H) se l'angolo è < 30 gradi da 0 o 180
    is_h = (angles < 30) | (angles > 150)
    # I Verticali (V) sono tutti gli altri, inclusi i segmenti prospettici
    is_v = ~is_h 
    
    horiz_segments = segments[is_h]
    vert_segments = segments[is_v]

    # CONTROLLO CRITICO DEL CEMENTO
    if len(horiz_segments) < 2 or len(vert_segments) < 2:
        print("ATTENZIONE: M2 non ha trovato 2 H e 2 V distinti.")
        print(f"H trovate: {len(horiz_segments)}, V trovate: {len(vert_segments)}")
        # Se fallisce qui, potremmo provare a rilassare ulteriormente gli angoli 
        # o usare la posizione centrale (y) come discriminante principale.
        # Per ora, simuliamo il fallimento come nel tuo output per CEMENTO.
        # Se questo è il problema del CEMENTO, devi ottimizzare gli angoli nel tuo M2.
        return None, None, None, None


    # 3. SELEZIONE LINEE ORIZZONTALI (Base e Servizio)
    # Le linee orizzontali sono ordinate in base alla loro posizione Y centrale (dall'alto al basso)
    y_centers_h = (horiz_segments[:,1] + horiz_segments[:,3]) / 2
    sorted_h_indices = np.argsort(y_centers_h)
    
    # La Base Line è la linea orizzontale più in basso (Y più grande)
    base_line = horiz_segments[sorted_h_indices[-1]]
    # La Service Line è la seconda linea orizzontale più in basso
    service_line = horiz_segments[sorted_h_indices[-2]]
    
    # 4. SELEZIONE LINEE VERTICALI (Laterali)
    # Le linee verticali sono ordinate in base alla loro posizione X centrale (da sinistra a destra)
    x_centers_v = (vert_segments[:,0] + vert_segments[:,2]) / 2
    sorted_v_indices = np.argsort(x_centers_v)
    
    # La Linea Laterale Sinistra è la più a sinistra (X più piccolo)
    side_left = vert_segments[sorted_v_indices[0]]
    # La Linea Laterale Destra è la più a destra (X più grande)
    side_right = vert_segments[sorted_v_indices[-1]]
    
    
    # --- Controllo di Sanità (per evitare lo scambio di ruoli in Erba/Terra) ---
    # La Linea di Base (più in basso) deve avere un Y > della Linea di Servizio.
    if np.mean([base_line[1], base_line[3]]) < np.mean([service_line[1], service_line[3]]):
        # Scambia i ruoli se l'ordinamento è invertito (questo è il motivo per cui è stato usato [-1] e [-2] sopra)
        # Se fallisce qui, significa che Base e Servizio sono state erroneamente identificate.
        pass # La logica [-1] e [-2] dovrebbe evitarlo, ma è un buon punto di debug.

    return base_line, service_line, side_left, side_right
