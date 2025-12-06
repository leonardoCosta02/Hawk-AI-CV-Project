import cv2 as cv
import numpy as np
from src import config


# -------------------------------
#  SUPPORT FUNCTIONS
# -------------------------------

def line_angle(x1, y1, x2, y2):
    """Restituisce l'angolo in gradi di un segmento."""
    return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def filter_by_angle(lines, tol=10):
    """Filtra solo linee orizzontali o verticali entro una certa tolleranza."""
    filtered = []
    for x1, y1, x2, y2 in lines:
        a = line_angle(x1, y1, x2, y2)
        if a < tol or abs(a - 180) < tol:   # orizzontali
            filtered.append([x1, y1, x2, y2])
        elif abs(a - 90) < tol:            # verticali
            filtered.append([x1, y1, x2, y2])
    return filtered


def cluster_lines(lines, axis='h', tol=8):
    """
    Raggruppa linee parallele entro una certa distanza.
    axis='h' → cluster orizzontali (simile y)
    axis='v' → cluster verticali   (simile x)
    """
    if not lines:
        return []

    lines = sorted(lines, key=lambda L: L[1] if axis == 'h' else L[0])
    merged = []
    group = [lines[0]]

    for line in lines[1:]:
        if axis == 'h':   # confronto su y
            if abs(line[1] - group[-1][1]) < tol:
                group.append(line)
            else:
                merged.append(np.mean(group, axis=0).astype(int).tolist())
                group = [line]
        else:             # confronto su x
            if abs(line[0] - group[-1][0]) < tol:
                group.append(line)
            else:
                merged.append(np.mean(group, axis=0).astype(int).tolist())
                group = [line]

    merged.append(np.mean(group, axis=0).astype(int).tolist())
    return merged


def extend_line(line):
    """Estende una linea orizzontale o verticale al bounding completo del gruppo."""
    x1, y1, x2, y2 = line

    # Orizzontale
    if abs(y1 - y2) < 5:
        return [min(x1, x2), y1, max(x1, x2), y1]

    # Verticale
    else:
        return [x1, min(y1, y2), x1, max(y1, y2)]


# -------------------------------
#  MAIN FUNCTION
# -------------------------------

def trova_linee(image_data: np.ndarray, surface_type: str = 'CEMENTO') -> np.ndarray:
    """
    Estrae le linee del campo con pulizia avanzata:
      - Filtering angolare
      - Clustering linee parallele
      - Merge segmenti spezzati
      - Filtraggio per lunghezza
    """
    if image_data is None:
        return np.array([])

    params = config.ALL_SURFACE_PARAMS.get(surface_type.upper(), config.PARAMS_CEMENTO)
    hough_common = config.HOUGH_COMMON_PARAMS

    # -------------------------------------
    # 1) PREPROCESSING
    # -------------------------------------
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, params['CANNY_LOW'], params['CANNY_HIGH'])

    # -------------------------------------
    # 2) HOUGH LINES
    # -------------------------------------
    raw = cv.HoughLinesP(
        edges,
        rho=hough_common['RHO'],
        theta=hough_common['THETA'],
        threshold=params['HOUGH_THRESHOLD'],
        minLineLength=hough_common['MIN_LENGTH'],
        maxLineGap=hough_common['MAX_GAP']
    )

    if raw is None:
        return np.array([])

    lines = raw.reshape(-1, 4)

    # -------------------------------------
    # 3) FILTRO ANGOLARE
    # -------------------------------------
    lines = filter_by_angle(lines)

    if not lines:
        return np.array([])

    # -------------------------------------
    # 4) SEPARA ORIZZONTALI E VERTICALI
    # -------------------------------------
    horiz = [L for L in lines if abs(line_angle(*L) - 0) < 10 or abs(line_angle(*L) - 180) < 10]
    vert =  [L for L in lines if abs(line_angle(*L) - 90) < 10]

    # -------------------------------------
    # 5) CLUSTER LINEE PARALLELE
    # -------------------------------------
    horiz = cluster_lines(horiz, axis='h')
    vert  = cluster_lines(vert, axis='v')

    # -------------------------------------
    # 6) MERGE (ESTENSIONE) SEGMENTI
    # -------------------------------------
    horiz = [extend_line(L) for L in horiz]
    vert  = [extend_line(L) for L in vert]

    # -------------------------------------
    # 7) FILTRO LUNGHEZZA MINIMA
    # -------------------------------------
    #h, w = image_data.shape[:2]
   # min_len = w * 0.25

    final = []
    for x1, y1, x2, y2 in horiz + vert:
        if np.hypot(x2 - x1, y2 - y1) >= min_len:
            final.append([x1, y1, x2, y2])

    return np.array(final)
