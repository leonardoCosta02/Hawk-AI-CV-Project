import cv2 as cv
import numpy as np

def trova_linee(image_data: np.ndarray, canny_low_threshold: int, canny_high_threshold: int, 
                hough_threshold: int, hough_min_length: int, hough_max_gap: int) -> np.ndarray:
    """
    Esegue il preprocessing (Blur) e l'estrazione delle linee (Canny + Hough) da un frame.

    Args:
        image_data: Il frame statico del campo da tennis letto da OpenCV.
        canny_low_threshold: Soglia inferiore per l'Edge Detection di Canny.
        canny_high_threshold: Soglia superiore per l'Edge Detection di Canny.
        hough_threshold: Numero minimo di intersezioni per essere considerata una linea.
        hough_min_length: Lunghezza minima della linea da rilevare.
        hough_max_gap: Distanza massima tra segmenti di linea per essere considerati una singola linea.

    Returns:
        Un array NumPy contenente i segmenti di linea raw in formato [[x1, y1, x2, y2], ...]. 
        L'output sarà usato dal Membro 3.
    """
    if image_data is None:
        return np.array([])
    
    # 1. PREPROCESSING [cite: 14]
    # Conversione in Grayscale
    gray = cv.cvtColor(image_data, cv.COLOR_BGR2GRAY)
    
    # Applicazione di Gaussian Blur per minimizzare il rumore dalla superficie del campo [cite: 14]
    # Kernel size 5x5 è standard.
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 2. EDGE DETECTION (Canny) [cite: 15]
    # Identifica le transizioni ad alto gradiente (le linee bianche)
    edges = cv.Canny(blurred, canny_low_threshold, canny_high_threshold)
    
    # 3. LINE DETECTION (Probabilistic Hough Transform) [cite: 16]
    # Converte i pixel dei bordi in segmenti vettoriali
    raw_lines = cv.HoughLinesP(
        edges,
        rho=1, # Risoluzione della distanza in pixel
        theta=np.pi / 180, # Risoluzione dell'angolo in radianti
        threshold=hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap
    )

    # 4. OUTPUT [cite: 17]
    if raw_lines is not None:
        # Formatta l'output in un array [x1, y1, x2, y2]
        lines_list = raw_lines.reshape(-1, 4)
        return lines_list
    else:
        return np.array([])
