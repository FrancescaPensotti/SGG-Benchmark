#!/usr/bin/env python3
# genera_charuco.py
# ─────────────────────────────────────────────────────────────
# Genera una board ChArUco pronta da stampare per la hand-eye
# calibration con moveit2_calibration.
#
# Uso:
#   python3 genera_charuco.py
#
# Produce due file nella cartella corrente:
#   - charuco_board.png  (immagine ad alta risoluzione)
#   - charuco_board.pdf  (formato A4, da stampare "dimensione reale / 100%")
#
# IMPORTANTE: in stampa, scegli "Dimensioni effettive" / "100%" / "Nessun
# adattamento", altrimenti le misure cambiano. Dopo la stampa, MISURA col
# calibro un quadrato e un marker e usa quei valori reali nel pannello RViz.
# ─────────────────────────────────────────────────────────────

import cv2
import cv2.aruco as aruco
import numpy as np

# ── PARAMETRI DELLA BOARD (servono nel pannello RViz) ──
SQUARES_X      = 5            # numero di quadrati in orizzontale
SQUARES_Y      = 7            # numero di quadrati in verticale
SQUARE_MM      = 30.0         # lato di un quadrato della scacchiera (mm)
MARKER_MM      = 22.0         # lato di un marker ArUco (mm), deve essere < SQUARE_MM
DICTIONARY     = aruco.DICT_5X5_250   # dizionario ArUco
DPI            = 300         # risoluzione di stampa
MARGIN_MM      = 10.0         # bordo bianco attorno alla board (mm)

# ── Conversioni ──────────────────────────────────────────────
MM_PER_INCH = 25.4
def mm_to_px(mm):
    return int(round(mm / MM_PER_INCH * DPI))

square_m = SQUARE_MM / 1000.0
marker_m = MARKER_MM / 1000.0
board_w_px = mm_to_px(SQUARES_X * SQUARE_MM)
board_h_px = mm_to_px(SQUARES_Y * SQUARE_MM)
margin_px  = mm_to_px(MARGIN_MM)

dictionary = aruco.getPredefinedDictionary(DICTIONARY)

# ── Creazione board (gestisce sia OpenCV >=4.7 sia versioni vecchie) ──
try:
    # API nuova (OpenCV >= 4.7)
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_m, marker_m, dictionary)
    img = board.generateImage((board_w_px, board_h_px), marginSize=margin_px, borderBits=1)
except AttributeError:
    # API vecchia (OpenCV < 4.7)
    board = aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, square_m, marker_m, dictionary)
    img = board.draw((board_w_px, board_h_px), marginSize=margin_px, borderBits=1)

# ── Salva PNG ────────────────────────────────────────────────
cv2.imwrite("charuco_board.png", img)

# ── Salva PDF A4 a dimensione reale ──────────────────────────
# A4 = 210 x 297 mm. Centriamo la board nella pagina.
A4_W_MM, A4_H_MM = 210.0, 297.0
a4_w_px, a4_h_px = mm_to_px(A4_W_MM), mm_to_px(A4_H_MM)
page = np.full((a4_h_px, a4_w_px), 255, dtype=np.uint8)  # pagina bianca
y0 = (a4_h_px - img.shape[0]) // 2
x0 = (a4_w_px - img.shape[1]) // 2
if y0 >= 0 and x0 >= 0:
    page[y0:y0+img.shape[0], x0:x0+img.shape[1]] = img
else:
    page = img  # board piu' grande di A4: salva senza centrare

# Scrive il PDF impostando il DPI cosi' la stampa a 100% rispetta le misure
try:
    from PIL import Image
    pil = Image.fromarray(page)
    pil.save("charuco_board.pdf", "PDF", resolution=DPI)
except ImportError:
    cv2.imwrite("charuco_board_A4.png", page)
    print("PIL non disponibile: salvato charuco_board_A4.png invece del PDF.")

# ── Riepilogo parametri (da annotare per il pannello RViz) ──
print("="*55)
print("BOARD CHARUCO GENERATA")
print("="*55)
print(f"  Quadrati X (squares_x):      {SQUARES_X}")
print(f"  Quadrati Y (squares_y):      {SQUARES_Y}")
print(f"  Lato quadrato (nominale):    {SQUARE_MM} mm  = {square_m} m")
print(f"  Lato marker (nominale):      {MARKER_MM} mm  = {marker_m} m")
print(f"  Dizionario:                  DICT_5X5_250")
print(f"  DPI di stampa:               {DPI}")
print("="*55)
print("File creati: charuco_board.png  e  charuco_board.pdf")
print("\nSTAMPA: usa 'Dimensione reale / 100% / Nessun adattamento'.")
print("DOPO LA STAMPA: misura col calibro un quadrato e un marker,")
print("e inserisci i valori REALI (in metri) nel pannello RViz.")
print("="*55)