import os
import cv2
from PIL import Image

adresar = "./assets/"

output_filename = "detekovano.avi"

# Nastavení video kodeku a parametry výstupního videa
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 5  # Počet snímků za sekundu
video_size = None

# Seznam pro uložení všech obrázků
obrazky = []

# Načtení všech obrázků z adresáře
for soubor in sorted(os.listdir(adresar)):
    if soubor.endswith(('.jpg', )):
        cesta_k_obrazku = os.path.join(adresar, soubor)
        obrazek = Image.open(cesta_k_obrazku)
        if video_size is None:
            video_size = obrazek.size
        obrazky.append(cesta_k_obrazku)

# Vytvoření video spisovače
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, video_size)

# Přidání obrázků do videa
for obrazek_path in obrazky:
    obrazek = cv2.imread(obrazek_path)
    video_writer.write(obrazek)

# Uvolnění video spisovače
video_writer.release()


