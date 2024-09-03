import os
import cv2
from PIL import Image

adresar = "./Images/"

output_filename = "detekovano.avi"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 5  
video_size = None

obrazky = []

for soubor in sorted(os.listdir(adresar)):
    if soubor.endswith(('.jpg', )):
        cesta_k_obrazku = os.path.join(adresar, soubor)
        obrazek = Image.open(cesta_k_obrazku)
        if video_size is None:
            video_size = obrazek.size
        obrazky.append(cesta_k_obrazku)

video_writer = cv2.VideoWriter(output_filename, fourcc, fps, video_size)

for obrazek_path in obrazky:
    obrazek = cv2.imread(obrazek_path)
    video_writer.write(obrazek)

video_writer.release()


