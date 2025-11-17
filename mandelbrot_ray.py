# mandelbrot_ray.py
import ray
import numpy as np
import matplotlib.pyplot as plt
import time
from math import sqrt

# ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=6
# ray start --address='ip:6379' --num-cpus=6


# --- Parametry obrazu ---
WIDTH = 4000
HEIGHT = 4000
MAX_IT = 1000
RE_MIN, RE_MAX = -2.0, 1.0
IM_MIN, IM_MAX = -1.5, 1.5
TILE_ROWS = 10   # podział w pionie
TILE_COLS = 10   # podział w poziomie


@ray.remote
def compute_tile(x0, x1, y0, y1, width, height, max_it,
                 re_min, re_max, im_min, im_max):
    w = x1 - x0
    h = y1 - y0
    tile = np.zeros((h, w), dtype=np.uint16)

    print(f"Start tile: x=({x0}-{x1}) y=({y0}-{y1}) on node")

    for py in range(y0, y1):
        im = im_min + (py / (height - 1)) * (im_max - im_min)
        for px in range(x0, x1):
            re = re_min + (px / (width - 1)) * (re_max - re_min)
            c_re, c_im = re, im
            z_re, z_im = 0.0, 0.0
            it = 0
            while (z_re * z_re + z_im * z_im <= 4.0) and (it < max_it):
                z_re, z_im = z_re * z_re - z_im * z_im + c_re, 2 * z_re * z_im + c_im
                it += 1
            tile[py - y0, px - x0] = it

    print(f"End tile: x=({x0}-{x1}) y=({y0}-{y1})")
    return (x0, y0, tile)


def assemble_image(width, height, tiles, tile_size_x, tile_size_y):
    img = np.zeros((height, width), dtype=np.uint16)
    for (x0, y0, tile) in tiles:
        h, w = tile.shape
        img[y0:y0+h, x0:x0+w] = tile
    return img

if __name__ == "__main__":
    # Inicjalizacja Ray (połączony z head clusterem)
    #ray.init() #rownolegle
    #ray.init(num_cpus=4) #rownolegle
    ray.init(address="auto") # rozproszone
    start = time.time()

    # przygotuj siatkę kafelków
    tile_w = WIDTH // TILE_COLS
    tile_h = HEIGHT // TILE_ROWS
    tasks = []
    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            x0 = c * tile_w
            x1 = (c+1)*tile_w if c < TILE_COLS-1 else WIDTH
            y0 = r * tile_h
            y1 = (r+1)*tile_h if r < TILE_ROWS-1 else HEIGHT
            tasks.append(compute_tile.remote(x0, x1, y0, y1,
                                             WIDTH, HEIGHT, MAX_IT,
                                             RE_MIN, RE_MAX, IM_MIN, IM_MAX))

    # zbierz wyniki (Ray automatycznie rozdzieli zadania na węzły i rdzenie)
    tiles = ray.get(tasks)

    # złożenie obrazu i zapis
    img = assemble_image(WIDTH, HEIGHT, tiles, tile_w, tile_h)

    # opcjonalne ugrupowanie kolorów (prosty maping)
    norm = img.astype(np.float32) / MAX_IT
    cmap = plt.cm.viridis
    rgba_img = cmap(norm)
    plt.imsave("mandelbrot_ray.png", rgba_img)

    end = time.time()
    print(f"Zrobione w {end - start:.2f} s")
