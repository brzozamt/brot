# wave_2d_ray.py
import ray
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ---------- PARAMETRY ----------
GRID_W = 200
GRID_H = 200
ITERATIONS = 2000
TILE = 200      # kafelek
C = 0.5         # prędkość fali
DT = 0.1
DX = 1.0
SAVE_EVERY = 10
OUT_DIR = "wave_frames"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- KAFELKI ----------
@ray.remote
def update_tile(x0, x1, y0, y1, u_curr, u_prev):
    new_tile = np.copy(u_curr[y0:y1, x0:x1])
    for y in range(y0, y1):
        for x in range(x0, x1):
            if 1 <= x < u_curr.shape[1]-1 and 1 <= y < u_curr.shape[0]-1:
                laplacian = (
                    u_curr[y+1, x] + u_curr[y-1, x] +
                    u_curr[y, x+1] + u_curr[y, x-1] -
                    4*u_curr[y, x]
                )
                new_tile[y-y0, x-x0] = 2*u_curr[y, x] - u_prev[y, x] + (C*C)*(DT*DT)/(DX*DX)*laplacian
    return (x0, y0, new_tile)

# ---------- SKŁADANIE SIATKI ----------
def assemble_grid(grid, tiles):
    for (x0, y0, tile) in tiles:
        h, w = tile.shape
        grid[y0:y0+h, x0:x0+w] = tile
    return grid

# ---------- PROGRAM GŁÓWNY ----------
if __name__ == "__main__":
    ray.init()  # lokalnie, lub ray.init(address="auto") w rozproszeniu

    print("Tworzenie siatki...")
    u_prev = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    u_curr = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    # Źródła fal
    sources = [
        (GRID_W//4, GRID_H//4),
        (3*GRID_W//4, GRID_H//4),
        (GRID_W//2, 3*GRID_H//4)
    ]
    for (sx, sy) in sources:
        u_curr[sy, sx] = 100.0

    tiles_x = GRID_W // TILE
    tiles_y = GRID_H // TILE

    print("Start symulacji...")
    start_time = time.time()

    for it in range(ITERATIONS):
        # --- dodanie drugiego impulsu po 500 iteracji ---
        if it == 800:
            print("Dodanie drugiego impulsu!")
            # przykładowe pozycje nowych źródeł
            new_sources = [
                (GRID_W // 3, GRID_H // 3),
                (2 * GRID_W // 3, 2 * GRID_H // 3)
            ]
            for (sx, sy) in new_sources:
                u_curr[sy, sx] += 100.0  # dodaj amplitudę do istniejącej

        # --- obliczenia kafelków ---
        tasks = []
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x0 = tx * TILE
                x1 = min((tx + 1) * TILE, GRID_W)
                y0 = ty * TILE
                y1 = min((ty + 1) * TILE, GRID_H)
                tasks.append(update_tile.remote(x0, x1, y0, y1, u_curr, u_prev))

        tiles = ray.get(tasks)
        u_next = np.zeros_like(u_curr)
        u_next = assemble_grid(u_next, tiles)

        # zapis klatki
        if it % SAVE_EVERY == 0:
            plt.imsave(f"{OUT_DIR}/frame_{it:04d}.png", u_curr, cmap="viridis", vmin=-100, vmax=100)
            print(f"Klatka {it} zapisana.")

        # przygotowanie do następnego kroku
        u_prev, u_curr = u_curr, u_next

    end_time = time.time()
    print(f"Symulacja zakończona w {end_time-start_time:.2f} s")
