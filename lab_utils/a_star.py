import heapq
from PIL import Image
import copy
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List


def astar(grid, start, end):
    """A* pathfinding algorithm with Manhattan heuristic and diagonal movement."""
    

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),]
                #  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    open_heap = []
    heapq.heappush(open_heap, (0, *start))
    came_from = {}
    g_scores = {start: 0}
    f_scores = {start: heuristic(start, end)}

    while open_heap:
        current_f, cx, cy = heapq.heappop(open_heap)
        if (cx, cy) == end:
            return reconstruct_path(came_from, end)

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (
                not (0 <= nx < len(grid) and 0 <= ny < len(grid[0]))
                or grid[nx][ny] != 0
            ):
                continue

            

            move_cost = 1.414 if dx != 0 and dy != 0 else 1
            tentative_g = g_scores[(cx, cy)] + move_cost
            
            if tentative_g < g_scores.get((nx, ny), float("inf")):
                came_from[(nx, ny)] = (cx, cy)
                g_scores[(nx, ny)] = tentative_g
                f = tentative_g + heuristic((nx, ny), end)
                heapq.heappush(open_heap, (f, nx, ny))

    return None


def heuristic(a, b):
    """Modified heuristic using diagonal distance."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    

    return max(dx, dy) + (1.414 - 1) * min(dx, dy)


def reconstruct_path(came_from, end):
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    return path[::-1]


def load_grid(image_path):
    """Convert image to binary grid representation."""
    with Image.open(image_path) as img:
        W, H = img.size
        return (
            [
                [0 if is_white(img.getpixel((j, i))) else 1 for j in range(W)]
                for i in range(H)
            ],
            W,
            H,
        )


def is_white(pixel):
    """Determine if pixel represents white space."""
    if isinstance(pixel, int):
        return pixel == 255
    channels = pixel[:3]
    return channels == (255, 255, 255)


def inflate_obstacles(grid: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * radius + 1, 2 * radius + 1))
    return binary_dilation(grid, structure=struct).astype(np.int32)


def real_to_grid(x, y, x_bounds, y_bounds, grid_size):
    """Convert real coordinates to grid indices."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    W, H = grid_size
    return (
        min(max(int((y_max - y) / (y_max - y_min) * H), 0), H - 1),
        min(max(int((x - x_min) / (x_max - x_min) * W), 0), W - 1),
    )


def grid_to_real(i, j, x_bounds, y_bounds, grid_size):
    """Convert grid indices to real coordinates."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    W, H = grid_size
    return (
        x_min + (j + 0.5) * (x_max - x_min) / W,
        y_max - (i + 0.5) * (y_max - y_min) / H,
    )


def save_path_image(grid_path, path, save_path=None):

    plt.figure(figsize=(10, 10))

    plt.imshow(grid_path, cmap='binary')
    
    if path:
        
        path_i, path_j = zip(*path)
        plt.plot(path_j, path_i, 'r-', linewidth=2, label='Path')
        plt.plot(path_j[0], path_i[0], 'go', markersize=10, label='Start')
        plt.plot(path_j[-1], path_i[-1], 'bo', markersize=10, label='End')
    
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plan_navigation_path(task_info: dict) -> Optional[Tuple[List[List[float]], List[List[int]]]]:

    grid, W, H = load_grid(task_info['asset']['barrier_image_path'])
    
    x_bounds = task_info['asset']['x_bounds']
    y_bounds = task_info['asset']['y_bounds']
    meters_per_pixel_x = (x_bounds[1] - x_bounds[0]) / W
    meters_per_pixel_y = (y_bounds[1] - y_bounds[0]) / H
    meters_per_pixel = min(meters_per_pixel_x, meters_per_pixel_y)
    radius_pixels = int(task_info['asset']['offset_radius'] / meters_per_pixel)

    inflated_grid = inflate_obstacles(grid, radius_pixels)
    start = real_to_grid(
        task_info['start'][0], task_info['start'][1],
        x_bounds, y_bounds, (W, H)
    )
    end = real_to_grid(
        task_info['end'][0], task_info['end'][1],
        x_bounds, y_bounds, (W, H)
    )

    if inflated_grid[start[0]][start[1]] == 1 or inflated_grid[end[0]][end[1]] == 1:
        return None

    path_grid = astar(inflated_grid, start, end)
    if not path_grid:
        print("No path found.")
        return None

    real_path = []
    for point in path_grid:
        real_x, real_y = grid_to_real(
            point[0], point[1],
            x_bounds, y_bounds, (W, H)
        )
        real_path.append([real_x, real_y, 0.0])  

        
    return real_path, path_grid
