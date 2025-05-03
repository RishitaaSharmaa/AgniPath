import numpy as np
import cv2
import heapq


img = cv2.imread('Evac/map.png')
if img is None:
    raise Exception("Could not load image")
h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, occupancy_grid = cv2.threshold(gray, 220, 1, cv2.THRESH_BINARY_INV)
occupancy_grid = 1 - occupancy_grid  # 0 = walkable, 1 = wall

cv2.imwrite('occupancy_grid_debug.jpg', occupancy_grid * 255)

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, (0 + np.linalg.norm(np.subtract(goal, start)), 0, start, [start]))
    visited = set()

    while open_list:
        est_total, cost, current, path = heapq.heappop(open_list)
        current = tuple(map(int, current))

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
                new_cost = cost + 1
                est = new_cost + np.linalg.norm(np.subtract(goal, (nx, ny)))
                heapq.heappush(open_list, (est, new_cost, (nx, ny), path + [(nx, ny)]))
    return None

# --- Step 4: Start & Goal (adjusted to known walkable pixels manually) ---
start = (30, h - 30)       # Bottom-left room
goal = (w - 80, 30)        # Top-right room

# --- Step 5: Confirm Start and Goal are on Walkable Pixels ---
def find_nearest_walkable(pt, grid):
    x, y = pt
    if grid[y, x] == 0:
        return pt
    for r in range(1, 10):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    if grid[ny, nx] == 0:
                        return (nx, ny)
    raise ValueError("No walkable point near", pt)

start = find_nearest_walkable(start, occupancy_grid)
goal = find_nearest_walkable(goal, occupancy_grid)

# --- Step 6: Run A* ---
path = astar(occupancy_grid, start, goal)

# --- Step 7: Visualize ---
if path:
    print(f"✅ Path found with {len(path)} steps.")
    img_path = img.copy()
    for i in range(1, len(path)):
        cv2.line(img_path, path[i - 1], path[i], (0, 255, 0), 2)
    cv2.circle(img_path, start, 6, (255, 0, 0), -1)
    cv2.circle(img_path, goal, 6, (0, 0, 255), -1)
    cv2.putText(img_path, "Start", (start[0] + 10, start[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img_path, "Goal", (goal[0] + 10, goal[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite('final_path_result.jpg', img_path)
    print("🖼️ Saved to: final_path_result.jpg")
else:
    print("❌ No path found.")
