import heapq
import math
import random
import tkinter as tk
from typing import List, Tuple, Dict, Optional, Iterator

Grid = List[List[int]]
Coord = Tuple[int, int]

ROWS = 10
COLS = 10
OBSTACLE_PROB = 0.2  # default probability for randomizing

CELL_SIZE = 40  # pixels per cell for canvas display

# ---------- Heuristics ----------
def heuristic(a: Coord, b: Coord, method: str = "manhattan") -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if method == "manhattan":
        return dx + dy
    elif method == "euclidean":
        return math.hypot(dx, dy)
    elif method == "octile":  # good for diagonal cost = sqrt(2)
        D = 1.0
        D2 = math.sqrt(2)
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
    else:
        return dx + dy


def neighbors(pos: Coord, allow_diagonal: bool = False) -> Iterator[Tuple[Coord, float]]:
    r, c = pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            yield (nr, nc), 1.0
    if allow_diagonal:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                yield (nr, nc), math.sqrt(2)


def a_star(grid: Grid, start: Coord, goal: Coord, allow_diag: bool = False, heur: str = "manhattan") -> Optional[Tuple[List[Coord], float]]:
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None

    open_heap: List[Tuple[float, int, Coord]] = []
    counter = 0
    start_h = heuristic(start, goal, heur)
    heapq.heappush(open_heap, (start_h, counter, start))
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    f_score: Dict[Coord, float] = {start: start_h}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal]

        for (nb, cost) in neighbors(current, allow_diagonal=allow_diag):
            if grid[nb[0]][nb[1]] == 1:
                continue
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f = tentative_g + heuristic(nb, goal, heur)
                if f < f_score.get(nb, float("inf")):
                    f_score[nb] = f
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, nb))
    return None


def random_grid(ob_prob: float = OBSTACLE_PROB, seed: Optional[int] = None) -> Grid:
    if seed is not None:
        random.seed(seed)
    return [[1 if random.random() < ob_prob else 0 for _ in range(COLS)] for _ in range(ROWS)]


def example_grid() -> Grid:
    g = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    blocks = [
        (1, 2), (2, 2), (3, 2), (4, 2),
        (5, 5), (5, 6), (5, 7),
        (7, 3), (7, 4), (7, 5)
    ]
    for r, c in blocks:
        g[r][c] = 1
    return g


# ---------- GUI ----------
class AStarGUI:
    def __init__(self, master):
        self.master = master
        master.title("A* Visualizer")

        self.grid = random_grid()
        self.start: Coord = (0, 0)
        self.goal: Coord = (ROWS - 1, COLS - 1)
        self.path: Optional[List[Coord]] = None
        self.path_cost: Optional[float] = None

        self.mode = tk.StringVar(value="obstacle")  # "obstacle", "start", "goal"
        self.heur = tk.StringVar(value="manhattan")  # "manhattan", "euclidean", "octile"
        self.allow_diag = tk.BooleanVar(value=False)

        # Canvas
        canvas_width = COLS * CELL_SIZE
        canvas_height = ROWS * CELL_SIZE
        self.canvas = tk.Canvas(master, width=canvas_width, height=canvas_height, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=6)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Controls
        tk.Radiobutton(master, text="Draw Obstacles", variable=self.mode, value="obstacle").grid(row=1, column=0, sticky="w")
        tk.Radiobutton(master, text="Set Start (S)", variable=self.mode, value="start").grid(row=1, column=1, sticky="w")
        tk.Radiobutton(master, text="Set Goal (G)", variable=self.mode, value="goal").grid(row=1, column=2, sticky="w")

        tk.Button(master, text="Randomize", command=self.randomize).grid(row=2, column=0)
        tk.Button(master, text="Example", command=self.use_example).grid(row=2, column=1)
        tk.Button(master, text="Clear Obstacles", command=self.clear_obstacles).grid(row=2, column=2)
        tk.Button(master, text="Solve", command=self.solve).grid(row=2, column=3)

        tk.Label(master, text="Obstacle prob:").grid(row=3, column=0, sticky="e")
        self.prob_entry = tk.Entry(master, width=6)
        self.prob_entry.insert(0, str(OBSTACLE_PROB))
        self.prob_entry.grid(row=3, column=1, sticky="w")

        # Heuristic selection and diagonal toggle
        tk.Label(master, text="Heuristic:").grid(row=1, column=3, sticky="e")
        heur_menu = tk.OptionMenu(master, self.heur, "manhattan", "euclidean", "octile")
        heur_menu.grid(row=1, column=4, sticky="w")

        tk.Checkbutton(master, text="Allow Diagonals", variable=self.allow_diag).grid(row=1, column=5, sticky="w")

        self.status = tk.Label(master, text="Click cells to edit. Then press Solve.", anchor="w")
        self.status.grid(row=4, column=0, columnspan=6, sticky="we")

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                fill = "white"
                if self.grid[r][c] == 1:
                    fill = "black"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="gray")
        if self.path:
            for (r, c) in self.path:
                x0 = c * CELL_SIZE + 4
                y0 = r * CELL_SIZE + 4
                x1 = x0 + CELL_SIZE - 8
                y1 = y0 + CELL_SIZE - 8
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="blue", outline="")
        # Draw start and goal on top
        sr, sc = self.start
        gr, gc = self.goal
        self._draw_cell_marker(sr, sc, "green", "S")
        self._draw_cell_marker(gr, gc, "red", "G")

    def _draw_cell_marker(self, r, c, color, label):
        x0 = c * CELL_SIZE
        y0 = r * CELL_SIZE
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="gray")
        self.canvas.create_text(x0 + CELL_SIZE/2, y0 + CELL_SIZE/2, text=label, fill="white", font=("Helvetica", 12, "bold"))

    def on_canvas_click(self, event):
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return
        mode = self.mode.get()
        if mode == "obstacle":
            # toggle obstacle but don't overwrite start/goal
            if (r, c) == self.start or (r, c) == self.goal:
                return
            self.grid[r][c] = 0 if self.grid[r][c] == 1 else 1
            self.path = None
            self.path_cost = None
            self.status.config(text="Edited obstacles.")
        elif mode == "start":
            if self.grid[r][c] == 1:
                self.status.config(text="Cannot set start on obstacle.")
                return
            self.start = (r, c)
            self.path = None
            self.path_cost = None
            self.status.config(text=f"Start set to {self.start}.")
        elif mode == "goal":
            if self.grid[r][c] == 1:
                self.status.config(text="Cannot set goal on obstacle.")
                return
            self.goal = (r, c)
            self.path = None
            self.path_cost = None
            self.status.config(text=f"Goal set to {self.goal}.")
        self.draw_grid()

    def randomize(self):
        try:
            prob = float(self.prob_entry.get())
        except ValueError:
            prob = OBSTACLE_PROB
            self.prob_entry.delete(0, tk.END)
            self.prob_entry.insert(0, str(prob))
        self.grid = random_grid(ob_prob=prob)
        # ensure start and goal are free
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0
        self.path = None
        self.path_cost = None
        self.status.config(text="Randomized grid.")
        self.draw_grid()

    def use_example(self):
        self.grid = example_grid()
        self.grid[self.start[0]][self.start[1]] = 0
        self.grid[self.goal[0]][self.goal[1]] = 0
        self.path = None
        self.path_cost = None
        self.status.config(text="Loaded example grid.")
        self.draw_grid()

    def clear_obstacles(self):
        self.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.path = None
        self.path_cost = None
        self.status.config(text="Cleared obstacles.")
        self.draw_grid()

    def solve(self):
        res = a_star(self.grid, self.start, self.goal, allow_diag=self.allow_diag.get(), heur=self.heur.get())
        if res:
            self.path, self.path_cost = res
            steps = len(self.path) - 1
            self.status.config(text=f"Path found! Steps: {steps}, Distance: {self.path_cost:.3f}")
        else:
            self.path = None
            self.path_cost = None
            self.status.config(text="No path found.")
        self.draw_grid()


if __name__ == "__main__":
    root = tk.Tk()
    app = AStarGUI(root)
    root.mainloop()