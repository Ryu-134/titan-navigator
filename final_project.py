import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform

from collections import deque
import heapq
import time

import PyPDF2
import docx


# graph algos
def bfs(graph, start, goal):
    if start not in graph or goal not in graph:
        return None

    q = deque([start])
    parent = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v, _w in graph[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)

    if goal not in parent:
        return None

    # reconstruct path
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def dfs(graph, start):
    visited = set()
    order = []

    def _dfs(u):
        visited.add(u)
        order.append(u)
        for v, _w in graph[u]:
            if v not in visited:
                _dfs(v)

    if start not in graph:
        return [], False

    _dfs(start)
    is_connected = (len(visited) == len(graph))
    return order, is_connected


def dijkstra(graph, start, goal):
    if start not in graph or goal not in graph:
        return float('inf'), None

    dist = {node: float('inf') for node in graph}
    parent = {node: None for node in graph}
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == goal:
            break
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[goal] == float('inf'):
        return float('inf'), None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return dist[goal], path


def prim_mst(graph, start=None):
    if not graph:
        return [], 0

    if start is None:
        start = next(iter(graph.keys()))

    visited = set([start])
    edges = []
    heap = []

    for v, w in graph[start]:
        heapq.heappush(heap, (w, start, v))

    while heap and len(visited) < len(graph):
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        edges.append((u, v, w))
        for nxt, w2 in graph[v]:
            if nxt not in visited:
                heapq.heappush(heap, (w2, v, nxt))

    total = sum(w for _u, _v, w in edges)
    return edges, total


# study planner algos
def greedy_schedule(tasks, capacity):
    tasks_sorted = sorted(tasks, key=lambda t: t[2] / t[1], reverse=True)
    chosen = []
    total_time = 0
    total_value = 0

    for name, t, v in tasks_sorted:
        if total_time + t <= capacity:
            chosen.append((name, t, v))
            total_time += t
            total_value += v

    return chosen, total_time, total_value


def dp_schedule(tasks, capacity):
    n = len(tasks)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        name, time_req, val = tasks[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if time_req <= c:
                candidate = dp[i - 1][c - time_req] + val
                if candidate > dp[i][c]:
                    dp[i][c] = candidate

    chosen = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            name, time_req, val = tasks[i - 1]
            chosen.append((name, time_req, val))
            c -= time_req
    chosen.reverse()
    total_time = sum(t for _, t, _ in chosen)
    total_value = dp[n][capacity]
    return chosen, total_time, total_value


# string matching algos
def naive_search(text, pattern):
    n, m = len(text), len(pattern)
    matches = []
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            matches.append(i)
    return matches


def rabin_karp(text, pattern, base=256, mod=10**9 + 7):
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []

    matches = []
    h = 1
    for _ in range(m - 1):
        h = (h * base) % mod

    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (p_hash * base + ord(pattern[i])) % mod
        t_hash = (t_hash * base + ord(text[i])) % mod

    for i in range(n - m + 1):
        if p_hash == t_hash:
            if text[i:i + m] == pattern:
                matches.append(i)
        if i < n - m:
            t_hash = (t_hash - ord(text[i]) * h) % mod
            t_hash = (t_hash * base + ord(text[i + m])) % mod
            t_hash = (t_hash + mod) % mod

    return matches


def kmp_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return []

    lps = kmp_lps(pattern)
    i = 0
    j = 0
    matches = []

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches


# text loading helper
def load_text_from_file(filepath):
    if filepath.lower().endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if filepath.lower().endswith(".pdf"):
        text = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    if filepath.lower().endswith(".docx"):
        document = docx.Document(filepath)
        return "\n".join(p.text for p in document.paragraphs)

    raise RuntimeError("Unsupported file type")


# helper: create Text with colors derived from current ttk theme
def themed_text(parent):
    style = ttk.Style()
    bg = style.lookup("TFrame", "background") or "#ffffff"
    fg = style.lookup("TLabel", "foreground") or "#000000"
    return tk.Text(
        parent,
        wrap="word",
        bg=bg,
        fg=fg,
        insertbackground=fg
    )


# GUI
class gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Titan Campus Algorithmic Assistant (TCAA)")
        self.geometry("900x600")

        self.style = ttk.Style(self)

        # theme choice prefer aqua (but only works for MacOS) => fallback to clam
        if "aqua" in self.style.theme_names():
            self.style.theme_use("aqua")
        else:
            self.style.theme_use("clam")

        # fonts (optional)
        if platform.system() == "Windows":
            base_font = ("Segoe UI", 10)
        else:
            base_font = ("SF Pro Text", 11)

        self.option_add("*TButton.Font", base_font)
        self.option_add("*TLabel.Font", base_font)
        self.option_add("*TEntry.Font", base_font)
        self.option_add("*TCombobox.Font", base_font)
        self.option_add("*Text.Font", base_font)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.campus_graph = self.create_sample_campus_graph()

        self.navigator_tab = CampusNavigatorTab(notebook, self.campus_graph)
        self.study_tab = StudyPlannerTab(notebook)
        self.search_tab = NotesSearchTab(notebook)
        self.info_tab = InfoTab(notebook)

        notebook.add(self.navigator_tab, text="Campus Navigator")
        notebook.add(self.study_tab, text="Study Planner")
        notebook.add(self.search_tab, text="Notes Search")
        notebook.add(self.info_tab, text="Algorithm Info")

    @staticmethod
    def create_sample_campus_graph():
        graph = {
            "Library": [("CS", 5), ("Gym", 7)],
            "CS": [("Library", 5), ("Cafeteria", 3)],
            "Gym": [("Library", 7), ("Cafeteria", 4)],
            "Cafeteria": [("CS", 3), ("Gym", 4)],
        }
        return graph


class CampusNavigatorTab(ttk.Frame):
    def __init__(self, parent, graph):
        super().__init__(parent)

        self.graph = graph
        nodes = list(self.graph.keys())

        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=16, pady=12)

        ttk.Label(top_frame, text="Start:").pack(side="left")
        self.start_var = tk.StringVar(value=nodes[0] if nodes else "")
        self.start_menu = ttk.Combobox(
            top_frame,
            textvariable=self.start_var,
            values=nodes,
            state="readonly",
            width=15
        )
        self.start_menu.pack(side="left", padx=6)

        ttk.Label(top_frame, text="End:").pack(side="left", padx=(16, 0))
        self.end_var = tk.StringVar(value=nodes[0] if nodes else "")
        self.end_menu = ttk.Combobox(
            top_frame,
            textvariable=self.end_var,
            values=nodes,
            state="readonly",
            width=15
        )
        self.end_menu.pack(side="left", padx=6)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="top", fill="x", padx=16, pady=(0, 8))

        ttk.Button(btn_frame, text="BFS Path",
                   command=self.run_bfs).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="DFS Traversal",
                   command=self.run_dfs).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Dijkstra Path",
                   command=self.run_dijkstra).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Prim MST",
                   command=self.run_prim).pack(side="left", padx=4)

        self.output = themed_text(self)
        self.output.pack(fill="both", expand=True, padx=16, pady=12)

    def clear_output(self):
        self.output.delete("1.0", tk.END)

    def println(self, text):
        self.output.insert(tk.END, str(text) + "\n")

    def run_bfs(self):
        self.clear_output()
        start = self.start_var.get()
        end = self.end_var.get()
        path = bfs(self.graph, start, end)
        if path is None:
            self.println("No path found")
        else:
            self.println(f"BFS fewest hops path from {start} to {end}:")
            self.println(" -> ".join(path))
            self.println(f"Number of hops: {len(path) - 1}")

    def run_dfs(self):
        self.clear_output()
        start = self.start_var.get()
        order, is_connected = dfs(self.graph, start)
        self.println(f"DFS order starting from {start}:")
        self.println(" -> ".join(order))
        self.println(f"Graph connected (from {start})? {is_connected}")

    def run_dijkstra(self):
        self.clear_output()
        start = self.start_var.get()
        end = self.end_var.get()
        dist, path = dijkstra(self.graph, start, end)
        if path is None:
            self.println("No path found")
        else:
            self.println(f"Dijkstra shortest path from {start} to {end}:")
            self.println(" -> ".join(path))
            self.println(f"Total distance: {dist}")

    def run_prim(self):
        self.clear_output()
        start = self.start_var.get()
        edges, total = prim_mst(self.graph, start)
        self.println(f"Prim MST starting from {start}:")
        for u, v, w in edges:
            self.println(f"{u} -- {v} (w={w})")
        self.println(f"Total MST weight: {total}")


class StudyPlannerTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.tasks = []

        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=16, pady=12)

        ttk.Label(top_frame, text="Task name:").grid(row=0, column=0, sticky="w")
        self.name_entry = ttk.Entry(top_frame, width=24)
        self.name_entry.grid(row=0, column=1, padx=6, pady=2)

        ttk.Label(top_frame, text="Time (int):").grid(row=1, column=0, sticky="w")
        self.time_entry = ttk.Entry(top_frame, width=12)
        self.time_entry.grid(row=1, column=1, padx=6, pady=2)

        ttk.Label(top_frame, text="Value (int):").grid(row=2, column=0, sticky="w")
        self.value_entry = ttk.Entry(top_frame, width=12)
        self.value_entry.grid(row=2, column=1, padx=6, pady=2)

        ttk.Button(top_frame, text="Add Task",
                   command=self.add_task).grid(row=3, column=0, columnspan=2, pady=8)

        ttk.Label(top_frame, text="Available time (int):").grid(row=4, column=0, sticky="w")
        self.capacity_entry = ttk.Entry(top_frame, width=12)
        self.capacity_entry.grid(row=4, column=1, padx=6, pady=2)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="top", fill="x", padx=16, pady=(0, 8))

        ttk.Button(btn_frame, text="Run Greedy",
                   command=self.run_greedy).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Run DP",
                   command=self.run_dp).pack(side="left", padx=4)

        self.output = themed_text(self)
        self.output.pack(fill="both", expand=True, padx=16, pady=12)

    def clear_output(self):
        self.output.delete("1.0", tk.END)

    def println(self, text):
        self.output.insert(tk.END, str(text) + "\n")

    def add_task(self):
        name = self.name_entry.get().strip()
        try:
            t = int(self.time_entry.get())
            v = int(self.value_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Time and value must be integers")
            return
        if not name:
            messagebox.showerror("Error", "Task name required")
            return
        self.tasks.append((name, t, v))
        self.println(f"Added task: {name}, time={t}, value={v}")
        self.name_entry.delete(0, tk.END)
        self.time_entry.delete(0, tk.END)
        self.value_entry.delete(0, tk.END)

    def get_capacity(self):
        try:
            return int(self.capacity_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Available time must be integer")
            return None

    def run_greedy(self):
        self.clear_output()
        cap = self.get_capacity()
        if cap is None:
            return
        chosen, total_time, total_value = greedy_schedule(self.tasks, cap)
        self.println("Greedy schedule:")
        for name, t, v in chosen:
            self.println(f"- {name} (time={t}, value={v})")
        self.println(f"Total time: {total_time}")
        self.println(f"Total value: {total_value}")

    def run_dp(self):
        self.clear_output()
        cap = self.get_capacity()
        if cap is None:
            return
        chosen, total_time, total_value = dp_schedule(self.tasks, cap)
        self.println("DP optimal schedule:")
        for name, t, v in chosen:
            self.println(f"- {name} (time={t}, value={v})")
        self.println(f"Total time: {total_time}")
        self.println(f"Total value: {total_value}")


class NotesSearchTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.text = ""

        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=16, pady=12)

        ttk.Button(top_frame, text="Load File",
                   command=self.load_file).grid(row=0, column=0, padx=6)
        self.file_label = ttk.Label(top_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, sticky="w")

        ttk.Label(top_frame, text="Pattern:").grid(row=1, column=0, sticky="w")
        self.pattern_entry = ttk.Entry(top_frame, width=30)
        self.pattern_entry.grid(row=1, column=1, padx=6)

        ttk.Label(top_frame, text="Algorithm:").grid(row=2, column=0, sticky="w")
        self.alg_var = tk.StringVar(value="ALL")
        self.alg_menu = ttk.Combobox(
            top_frame,
            textvariable=self.alg_var,
            values=["NAIVE", "RABIN-KARP", "KMP", "ALL"],
            state="readonly"
        )
        self.alg_menu.grid(row=2, column=1, padx=6)

        ttk.Button(top_frame, text="Search",
                   command=self.run_search).grid(row=3, column=0, columnspan=2, pady=8)

        self.output = themed_text(self)
        self.output.pack(fill="both", expand=True, padx=16, pady=12)

    def clear_output(self):
        self.output.delete("1.0", tk.END)

    def println(self, text):
        self.output.insert(tk.END, str(text) + "\n")

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("All supported", "*.txt *.pdf *.docx"),
                       ("Text", "*.txt"),
                       ("PDF", "*.pdf"),
                       ("Word", "*.docx"),
                       ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            self.text = load_text_from_file(filepath)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.file_label.config(text=filepath)
        self.println(f"Loaded file: {filepath}")
        self.println(f"Text length: {len(self.text)}")

    def run_search(self):
        self.clear_output()
        if not self.text:
            self.println("No text loaded")
            return
        pattern = self.pattern_entry.get()
        if not pattern:
            self.println("Pattern is empty")
            return
        alg = self.alg_var.get()

        def time_alg(fn, name):
            start = time.perf_counter()
            matches = fn(self.text, pattern)
            elapsed = (time.perf_counter() - start) * 1000
            self.println(f"{name}:")
            self.println(f"  Matches at indices: {matches}")
            self.println(f"  Time: {elapsed:.4f} ms")
            self.println("")
            return matches, elapsed

        if alg in ("NAIVE", "ALL"):
            time_alg(naive_search, "Naive search")
        if alg in ("RABIN-KARP", "ALL"):
            time_alg(rabin_karp, "Rabin-Karp")
        if alg in ("KMP", "ALL"):
            time_alg(kmp_search, "KMP")


class InfoTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        text = themed_text(self)
        text.pack(fill="both", expand=True, padx=16, pady=12)

        content = []
        content.append("Algorithm Info / Big-O\n")
        content.append("------------------------\n\n")

        content.append("Graph algorithms:\n")
        content.append("  BFS       : O(V + E)\n")
        content.append("  DFS       : O(V + E)\n")
        content.append("  Dijkstra  : O((V + E) log V) using heap\n")
        content.append("  Prim MST  : O(E log V) using heap\n\n")

        content.append("Study planner (Knapsack):\n")
        content.append("  Greedy density sort : O(n log n)\n")
        content.append("  DP 0/1 knapsack     : O(n * C) where C=capacity\n\n")

        content.append("String matching:\n")
        content.append("  Naive      : O(n * m)\n")
        content.append("  Rabin-Karp : O(n + m) average, O(n * m) worst (hash collisions)\n")
        content.append("  KMP        : O(n + m)\n\n")

        content.append("P vs NP reflection (brief):\n")
        content.append("  - P   : Problems solvable in polynomial time, e.g., BFS, Dijkstra (with non-negative weights)\n")
        content.append("  - NP  : Solutions can be verified in polynomial time\n")
        content.append("  - 0/1 Knapsack is NP-complete in general\n")
        content.append("  - We used a pseudo-polynomial DP because capacity is treated as a number dimension\n")
        content.append("  - Open question: is P = NP?  (Still unsolved)\n")

        text.insert(tk.END, "".join(content))
        text.config(state="disabled")


if __name__ == "__main__":
    app = gui()
    app.mainloop()
