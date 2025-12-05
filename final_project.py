import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform
import os
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


# helper to create text with colors derived from current ttk theme
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
        self.geometry("1400x1000")

        self.style = ttk.Style(self)

        # theme choice prefer aqua (but only works for MacOS) => fallback to clam
        if "aqua" in self.style.theme_names():
            self.style.theme_use("aqua")
        else:
            self.style.theme_use("clam")

        # fonts 
        if platform.system() == "Windows":
            base_font = ("Segoe UI", 12)
        else:
            base_font = ("SF Pro Text", 12)

        self.option_add("*TButton.Font", base_font)
        self.option_add("*TLabel.Font", base_font)
        self.option_add("*TEntry.Font", base_font)
        self.option_add("*TCombobox.Font", base_font)
        self.option_add("*Text.Font", base_font)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.campus_graph = self.campus_graph()

        self.navigator_tab = CampusNavigatorTab(notebook, self.campus_graph)
        self.study_tab = StudyPlannerTab(notebook)
        self.search_tab = NotesSearchTab(notebook)
        self.info_tab = InfoTab(notebook)

        notebook.add(self.navigator_tab, text="Campus Navigator")
        notebook.add(self.study_tab, text="Study Planner")
        notebook.add(self.search_tab, text="Notes Search")
        notebook.add(self.info_tab, text="Algorithm Info")

    @staticmethod
    def campus_graph():
        graph = {
            "CS": [("E", 1), ("EC", 4)],
            "E": [("CS", 1), ("EC", 3), ("KHS", 6)],
            "KHS": [("E", 6), ("CPAC", 8), ("VA", 9)],
            "EC": [("CS", 4), ("E", 3), ("H", 3)],
            "H": [("EC", 3), ("MH", 4)],
            "VA": [("CPAC", 2), ("KHS", 9)],
            "CPAC": [("VA", 2), ("MH", 2), ("KHS", 8)],
            "MH": [("DBH", 1), ("CPAC", 2), ("H", 5)],
            "DBH": [("MH", 1), ("LH", 2)],
            "LH": [("DBH", 1), ("SGMH", 2)],
            "SGMH": [("LH", 2), ("CP", 5)],
            "CP": [("SGMH", 5)],
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

        ttk.Button(
            top_frame,
            text="Show Graph (Adj List)",
            command=self.show_graph_adj_list
        ).pack(side="left", padx=10)

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
        
        
    def show_graph_adj_list(self):
        popup = tk.Toplevel(self)
        popup.title("Campus Graph (Adjacency List)")
        popup.geometry("400x300")  
        text_widget = tk.Text(popup, wrap="word")
        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # build adjacency list text
        lines = []
        for u in sorted(self.graph.keys()):
            neighbors = ", ".join(f"{v} ({w}) " for v, w in self.graph[u])
            lines.append(f"{u}:  {neighbors}")
        text_widget.insert("1.0", "\n".join(lines))

        text_widget.config(state="disabled")  


        

    def run_bfs(self):
        self.clear_output()
        start = self.start_var.get()
        end = self.end_var.get()

        path = bfs(self.graph, start, end)

        self.println("Breadth-First Search (BFS):")
        self.println("")

        if path is None:
            self.println(f"No path found from {start} to {end}.")
        else:
            self.println(f"BFS fewest-hops path from {start} to {end}:")
            self.println(" --> ".join(path))
            self.println(f"Number of hops: {len(path) - 1}")

        _, is_connected = dfs(self.graph, start)  # use DFS to check global connectivity from same start
        if is_connected:
            self.println("\nThe graph is CONNECTED.")
        else:
            self.println("\nThe graph is NOT CONNECTED.")
        
        self.println("\nNote: BFS ignores edge weights and minimizes hops only.")

    def run_dfs(self):
        self.clear_output()
        start = self.start_var.get()
        order, is_connected = dfs(self.graph, start)

        self.println("Depth-First Search (DFS):")
        self.println("")
        self.println(f"DFS order starting from {start}:")
        if order:
            self.println(" --> ".join(order))
            self.println(f"Number of nodes visited: {len(order)} / {len(self.graph)}")
        else:
            self.println("No nodes visited (start node not in graph).")
        self.println("")

        if is_connected:
            self.println("The graph is CONNECTED.")
        else:
            self.println("The graph is NOT CONNECTED.")
        
        self.println("\nNote:")
        self.println(" - DFS ignores the selected End node.")
        self.println(" - It performs a full traversal from the Start node to test reachability and connectivity.")


    def run_dijkstra(self):
        self.clear_output()
        start = self.start_var.get()
        end = self.end_var.get()

        t0 = time.perf_counter()
        dist, path = dijkstra(self.graph, start, end)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.println("Dijkstra's Algorithm (Shortest Path):")
        self.println("")

        if path is None:
            self.println(f"No path found from {start} to {end}.")
            self.println(f"Computation time: {elapsed_ms:.4f} ms")
            return

        detailed = [] # shows detailed weighted path: u --w--> v
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            w = None
            for nbr, w_candidate in self.graph[u]:
                if nbr == v:
                    w = w_candidate
                    break
            if w is not None:
                detailed.append(f"    {u}   --{w}-->   {v}\n")
            else:
                # fallback in weird edge cases
                detailed.append(f"{u} --> {v}")

        self.println(f"Shortest path from {start} to {end}:")
        if detailed:
            self.println("  " + "  ".join(detailed))
        else:
            self.println(f"  {start}") # single-node path case start == end
        self.println(f"Total distance: {dist}")
        self.println(f"Number of edges: {len(path) - 1}")
        self.println(f"Computation time: {elapsed_ms:.4f} ms\n")

        _, is_connected = dfs(self.graph, start)
        if is_connected:
            self.println("The graph is CONNECTED.")
        else:
            self.println("The graph is NOT CONNECTED.")
        
        self.println("\nNote: Dijkstra assumes non-negative edge weights.")


    def run_prim(self):
        self.clear_output()
        start = self.start_var.get()
        edges, total = prim_mst(self.graph, start)

        self.println("Minimum Spanning Tree (Prim's Algorithm):")
        self.println("")

        if not edges:
            self.println("No MST edges (graph may be empty or disconnected).")
            return

        for u, v, w in edges:
            self.println(f"{u} -- {v} (weight = {w})")

        self.println("")
        self.println(f"Total MST cost: {total}")

        self.println(f"Number of edges in MST: {len(edges)}")
        if len(edges) == max(0, len(self.graph) - 1):        # if MST has |V|-1 edges, the underlying undirected graph is connected
            self.println("The MST spans all vertices (graph is CONNECTED).")
        else:
            self.println("The MST does NOT span all vertices (graph is NOT fully connected).")
        self.println("")
        self.println("Note: Prim's MST ignores the selected End node.")
        self.println("The MST spans all buildings using minimum total distance.")
            



class StudyPlannerTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.tasks = []  
        input_frame = ttk.Frame(self)
        input_frame.pack(side="top", fill="x", padx=16, pady=8)

        ttk.Label(input_frame, text="Task:").pack(side="left", padx=(0, 4))
        self.name_entry = ttk.Entry(input_frame, width=25)
        self.name_entry.pack(side="left", padx=(0, 8))

        ttk.Label(input_frame, text="Time (h):").pack(side="left")
        self.time_entry = ttk.Entry(input_frame, width=8)
        self.time_entry.pack(side="left", padx=(0, 8))

        ttk.Label(input_frame, text="Value:").pack(side="left")
        self.value_entry = ttk.Entry(input_frame, width=8)
        self.value_entry.pack(side="left", padx=(0, 8))

        ttk.Button(input_frame, text="Add", command=self.add_task).pack(side="left", padx=(4, 4))
        ttk.Button(input_frame, text="Remove Selected", command=self.remove_selected).pack(side="left", padx=(4, 0))

        table_frame = ttk.Frame(self)
        table_frame.pack(side="top", fill="both", expand=False, padx=16, pady=(0, 8))

        self.tree = ttk.Treeview(
            table_frame,
            columns=("task", "time", "value"),
            show="headings",
            height=6
        )
        self.tree.heading("task", text="Task")
        self.tree.heading("time", text="Time (h)")
        self.tree.heading("value", text="Value")

        self.tree.column("task", width=250, anchor="w")
        self.tree.column("time", width=80, anchor="center")
        self.tree.column("value", width=80, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side="top", fill="x", padx=16, pady=(0, 8))

        ttk.Label(bottom_frame, text="Available Time (h):").pack(side="left")
        self.capacity_entry = ttk.Entry(bottom_frame, width=10)
        self.capacity_entry.pack(side="left", padx=(4, 12))

        ttk.Button(bottom_frame, text="Run Greedy", command=self.run_greedy).pack(side="left", padx=4)
        ttk.Button(bottom_frame, text="Run DP (Knapsack)", command=self.run_dp).pack(side="left", padx=4)
        ttk.Button(bottom_frame, text="Run Both (Compare)", command=self.run_both).pack(side="left", padx=4)

        self.output = themed_text(self)
        self.output.pack(fill="both", expand=True, padx=16, pady=8)
        self.set_idle_message()


    def set_idle_message(self):
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(
            tk.END,
            "Study Planner:\n"
            "Add tasks, set available time, then run Greedy or DP.\n"
        )
        self.output.config(state="disabled")


    def clear_output(self):
        self.output.config(state="normal")
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
        self.tree.insert("", tk.END, values=(name, t, v))

        self.name_entry.delete(0, tk.END)
        self.time_entry.delete(0, tk.END)
        self.value_entry.delete(0, tk.END)


    def remove_selected(self):
        selected = self.tree.selection()
        if not selected:
            return

        for item in selected:
            vals = self.tree.item(item, "values")
            name, t_str, v_str = vals
            try:
                t = int(t_str)
                v = int(v_str)
            except ValueError:
                t = v = None
            for idx, (n, tt, vv) in enumerate(self.tasks):
                if n == name and (t is None or (tt == t and vv == v)):
                    del self.tasks[idx]
                    break
            self.tree.delete(item)

    def get_capacity(self):
        try:
            return int(self.capacity_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Available time must be integer")
            return None

    def run_greedy(self):
        cap = self.get_capacity()
        if cap is None:
            return
        self.clear_output()
        chosen, total_time, total_value = greedy_schedule(self.tasks, cap)
        self.println("Greedy schedule:")
        for name, t, v in chosen:
            self.println(f"   - {name}  (time = {t}, value = {v})")
        self.println(f"\nTotal time: {total_time}")
        self.println(f"Total value: {total_value}")
        self.output.config(state="disabled")


    def run_dp(self):
        cap = self.get_capacity()
        if cap is None:
            return
        self.clear_output()
        chosen, total_time, total_value = dp_schedule(self.tasks, cap)
        self.println("DP optimal schedule (0/1 Knapsack):")
        for name, t, v in chosen:
            self.println(f"   - {name} (time = {t},  value = {v})")
        self.println(f"\nTotal time: {total_time}")
        self.println(f"Total value: {total_value}")
        self.output.config(state="disabled")


    def run_both(self):
        cap = self.get_capacity()
        if cap is None:
            return

        self.clear_output()

        t0 = time.perf_counter()
        g_chosen, g_time, g_value = greedy_schedule(self.tasks, cap)
        g_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        d_chosen, d_time, d_value = dp_schedule(self.tasks, cap)
        d_ms = (time.perf_counter() - t0) * 1000

        self.println(f"Study Planner Comparison (capacity = {cap}):")
        self.println("")

        self.println("Greedy schedule:")
        for name, t, v in g_chosen:
            self.println(f"   - {name} (time = {t},  value = {v})")
        self.println(f"Greedy total time: {g_time}")
        self.println(f"Greedy total value: {g_value}")
        self.println(f"Greedy runtime: {g_ms:.4f} ms")
        self.println("")

        self.println("DP optimal schedule (0/1 Knapsack):")
        for name, t, v in d_chosen:
            self.println(f"   - {name} (time = {t},  value = {v})")
        self.println(f"DP total time: {d_time}")
        self.println(f"DP total value: {d_value}")
        self.println(f"DP runtime: {d_ms:.4f} ms")
        self.println("")

        self.println("\nSummary:")
        if g_value == d_value:
            self.println("  - Greedy and DP achieved the SAME total value.")
        elif g_value < d_value:
            self.println("  - DP found a BETTER schedule (higher total value) than Greedy.")
        else:
            self.println("  - Greedy found a schedule with higher value (unusual for 0/1 knapsack).")

        self.println("  - Greedy is a fast heuristic based on value density.")
        self.println("  - DP is guaranteed optimal but uses more time/space (O(n * C)).")

        self.output.config(state="disabled")



class NotesSearchTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.text = ""
        self.current_doc_name = None

        top_frame = ttk.Frame(self)
        top_frame.pack(side="top", fill="x", padx=16, pady=8)

        ttk.Label(top_frame, text="Document:").grid(row=0, column=0, sticky="w")

        self.doc_var = tk.StringVar(value="No document loaded")
        self.doc_entry = ttk.Entry(top_frame, textvariable=self.doc_var, state="readonly", width=40)
        self.doc_entry.grid(row=0, column=1, sticky="we", padx=(4, 8))

        ttk.Button(top_frame, text="Load Document...",
                   command=self.load_file).grid(row=0, column=2, padx=(0, 12))

        ttk.Label(top_frame, text="Pattern:").grid(row=0, column=3, sticky="w")
        self.pattern_entry = ttk.Entry(top_frame, width=20)
        self.pattern_entry.grid(row=0, column=4, padx=(4, 0))

        top_frame.columnconfigure(1, weight=1)

        alg_frame = ttk.Frame(self)
        alg_frame.pack(side="top", fill="x", padx=16, pady=(0, 8))

        ttk.Label(alg_frame, text="Algorithm:").pack(side="left")

        self.alg_var = tk.StringVar(value="ALL")  

        ttk.Radiobutton(alg_frame, text="Naive",
                        variable=self.alg_var, value="NAIVE").pack(side="left", padx=4)
        ttk.Radiobutton(alg_frame, text="Rabin-Karp",
                        variable=self.alg_var, value="RABIN-KARP").pack(side="left", padx=4)
        ttk.Radiobutton(alg_frame, text="KMP",
                        variable=self.alg_var, value="KMP").pack(side="left", padx=4)
        ttk.Radiobutton(alg_frame, text="All (compare)",
                        variable=self.alg_var, value="ALL").pack(side="left", padx=4)

        ttk.Button(alg_frame, text="Search",
                   command=self.run_search).pack(side="left", padx=12)

        doc_frame = ttk.LabelFrame(self, text="Document View")
        doc_frame.pack(side="top", fill="both", expand=True, padx=16, pady=(0, 8))

        self.doc_view = themed_text(doc_frame)
        doc_scroll = ttk.Scrollbar(doc_frame, orient="vertical", command=self.doc_view.yview)
        self.doc_view.configure(yscrollcommand=doc_scroll.set)

        self.doc_view.pack(side="left", fill="both", expand=True)
        doc_scroll.pack(side="right", fill="y")
        self.doc_view.config(state="disabled")

        result_frame = ttk.LabelFrame(self, text="Results")
        result_frame.pack(side="top", fill="both", expand=True, padx=16, pady=(0, 8))

        self.result_view = themed_text(result_frame)
        res_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_view.yview)
        self.result_view.configure(yscrollcommand=res_scroll.set)

        self.result_view.pack(side="left", fill="both", expand=True)
        res_scroll.pack(side="right", fill="y")

        self.set_idle_message()

    def set_idle_message(self):
        self.result_view.config(state="normal")
        self.result_view.delete("1.0", tk.END)
        self.result_view.insert(
            tk.END,
            "Load a document, enter a pattern, then choose an algorithm and press Search.\n"
        )
        self.result_view.config(state="disabled")

    def clear_results(self):
        self.result_view.config(state="normal")
        self.result_view.delete("1.0", tk.END)

    def println_result(self, text):
        self.result_view.insert(tk.END, str(text) + "\n")

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

        self.current_doc_name = os.path.basename(filepath)
        self.doc_var.set(self.current_doc_name)

        self.doc_view.config(state="normal")
        self.doc_view.delete("1.0", tk.END)
        self.doc_view.insert(tk.END, self.text)
        self.doc_view.config(state="disabled")

        self.set_idle_message()


    def run_search(self):
        if not self.text:
            messagebox.showinfo("Info", "No document loaded.")
            return

        pattern = self.pattern_entry.get()
        if not pattern:
            messagebox.showinfo("Info", "Pattern is empty.")
            return

        alg = self.alg_var.get()
        self.clear_results()
        doc_name = self.current_doc_name or "document"
        self.println_result(
            f"Comparing algorithms on '{doc_name}' for pattern '{pattern}':\n"
        )


        def time_alg(fn, label):
            start = time.perf_counter()
            matches = fn(self.text, pattern)
            elapsed = time.perf_counter() - start  
            self.println_result(f"[{label}]")
            if matches:
                self.println_result("Matches at indices: " + ", ".join(map(str, matches)))
            else:
                self.println_result("Matches at indices: (none)")
            self.println_result(f"Time: {elapsed:.6f} s\n")

        if alg == "NAIVE":
            time_alg(naive_search, "Naive")
        elif alg == "RABIN-KARP":
            time_alg(rabin_karp, "Rabin-Karp")
        elif alg == "KMP":
            time_alg(kmp_search, "KMP")
        else:  
            time_alg(naive_search, "Naive")
            time_alg(rabin_karp, "Rabin-Karp")
            time_alg(kmp_search, "KMP")
        self.result_view.config(state="disabled")


class InfoTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        text = themed_text(self)
        text.pack(fill="both", expand=True, padx=16, pady=12)

        content = []
        content.append("Algorithm Info & Complexity (CPSC 335)\n")
        content.append("-----------------------------------------------\n")
        content.append("This project focuses on polynomial-time algorithms and practical heuristics that are efficient and useful in real applications.\n\n")

        content.append("Graph Algorithms:\n")
        content.append("  BFS: O(V + E)\n")
        content.append("  DFS: O(V + E)\n")
        content.append("  Dijkstra (heap): O((V + E) log V)\n")
        content.append("  Prim MST (heap): O(E log V)\n\n")

        content.append("Study Planner:\n")
        content.append("  Greedy schedule (value/time): O(n log n)\n")
        content.append("  DP 0/1 Knapsack: O(n * C) where C = capacity in time units\n\n")

        content.append("String Matching:\n")
        content.append("  1) Naive search:\n")
        content.append("       - Worst Case: O(n * m)  [pattern occurs everywhere or repeated prefix]\n")
        content.append("       - Average Case: O((n - m + 1) * m)  [normal random text]\n")
        content.append("       - Best Case: O(n) [mismatches on 1st char each shift]\n")

        content.append("  2) Rabin-Karp:\n")
        content.append("       - Worst Case: O(n * m)  [many hash collisions]\n")
        content.append("       - Average Case: O(n + m)  [hash checks mostly unique]\n")
        content.append("       - Best Case: O(n + m)   [fast hash match + early mismatches]\n")

        content.append("  3) KMP: O(n + m)  in ALL cases  [pattern preprocessing avoids re-checking]\n\n")

        content.append("P vs NP (informal reflection):\n")
        content.append("   - In this app, we solve following in polynomial time (they are in P):\n")
        content.append("          - Shortest paths (Dijkstra)\n")
        content.append("          - Minimum spanning tree (Prim)\n")
 
        content.append("   - If instead we ask for:\n")
        content.append('          \"the shortest tour that visits every building exactly once and returns to the start,"\n')
        content.append("           - we get the Traveling Salesman Problem (TSP), which is NP-hard.\n")

        content.append("   - Intuitively:\n")
        content.append("          - P: problems we can solve in polynomial time.\n")
        content.append("          - NP: problems where we can verify a given solution in polynomial time.\n")
        content.append("          - NP-complete: the hardest problems in NP; a polynomial-time solution for one NP-complete problem would imply P = NP.\n")

        text.insert(tk.END, "".join(content))
        text.config(state="disabled")



if __name__ == "__main__":
    app = gui()
    app.mainloop()
