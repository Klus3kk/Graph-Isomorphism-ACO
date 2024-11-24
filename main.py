import random
import time
import json
import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations
from tqdm import tqdm

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.pheromone_matrix = [[1 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.start_time = time.time()

    def add_edge(self, u, v, weight):
        self.adjacency_matrix[u][v] = weight
        self.adjacency_matrix[v][u] = weight

    def generate_random_graph(self, edge_probability=0.3, max_weight=10, graph_type="random"):
        """
        Generuje losowy graf.
        Parametry:
        - edge_probability: prawdopodobieństwo utworzenia krawędzi między dwoma wierzchołkami
        - max_weight: maksymalna wartość wagi krawędzi
        """
        print("Generowanie grafu typu:", graph_type)
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if graph_type == "random" and random.random() <= edge_probability:
                    weight = random.randint(1, max_weight)
                    self.add_edge(i, j, weight)
                elif graph_type == "complete":
                    weight = random.randint(1, max_weight)
                    self.add_edge(i, j, weight)
                elif graph_type == "sparse" and random.random() <= edge_probability / 2:
                    weight = random.randint(1, max_weight)
                    self.add_edge(i, j, weight)

    def save_to_json(self, filename="graph.json"):
        """
        Zapisuje graf do pliku JSON w formacie macierzy sąsiedztwa.
        """
        data = {
            "num_vertices": self.num_vertices,
            "adjacency_matrix": self.adjacency_matrix,
            "pheromone_matrix": self.pheromone_matrix
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Graf zapisany do pliku {filename}")

    def load_from_json(self, filename):
        """
        Ładuje graf z pliku JSON.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        self.num_vertices = data["num_vertices"]
        self.adjacency_matrix = data["adjacency_matrix"]
        self.pheromone_matrix = data.get("pheromone_matrix", [[1 for _ in range(self.num_vertices)] for _ in range(self.num_vertices)])

    def initialize_pheromones(self, initial_pheromone=1):
        """
        Inicjalizuje wartości feromonów na każdej krawędzi.
        """
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.adjacency_matrix[i][j] != 0:
                    self.pheromone_matrix[i][j] = initial_pheromone

    def greedy_isomorphism(self, graph2):
        """
        Implementacja zachłannego algorytmu do znajdowania izomorfizmu między grafami.
        """
        print("Wykonywanie algorytmu zachłannego...")
        mapping = []
        used_vertices_graph2 = set()
        for u in range(self.num_vertices):
            best_match = None
            best_score = -1
            for v in range(graph2.num_vertices):
                if v in used_vertices_graph2:
                    continue
                score = sum(1 for neighbor_u in range(self.num_vertices) 
                            if self.adjacency_matrix[u][neighbor_u] > 0 and 
                            graph2.adjacency_matrix[v][neighbor_u] > 0)
                if score > best_score:
                    best_match = v
                    best_score = score
            if best_match is not None:
                mapping.append((u, best_match))
                used_vertices_graph2.add(best_match)
        return mapping

    def randomized_greedy_isomorphism(self, graph2, randomness_factor=0.1):
        mapping = []
        used_vertices_graph2 = set()

        for u in range(self.num_vertices):
            candidates = []
            for v in range(graph2.num_vertices):
                if v not in used_vertices_graph2:
                    score = sum(1 for neighbor_u in range(self.num_vertices)
                                if self.adjacency_matrix[u][neighbor_u] > 0 and
                                graph2.adjacency_matrix[v][neighbor_u] > 0)
                    candidates.append((v, score))

            candidates.sort(key=lambda x: -x[1])
            if not candidates:
                continue

            if random.random() < randomness_factor and len(candidates) > 1:
                chosen = random.choice(candidates[1:])
            else:
                chosen = candidates[0]

            mapping.append((u, chosen[0]))
            used_vertices_graph2.add(chosen[0])

        return mapping

    def brute_force_isomorphism(self, graph2):
        print("Wykonywanie pełnego przeglądu (brute-force)...")
        best_mapping = None
        best_score = 0
        total_permutations = len(list(permutations(range(graph2.num_vertices))))  # Liczba permutacji
        with tqdm(total=total_permutations, desc="Brute-force progress", leave=True) as pbar:
            for perm in permutations(range(graph2.num_vertices)):
                mapping = [(u, v) for u, v in enumerate(perm)]
                score = self.evaluate_mapping(self, graph2, mapping)
                if score > best_score:
                    best_score = score
                    best_mapping = mapping
                pbar.update(1)  # Aktualizuj pasek postępu
        print("Algorytm brute-force zakończony!")
        return best_mapping, best_score


    def evaluate_mapping(self, graph1, graph2, mapping):
        """
        Ocena jakości dopasowania wierzchołków między grafami.
        """
        score = 0
        for u, v in mapping:
            for neighbor_u in range(graph1.num_vertices):
                if graph1.adjacency_matrix[u][neighbor_u] > 0:
                    mapped_neighbor_v = next((m[1] for m in mapping if m[0] == neighbor_u), None)
                    if mapped_neighbor_v is not None and graph2.adjacency_matrix[v][mapped_neighbor_v] == graph1.adjacency_matrix[u][neighbor_u]:
                        score += 1
        return score

    def update_pheromones(self, mappings, decay=0.5, pheromone_increase=1):
        """
        Aktualizuje feromony na podstawie dopasowań wierzchołków między dwoma grafami.
        """
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.adjacency_matrix[i][j] != 0:
                    self.pheromone_matrix[i][j] *= (1 - decay)
        for mapping in mappings:
            for (u, v) in mapping:
                self.pheromone_matrix[u][v] += pheromone_increase
                self.pheromone_matrix[v][u] += pheromone_increase

    def run_aco_for_isomorphism(self, graph1, graph2, num_ants=10, num_iterations=5, alpha=1, beta=1, decay=0.5, pheromone_increase=1):
        print("Rozpoczynanie algorytmu mrówkowego (ACO)...")
        best_mapping, best_score = None, float('-inf')
        for iteration in tqdm(range(num_iterations), desc="ACO iterations progress", leave=True):
            all_mappings = []
            for ant in range(num_ants):
                mapping = []
                used_vertices_graph2 = set()
                for u in range(graph1.num_vertices):
                    candidates = [v for v in range(graph2.num_vertices) if v not in used_vertices_graph2]
                    if not candidates:
                        break
                    v = random.choice(candidates)
                    used_vertices_graph2.add(v)
                    mapping.append((u, v))
                score = self.evaluate_mapping(graph1, graph2, mapping)
                all_mappings.append((mapping, score))
                if score > best_score:
                    best_mapping, best_score = mapping, score
            self.update_pheromones([m[0] for m in all_mappings], decay, pheromone_increase)
        print("Algorytm mrówkowy (ACO) zakończony!")
        return best_mapping, best_score


    def visualize_mapping(self, graph1, graph2, mapping):
        """
        Wizualizuje dopasowania wierzchołków między dwoma grafami.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        G1 = nx.Graph()
        for i in range(graph1.num_vertices):
            for j in range(i + 1, graph1.num_vertices):
                if graph1.adjacency_matrix[i][j] > 0:
                    G1.add_edge(i, j, weight=graph1.adjacency_matrix[i][j])
        pos1 = nx.spring_layout(G1, seed=42)
        for node in pos1:
            pos1[node] = (pos1[node][0] - 1.5, pos1[node][1])
        nx.draw(G1, pos1, with_labels=True, ax=ax, node_color='skyblue')
        edge_labels1 = nx.get_edge_attributes(G1, 'weight')
        nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_labels1, ax=ax)

        G2 = nx.Graph()
        for i in range(graph2.num_vertices):
            for j in range(i + 1, graph2.num_vertices):
                if graph2.adjacency_matrix[i][j] > 0:
                    G2.add_edge(i, j, weight=graph2.adjacency_matrix[i][j])
        pos2 = nx.spring_layout(G2, seed=42)
        for node in pos2:
            pos2[node] = (pos2[node][0] + 1.5, pos2[node][1])
        nx.draw(G2, pos2, with_labels=True, ax=ax, node_color='lightgreen')
        edge_labels2 = nx.get_edge_attributes(G2, 'weight')
        nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_labels2, ax=ax)

        for u, v in mapping:
            if u in pos1 and v in pos2:
                pos_u, pos_v = pos1[u], pos2[v]
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k--')
        plt.title("Dopasowanie między Grafem 1 a Grafem 2")
        plt.show()

    def evaluate_solution_quality(self, graph2, num_ants=5, num_iterations=3, alpha=1, beta=1, decay=0.3, pheromone_increase=2):
        print("Rozpoczynanie oceny jakości rozwiązań...")
        results = {}

        print("Uruchamianie algorytmu zachłannego...")
        start_time = time.time()
        mapping_greedy = self.greedy_isomorphism(graph2)
        score_greedy = self.evaluate_mapping(self, graph2, mapping_greedy)
        time_greedy = time.time() - start_time
        results['Greedy'] = {
            'Mapping': mapping_greedy,
            'Score': score_greedy,
            'Time': time_greedy
        }
        # Zapis grafu i mappingu do JSON
        self.save_mapping_to_json(graph2, mapping_greedy, "greedy_mapping.json")
        # Wizualizacja wyniku zachłannego
        self.visualize_mapping(self, graph2, mapping_greedy)

        print("Uruchamianie algorytmu brute-force...")
        start_time = time.time()
        mapping_brute, score_brute = self.brute_force_isomorphism(graph2)
        time_brute = time.time() - start_time
        results['BruteForce'] = {
            'Mapping': mapping_brute,
            'Score': score_brute,
            'Time': time_brute
        }
        # Zapis grafu i mappingu do JSON
        self.save_mapping_to_json(graph2, mapping_brute, "brute_force_mapping.json")
        # Wizualizacja wyniku brute-force
        if mapping_brute:
            self.visualize_mapping(self, graph2, mapping_brute)

        print("Uruchamianie algorytmu mrówkowego...")
        start_time = time.time()
        mapping_aco, score_aco = self.run_aco_for_isomorphism(self, graph2, num_ants=num_ants, num_iterations=num_iterations, alpha=alpha, beta=beta, decay=decay, pheromone_increase=pheromone_increase)
        time_aco = time.time() - start_time
        results['ACO'] = {
            'Mapping': mapping_aco,
            'Score': score_aco,
            'Time': time_aco
        }
        # Zapis grafu i mappingu do JSON
        self.save_mapping_to_json(graph2, mapping_aco, "aco_mapping.json")

        # Zapis wyników do pliku JSON
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
            print("Wyniki zapisane do pliku results.json")
        return results

    
    def save_mapping_to_json(self, graph2, mapping, filename):
        """
        Zapisuje grafy i mapping do pliku JSON.
        """
        data = {
            "Graph1": {
                "num_vertices": self.num_vertices,
                "adjacency_matrix": self.adjacency_matrix
            },
            "Graph2": {
                "num_vertices": graph2.num_vertices,
                "adjacency_matrix": graph2.adjacency_matrix
            },
            "Mapping": mapping
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Mapping zapisany do pliku {filename}")




# Przykład użycia i porównanie algorytmów
graph1 = Graph(10)
graph1.generate_random_graph(edge_probability=0.5, max_weight=20)
graph1.initialize_pheromones()

graph2 = Graph(10)
graph2.generate_random_graph(edge_probability=0.5, max_weight=20)
graph2.initialize_pheromones()

# Uruchomienie testów oceny jakości rozwiązań
results = graph1.evaluate_solution_quality(
    graph2, 
    num_ants=5, 
    num_iterations=5, 
    alpha=1, 
    beta=2, 
    decay=0.3, 
    pheromone_increase=2
)