import random
import time
import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        # Tworzymy macierz sąsiedztwa wypełnioną zerami (brak krawędzi)
        self.adjacency_matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.pheromone_matrix = [[1 for _ in range(num_vertices)] for _ in range(num_vertices)]
        self.start_time = time.time()

    def add_edge(self, u, v, weight):
        # Dodajemy krawędź z wagą (dla grafu nieskierowanego)
        self.adjacency_matrix[u][v] = weight
        self.adjacency_matrix[v][u] = weight

    def generate_random_graph(self, edge_probability=0.3, max_weight=10):
        """
        Generuje losowy graf. 
        Parametry:
        - edge_probability: prawdopodobieństwo utworzenia krawędzi między dwoma wierzchołkami
        - max_weight: maksymalna wartość wagi krawędzi
        """
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if random.random() <= edge_probability:
                    weight = random.randint(1, max_weight)
                    self.add_edge(i, j, weight)

    def initialize_pheromones(self, initial_pheromone=1):
        """
        Inicjalizuje wartości feromonów na każdej krawędzi.
        """
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.adjacency_matrix[i][j] != 0:
                    self.pheromone_matrix[i][j] = initial_pheromone

    def display_pheromones(self):
        print("Macierz feromonów:")
        for row in self.pheromone_matrix:
            print(row)

    def update_pheromones(self, mappings, decay=0.5, pheromone_increase=1):
        """
        Aktualizuje feromony na podstawie dopasowań wierzchołków między dwoma grafami.
        
        Parametry:
        - mappings: lista dopasowań (każdy element jest listą par wierzchołków z grafów)
        - decay: współczynnik odparowywania feromonów (rho)
        - pheromone_increase: ilość feromonów dodanych do używanych dopasowań
        """
        # Odparowywanie feromonów
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.adjacency_matrix[i][j] != 0:
                    self.pheromone_matrix[i][j] *= (1 - decay)

        # Wzmocnienie feromonów na dopasowaniach
        for mapping in mappings:
            for (u, v) in mapping:
                self.pheromone_matrix[u][v] += pheromone_increase


    def evaluate_mapping(self, graph1, graph2, mapping):
        """
        Ocena jakości dopasowania wierzchołków między grafami.
        
        Parametry:
        - graph1, graph2: dwa grafy, które są porównywane
        - mapping: aktualne dopasowanie wierzchołków między grafami
        
        Zwraca:
        - ocena dopasowania
        """
        score = 0
        for u, v in mapping:
            # Porównujemy sąsiadów wierzchołków u i v w obu grafach
            for neighbor_u in range(graph1.num_vertices):
                if graph1.adjacency_matrix[u][neighbor_u] > 0:
                    mapped_neighbor_v = next((m[1] for m in mapping if m[0] == neighbor_u), None)
                    if mapped_neighbor_v is not None and graph2.adjacency_matrix[v][mapped_neighbor_v] == graph1.adjacency_matrix[u][neighbor_u]:
                        score += 1
        return score


    def visualize_mapping(self, graph1, graph2, mapping):
        """
        Wizualizuje dopasowania wierzchołków między dwoma grafami.
        
        Parametry:
        - graph1, graph2: dwa grafy do wizualizacji
        - mapping: lista dopasowań wierzchołków między grafami
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graf 1
        G1 = nx.Graph()
        for i in range(graph1.num_vertices):
            for j in range(i + 1, graph1.num_vertices):
                if graph1.adjacency_matrix[i][j] > 0:
                    G1.add_edge(i, j, weight=graph1.adjacency_matrix[i][j])

        # Ustalona pozycja dla Grafu 1 po lewej stronie
        pos1 = nx.spring_layout(G1, seed=42)
        for node in pos1:
            pos1[node] = (pos1[node][0] - 1.5, pos1[node][1])  # Shift Graf 1 to the left
        
        nx.draw(G1, pos1, with_labels=True, ax=ax, node_color='skyblue')
        edge_labels1 = nx.get_edge_attributes(G1, 'weight')
        nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_labels1, ax=ax)

        # Graf 2
        G2 = nx.Graph()
        for i in range(graph2.num_vertices):
            for j in range(i + 1, graph2.num_vertices):
                if graph2.adjacency_matrix[i][j] > 0:
                    G2.add_edge(i, j, weight=graph2.adjacency_matrix[i][j])

        # Ustalona pozycja dla Grafu 2 po prawej stronie
        pos2 = nx.spring_layout(G2, seed=42)
        for node in pos2:
            pos2[node] = (pos2[node][0] + 1.5, pos2[node][1])  # Shift Graf 2 to the right
        
        nx.draw(G2, pos2, with_labels=True, ax=ax, node_color='lightgreen')
        edge_labels2 = nx.get_edge_attributes(G2, 'weight')
        nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_labels2, ax=ax)

        # Rysujemy dopasowania wierzchołków między Grafem 1 i Grafem 2
        for u, v in mapping:
            if u in pos1 and v in pos2:  # Sprawdź, czy wierzchołki istnieją w pos1 i pos2
                pos_u, pos_v = pos1[u], pos2[v]
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k--')

        plt.title("Dopasowanie między Grafem 1 a Grafem 2")
        plt.show()


    def run_aco_for_isomorphism(self, graph1, graph2, num_ants=10, num_iterations=5, alpha=1, beta=1, decay=0.5, pheromone_increase=1):
        """
        Pełny cykl algorytmu mrówkowego dla dopasowania izomorficznego.
        
        Parametry:
        - graph1, graph2: dwa grafy do porównania
        - num_ants: liczba mrówek w każdej iteracji
        - num_iterations: liczba iteracji algorytmu
        - alpha: wpływ feromonów na wybór ścieżki
        - beta: wpływ wagi krawędzi na wybór ścieżki
        - decay: współczynnik odparowywania feromonów
        - pheromone_increase: ilość feromonów dodanych do używanych ścieżek
        """
        best_mapping, best_score = None, float('-inf')

        for iteration in range(num_iterations):
            print(f"\n--- Iteracja {iteration + 1} ---")
            all_mappings = []
            
            for ant in range(num_ants):
                mapping = []
                used_vertices_graph2 = set()
                
                # Mrówka losowo odwzorowuje wierzchołki z Grafu 1 na Graf 2
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
                print(f"Dopasowanie mrówki {ant + 1}: {mapping} (Ocena: {score})")

            self.update_pheromones([m[0] for m in all_mappings], decay, pheromone_increase)
            print(f"\nNajlepsze dopasowanie w iteracji {iteration + 1}: {best_mapping} (Ocena: {best_score})")

        self.visualize_mapping(graph1, graph2, best_mapping)
        return best_mapping, best_score


# Przykład użycia
graph1 = Graph(5)
graph1.generate_random_graph(edge_probability=0.5, max_weight=10)
graph1.initialize_pheromones()

graph2 = Graph(5)
graph2.generate_random_graph(edge_probability=0.5, max_weight=10)
graph2.initialize_pheromones()

# Uruchomienie algorytmu mrówkowego do znalezienia izomorfizmu
best_mapping, best_score = graph1.run_aco_for_isomorphism(graph1, graph2, num_ants=5, num_iterations=3, alpha=1, beta=1, decay=0.3, pheromone_increase=2)
print(f"\nNajlepsze dopasowanie izomorficzne: {best_mapping} (Ocena: {best_score})")
