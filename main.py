import random
import time


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

    def display(self):
        # Wyświetla macierz sąsiedztwa
        for row in self.adjacency_matrix:
            print(row)

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


    # Ant Colony Algorithm
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

    def update_pheromones(self, paths, decay=0.5, pheromone_increase=1):
        """
        Aktualizuje feromony na podstawie ścieżek przejścia mrówek.
        
        Parametry:
        - paths: lista ścieżek przejścia mrówek (każda ścieżka to lista wierzchołków)
        - decay: współczynnik odparowywania feromonów (rho)
        - pheromone_increase: ilość feromonów dodanych do używanych krawędzi
        """
        # Odparowywanie feromonów
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if self.adjacency_matrix[i][j] != 0:
                    self.pheromone_matrix[i][j] *= (1 - decay)

        # Wzmocnienie feromonów na ścieżkach użytych przez mrówki
        for path in paths:
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                self.pheromone_matrix[u][v] += pheromone_increase
                self.pheromone_matrix[v][u] += pheromone_increase  # Graf nieskierowany

    def ant_move(self, current_vertex, alpha=1, beta=1):
        """
        Funkcja wykonująca ruch mrówki z jednego wierzchołka do sąsiedniego,
        wybierając kolejny wierzchołek na podstawie feromonów i wag.
        
        Parametry:
        - current_vertex: obecny wierzchołek mrówki
        - alpha: wpływ feromonów na wybór
        - beta: wpływ wagi krawędzi na wybór
        """
        neighbors = [j for j in range(self.num_vertices) if self.adjacency_matrix[current_vertex][j] > 0]
        if not neighbors:
            return None  # Brak sąsiadów, ruch niemożliwy

        # Obliczamy prawdopodobieństwa dla każdego sąsiada
        probabilities = []
        for neighbor in neighbors:
            pheromone = self.pheromone_matrix[current_vertex][neighbor] ** alpha
            weight = self.adjacency_matrix[current_vertex][neighbor] ** beta
            probabilities.append(pheromone * weight)

        # Normalizujemy prawdopodobieństwa
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Wybieramy kolejny wierzchołek na podstawie prawdopodobieństwa
        next_vertex = random.choices(neighbors, weights=probabilities, k=1)[0]
        return next_vertex
    
    def display_info(self, paths=None):
        """
        Wyświetla wszystkie istotne informacje o stanie programu.
        """
        print("\n=== Informacje o programie ===")
        # Wyświetlamy graf nieskierowany
        print("Macierz sąsiedztwa (graf nieskierowany):")
        for row in self.adjacency_matrix:
            print(row)
        
        # Wyświetlamy macierz feromonów
        print("\nMacierz feromonów:")
        for row in self.pheromone_matrix:
            print(row)
        
        # Wyświetlamy ścieżki użyte przez mrówki w ostatniej iteracji, jeśli są dostępne
        if paths:
            print("\nŚcieżki przejścia mrówek w ostatniej iteracji:")
            for path in paths:
                print(" -> ".join(map(str, path)))

        # Wyświetlamy czas wykonania
        elapsed_time = time.time() - self.start_time
        print(f"\nCzas wykonania programu: {elapsed_time:.2f} sekund")
    
    


# Przykład użycia
g = Graph(20)
g.generate_random_graph(edge_probability=0.5, max_weight=100)
g.initialize_pheromones()

# Przykładowe ścieżki przejścia mrówek (symulacja)
paths = [
    [0, 1, 4, 2],
    [1, 3, 0, 4],
    [2, 3, 1]
]

# Aktualizacja feromonów i wyświetlenie informacji
g.update_pheromones(paths, decay=0.3, pheromone_increase=2)
g.display_info(paths=paths)

