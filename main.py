import random

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        # Tworzymy macierz sąsiedztwa wypełnioną zerami (brak krawędzi)
        self.adjacency_matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]

    def add_edge(self, u, v, weight):
        # Dodajemy krawędź z wagą (dla grafu nieskierowanego)
        self.adjacency_matrix[u][v] = weight
        self.adjacency_matrix[v][u] = weight

    def display(self):
        # Wyświetla macierz sąsiedztwa
        for row in self.adjacency_matrix:
            print(row)

# Przykład użycia
g = Graph(5)
g.add_edge(0, 1, 4)
g.add_edge(1, 2, 5)
g.add_edge(2, 3, 2)
g.add_edge(3, 4, 1)
g.add_edge(4, 0, 3)
g.display()
