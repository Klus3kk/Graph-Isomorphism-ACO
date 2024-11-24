import random
import json

# Funkcja przygotowująca ręczne dane testowe
def prepare_manual_test_cases():
    """
    Tworzy ręczne dane testowe i zapisuje je do plików JSON.
    Dane mogą być edytowane w plikach po ich zapisaniu.
    """
    test_cases = [
        {
            "num_vertices": 4,
            "adjacency_matrix": [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0]
            ]
        },
        {
            "num_vertices": 5,
            "adjacency_matrix": [
                [0, 2, 0, 1, 0],
                [2, 0, 3, 0, 4],
                [0, 3, 0, 5, 0],
                [1, 0, 5, 0, 6],
                [0, 4, 0, 6, 0]
            ]
        }
    ]

    for i, test_case in enumerate(test_cases):
        filename = f"manual_test_case_{i + 1}.json"
        with open(filename, "w") as f:
            json.dump(test_case, f, indent=4)
        print(f"Ręczny test zapisany jako {filename}")


# Funkcja generująca dane testowe
def generate_test_instances(n_instances, n_vertices, edge_probability=0.5, max_weight=10):
    """
    Generuje dane testowe w formie grafów i zapisuje je jako pliki JSON.

    Parametry:
    - n_instances: liczba instancji do wygenerowania
    - n_vertices: liczba wierzchołków w grafie
    - edge_probability: prawdopodobieństwo krawędzi między wierzchołkami
    - max_weight: maksymalna waga krawędzi
    """
    for i in range(n_instances):
        adjacency_matrix = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]

        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                if random.random() <= edge_probability:
                    weight = random.randint(1, max_weight)
                    adjacency_matrix[u][v] = weight
                    adjacency_matrix[v][u] = weight

        data = {
            "num_vertices": n_vertices,
            "adjacency_matrix": adjacency_matrix
        }

        filename = f"generated_graph_{i + 1}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Wygenerowano graf zapisany jako {filename}")


# Przykłady użycia
if __name__ == "__main__":
    # Punkt 2: Przygotowanie ręcznych danych testowych
    print("Tworzenie ręcznych instancji...")
    prepare_manual_test_cases()

    # Punkt 3: Generowanie danych testowych
    print("Generowanie instancji...")
    generate_test_instances(n_instances=3, n_vertices=6, edge_probability=0.5, max_weight=20)
