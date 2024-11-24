import random
import json

def generate_two_test_instances():
    """
    Generuje dwa różne grafy w formacie JSON na podstawie danych wejściowych od użytkownika.
    """
    try:
        n_vertices = int(input("Podaj liczbę wierzchołków w grafach: "))
        edge_probability = float(input("Podaj prawdopodobieństwo krawędzi (0-1): "))
        max_weight = int(input("Podaj maksymalną wagę krawędzi: "))
    except ValueError:
        print("Wprowadzono nieprawidłowe dane. Spróbuj ponownie.")
        return

    # Funkcja pomocnicza do generowania pojedynczego grafu
    def generate_graph():
        adjacency_matrix = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]
        for u in range(n_vertices):
            for v in range(u + 1, n_vertices):
                if random.random() <= edge_probability:
                    weight = random.randint(1, max_weight)
                    adjacency_matrix[u][v] = weight
                    adjacency_matrix[v][u] = weight
        return {
            "num_vertices": n_vertices,
            "adjacency_matrix": adjacency_matrix
        }

    # Generowanie dwóch grafów
    graph1 = generate_graph()
    graph2 = generate_graph()

    # Pobieranie nazw plików od użytkownika
    filename1 = input("Podaj nazwę pliku dla pierwszego grafu (np. graph1): ") + ".json"
    filename2 = input("Podaj nazwę pliku dla drugiego grafu (np. graph2): ") + ".json"

    # Zapisywanie grafów do plików JSON
    with open(filename1, "w") as f:
        json.dump(graph1, f, indent=4)
    print(f"Wygenerowano pierwszy graf zapisany jako {filename1}")

    with open(filename2, "w") as f:
        json.dump(graph2, f, indent=4)
    print(f"Wygenerowano drugi graf zapisany jako {filename2}")


if __name__ == "__main__":
    print("Generowanie dwóch grafów...")
    generate_two_test_instances()
