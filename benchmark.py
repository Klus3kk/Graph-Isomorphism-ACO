import time
import matplotlib.pyplot as plt
from main import Graph

def compare_algorithms_and_quality():
    vertex_counts = [5, 10, 15, 20]  # Liczba wierzchołków do testowania
    greedy_times = []
    brute_force_times = []
    aco_times = []
    greedy_qualities = []
    brute_force_qualities = []
    aco_qualities = []

    for num_vertices in vertex_counts:
        print(f"Testowanie dla {num_vertices} wierzchołków...")
        graph1 = Graph(num_vertices)
        graph1.generate_random_graph(edge_probability=0.5, max_weight=20)
        graph1.initialize_pheromones()

        graph2 = Graph(num_vertices)
        graph2.generate_random_graph(edge_probability=0.5, max_weight=20)
        graph2.initialize_pheromones()

        # Algorytm zachłanny
        start_time = time.time()
        greedy_mapping = graph1.greedy_isomorphism(graph2)
        greedy_time = time.time() - start_time
        greedy_score = graph1.evaluate_mapping(graph1, graph2, greedy_mapping)

        greedy_times.append(greedy_time)
        greedy_qualities.append(greedy_score / (num_vertices * (num_vertices - 1)))  # Normalizowana jakość

        # Algorytm brute-force (dla mniejszych grafów)
        if num_vertices <= 10:
            start_time = time.time()
            brute_force_mapping, brute_force_score = graph1.brute_force_isomorphism(graph2)
            brute_force_time = time.time() - start_time

            brute_force_times.append(brute_force_time)
            brute_force_qualities.append(brute_force_score / (num_vertices * (num_vertices - 1)))  # Normalizowana jakość
        else:
            brute_force_times.append(None)
            brute_force_qualities.append(None)

        # Algorytm mrówkowy (ACO)
        start_time = time.time()
        aco_mapping, aco_score = graph1.run_aco_for_isomorphism(graph1, graph2, num_ants=5, num_iterations=5)
        aco_time = time.time() - start_time

        aco_times.append(aco_time)
        aco_qualities.append(aco_score / (num_vertices * (num_vertices - 1)))  # Normalizowana jakość

    # Generowanie wykresów
    create_time_plot(vertex_counts, greedy_times, brute_force_times, aco_times)
    create_quality_plot(vertex_counts, greedy_qualities, brute_force_qualities, aco_qualities)

def create_time_plot(vertex_counts, greedy_times, brute_force_times, aco_times):
    plt.figure(figsize=(10, 6))
    plt.plot(vertex_counts, greedy_times, label="Greedy", marker='o')
    plt.plot(vertex_counts, [t if t is not None else 0 for t in brute_force_times], label="Brute Force", marker='o')
    plt.plot(vertex_counts, aco_times, label="ACO", marker='o')
    plt.yscale('log')  # Logarithmic scale for better visibility of large differences
    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Czas wykonania (s) [log scale]")
    plt.title("Zależność czasu wykonania algorytmów od liczby wierzchołków")
    plt.legend()
    plt.grid()
    plt.savefig("algorithm_time_comparison_log.png")
    plt.show()

def create_quality_plot(vertex_counts, greedy_qualities, brute_force_qualities, aco_qualities):
    plt.figure(figsize=(10, 6))
    plt.plot(vertex_counts, greedy_qualities, label="Greedy Quality", marker='o')
    plt.plot(vertex_counts, [q if q is not None else 0 for q in brute_force_qualities], label="Brute Force Quality", marker='o')
    plt.plot(vertex_counts, aco_qualities, label="ACO Quality", marker='o')
    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Jakość rozwiązania (normalizowana)")
    plt.title("Zależność jakości rozwiązań od liczby wierzchołków")
    plt.legend()
    plt.grid()
    plt.savefig("algorithm_quality_comparison.png")
    plt.show()

if __name__ == "__main__":
    compare_algorithms_and_quality()
