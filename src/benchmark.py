import time
import matplotlib.pyplot as plt
from main import Graph
import os

output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(output_folder, exist_ok=True)

os.environ["OMP_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def compare_algorithms_and_quality():
    vertex_counts = range(5, 11)  # Liczba wierzchołków do testowania
    greedy_times = []
    brute_force_times = []
    aco_times = []
    greedy_qualities = []
    brute_force_qualities = []
    aco_qualities = []

    for num_vertices in vertex_counts:
        print(f"\n=== Testowanie dla {num_vertices} wierzchołków ===")

        # Generowanie grafów w locie
        graph1 = Graph(num_vertices)
        graph1.generate_random_graph(edge_probability=0.8, max_weight=20)
        graph1.initialize_pheromones()

        graph2 = Graph(num_vertices)
        graph2.generate_random_graph(edge_probability=0.8, max_weight=20)
        graph2.initialize_pheromones()

        # Algorytm zachłanny (Greedy)
        start_time = time.perf_counter()
        greedy_mapping = graph1.greedy_isomorphism(graph2)
        greedy_time = time.perf_counter() - start_time
        greedy_score = graph1.evaluate_mapping(graph1, graph2, greedy_mapping)
        greedy_times.append(greedy_time)
        greedy_qualities.append(greedy_score / max(1, (num_vertices * (num_vertices - 1))))  # Normalizowana jakość
        print(f"Greedy: czas = {greedy_time:.6f}s, jakość = {greedy_qualities[-1]:.6f}")

        # Algorytm brute-force
        start_time = time.perf_counter()
        brute_force_mapping, brute_force_score = graph1.brute_force_isomorphism(graph2)
        brute_force_time = time.perf_counter() - start_time
        brute_force_times.append(brute_force_time)
        brute_force_qualities.append(brute_force_score / max(1, (num_vertices * (num_vertices - 1))))
        print(f"BruteForce: czas = {brute_force_time:.6f}s, jakość = {brute_force_qualities[-1]:.6f}")

        # Algorytm mrówkowy (ACO)
        start_time = time.perf_counter()
        aco_mapping, aco_score = graph1.run_aco_for_isomorphism(
            graph1, graph2, num_ants=10, num_iterations=10, alpha=1, beta=2, decay=0.1, pheromone_increase=5
        )
        aco_time = time.perf_counter() - start_time
        aco_times.append(aco_time)
        aco_qualities.append(aco_score / max(1, (num_vertices * (num_vertices - 1))))
        print(f"ACO: czas = {aco_time:.6f}s, jakość = {aco_qualities[-1]:.6f}")

    # Generowanie wykresów
    create_time_plot(vertex_counts, greedy_times, brute_force_times, aco_times)
    create_quality_plot(vertex_counts, greedy_qualities, brute_force_qualities, aco_qualities)
    create_differences_plot(vertex_counts, greedy_qualities, brute_force_qualities, aco_qualities)
    create_time_vs_quality_plot(greedy_times, greedy_qualities, brute_force_times, brute_force_qualities, aco_times, aco_qualities)

def create_time_plot(vertex_counts, greedy_times, brute_force_times, aco_times):
    plt.figure(figsize=(10, 6))

    plt.plot(vertex_counts, greedy_times, label="Greedy", marker="o", linestyle="-")
    plt.plot(vertex_counts, brute_force_times, label="Brute Force", marker="o", linestyle="--")
    plt.plot(vertex_counts, aco_times, label="ACO", marker="o", linestyle="-")
    plt.yscale("log")  
    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Czas wykonania (s) [log scale]")
    plt.title("Zależność czasu wykonania algorytmów od liczby wierzchołków")
    plt.legend()
    plt.grid()
    filepath = os.path.join(output_folder, "algorithm_time_comparison.png")
    plt.savefig(filepath)
    plt.show()

def create_quality_plot(vertex_counts, greedy_qualities, brute_force_qualities, aco_qualities):
    plt.figure(figsize=(10, 6))

    plt.plot(vertex_counts, greedy_qualities, label="Greedy Quality", marker="o", linestyle="-")
    plt.plot(vertex_counts, brute_force_qualities, label="Brute Force Quality", marker="o", linestyle="--")
    plt.plot(vertex_counts, aco_qualities, label="ACO Quality", marker="o", linestyle="-")
    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Jakość rozwiązania (normalizowana)")
    plt.title("Zależność jakości rozwiązań od liczby wierzchołków")
    plt.legend()
    plt.grid()
    filepath = os.path.join(output_folder, "algorithm_quality_comparison.png")
    plt.savefig(filepath)
    plt.show()

def create_differences_plot(vertex_counts, greedy_qualities, brute_force_qualities, aco_qualities):
    plt.figure(figsize=(10, 6))

    plt.plot(vertex_counts, [aco - greedy for aco, greedy in zip(aco_qualities, greedy_qualities)], label="ACO - Greedy", marker="o")
    plt.plot(vertex_counts, [aco - brute for aco, brute in zip(aco_qualities, brute_force_qualities)], label="ACO - Brute Force", marker="o")
    plt.plot(vertex_counts, [greedy - brute for greedy, brute in zip(greedy_qualities, brute_force_qualities)], label="Greedy - Brute Force", marker="o")

    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Różnica w jakości")
    plt.title("Różnice w jakości rozwiązań między algorytmami")
    plt.legend()
    plt.grid()
    filepath = os.path.join(output_folder, "algorithm_quality_differences.png")
    plt.savefig(filepath)
    plt.show()

def create_time_vs_quality_plot(greedy_times, greedy_qualities, brute_force_times, brute_force_qualities, aco_times, aco_qualities):
    plt.figure(figsize=(10, 6))

    plt.scatter(greedy_times, greedy_qualities, label="Greedy", color="blue", marker="o")
    plt.scatter(brute_force_times, brute_force_qualities, label="Brute Force", color="green", marker="s")
    plt.scatter(aco_times, aco_qualities, label="ACO", color="red", marker="^")

    plt.xscale("log")
    plt.xlabel("Czas wykonania (s) [log scale]")
    plt.ylabel("Jakość rozwiązania (normalizowana)")
    plt.title("Zależność jakości rozwiązań od czasu wykonania")
    plt.legend()
    plt.grid()
    filepath = os.path.join(output_folder, "time_vs_quality_comparison.png")
    plt.savefig(filepath)
    plt.show()

if __name__ == "__main__":
    compare_algorithms_and_quality()
