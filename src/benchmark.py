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
    vertex_counts = range(5, 11)  # Liczba wierzchołków do testowania (5 do 10 włącznie)
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
        # print("Uruchamianie algorytmu zachłannego (Greedy)...")
        start_time = time.perf_counter()
        greedy_mapping = graph1.greedy_isomorphism(graph2)
        greedy_time = time.perf_counter() - start_time
        greedy_score = graph1.evaluate_mapping(graph1, graph2, greedy_mapping)
        greedy_times.append(greedy_time)
        greedy_qualities.append(greedy_score / (num_vertices * (num_vertices - 1)))  # Normalizowana jakość
        print(f"Greedy: czas = {greedy_time:.6f}s, jakość = {greedy_qualities[-1]:.6f}")

        # Algorytm brute-force 
        if num_vertices <= 30:  
            # print("Uruchamianie algorytmu brute-force...")
            start_time = time.perf_counter()
            brute_force_mapping, brute_force_score = graph1.brute_force_isomorphism(graph2)
            brute_force_time = time.perf_counter() - start_time
            brute_force_times.append(brute_force_time)
            brute_force_qualities.append(brute_force_score / (num_vertices * (num_vertices - 1)))  # Normalizowana jakość
            print(f"BruteForce: czas = {brute_force_time:.6f}s, jakość = {brute_force_qualities[-1]:.6f}")
        else:
            brute_force_times.append(None)
            brute_force_qualities.append(None)
            print("BruteForce: pominięte (zbyt duży graf).")

        # Algorytm mrówkowy (ACO)
        # print("Uruchamianie algorytmu mrówkowego (ACO)...")
        start_time = time.perf_counter()
        aco_mapping, aco_score = graph1.run_aco_for_isomorphism(
            graph1, graph2, num_ants=30, num_iterations=100, alpha=1, beta=2, decay=0.1, pheromone_increase=5
        )
        aco_time = time.perf_counter() - start_time
        aco_times.append(aco_time)
        aco_qualities.append(aco_score / (num_vertices * (num_vertices - 1)))  # Normalizowana jakość
        print(f"ACO: czas = {aco_time:.6f}s, jakość = {aco_qualities[-1]:.6f}")

    # Generowanie wykresów
    create_time_plot(vertex_counts, greedy_times, brute_force_times, aco_times)
    create_quality_plot(vertex_counts, greedy_qualities, brute_force_qualities, aco_qualities)


def create_time_plot(vertex_counts, greedy_times, brute_force_times, aco_times):
    plt.figure(figsize=(10, 6))
    brute_force_cleaned = [t if t is not None else float('inf') for t in brute_force_times]


    plt.plot(vertex_counts, greedy_times, label="Greedy", marker="o", linestyle="-")
    plt.plot(vertex_counts, brute_force_cleaned, label="Brute Force", marker="o", linestyle="--")
    plt.plot(vertex_counts, aco_times, label="ACO", marker="o", linestyle="-")
    plt.yscale("log")  # Logarytmiczna skala
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
    brute_force_cleaned = [q if q is not None else 0 for q in brute_force_qualities]

    plt.plot(vertex_counts, greedy_qualities, label="Greedy Quality", marker="o", linestyle="-")
    plt.plot(vertex_counts, brute_force_cleaned, label="Brute Force Quality", marker="o", linestyle="--")
    plt.plot(vertex_counts, aco_qualities, label="ACO Quality", marker="o", linestyle="-")
    plt.xlabel("Liczba wierzchołków")
    plt.ylabel("Jakość rozwiązania (normalizowana)")
    plt.title("Zależność jakości rozwiązań od liczby wierzchołków")
    plt.legend()
    plt.grid()
    filepath = os.path.join(output_folder, "algorithm_quality_comparison.png")
    plt.savefig(filepath)
    plt.show()


if __name__ == "__main__":
    compare_algorithms_and_quality()
