# Graph-Isomorphism-ACO

This project addresses the **Graph Isomorphism Problem**, which involves determining whether two graphs are structurally identical (isomorphic). The project implements three different algorithms to solve the problem:

1. **Greedy Algorithm** - A heuristic-based approach that attempts to find isomorphisms quickly.
2. **Brute Force Algorithm** - Exhaustively explores all possible vertex permutations for exact solutions.
3. **Ant Colony Optimization (ACO)** - A metaheuristic inspired by the behavior of ants to find approximate solutions efficiently.

## Features

- Generate random graphs for testing with customizable parameters (number of vertices, edge probabilities, and weights).
- Load graphs from JSON files for reproducibility and external data.
- Benchmark and compare the performance of the algorithms in terms of runtime and solution quality.
- Visualize graph mappings and connections using **NetworkX**.

## Usage

The project supports two modes: **graph generation** and **loading graphs from JSON files**.

### Running the Benchmark

To run benchmarks for graphs with vertex counts ranging from 5 to 10 (by default):

```bash
python benchmark.py
```

This will:

- Generate random graphs dynamically.
- Compare the execution time and quality of all three algorithms.
- Save plots for runtime (`algorithm_time_comparison_log.png`) and solution quality (`algorithm_quality_comparison.png`) in the `results/` folder.

### Running the Main Program

The main program supports custom configurations for graph input:

#### 1. Generate Random Graphs

To generate random graphs and test the algorithms:

```bash
python main.py --generate --vertices <num_vertices> --edge_probability <prob> --max_weight <weight>
```

##### Example of usage with random graphs

```bash
python main.py --generate --vertices 5 --edge_probability 0.5 --max_weight 10
```

#### 2. Load Graphs from JSON

To load graphs from JSON files for testing:

```bash
python main.py --file1 <path_to_graph1.json> --file2 <path_to_graph2.json>
```

##### Example of usage with files

```bash
python main.py --file1 data/graph1.json --file2 data/graph2.json
```

## Output Files

- **Results**:
  - `results/results.json`: Contains mappings, scores, and execution times for all three algorithms.
  - Individual mapping files:
    - `results/greedy_mapping.json`
    - `results/brute_force_mapping.json`
    - `results/aco_mapping.json`

- **Plots**:
  - `results/algorithm_time_comparison_log.png`: Runtime comparison of algorithms.
  - `results/algorithm_quality_comparison.png`: Quality comparison of solutions.

## How It Works

### Algorithms

1. **Greedy Algorithm**:
   - Matches vertices in one graph to another based on the highest connectivity.
   - Fast but not guaranteed to find the optimal solution.

2. **Brute Force**:
   - Tests all permutations of vertex mappings to find the exact isomorphism.
   - Guarantees correctness but scales poorly with larger graphs.

3. **Ant Colony Optimization (ACO)**:
   - Uses pheromones to guide a population of ants to explore promising mappings.
   - Balances exploration and exploitation to approximate solutions effectively.

### Graph Representation

- Adjacency matrices represent the graphs.
- Weights on edges indicate connections between vertices.

## Examples

### Benchmark Output Example

When running `benchmark.py`, you will see console outputs like:

```bash
=== Testowanie dla 5 wierzchołków ===
Uruchamianie algorytmu zachłannego (Greedy)...
Greedy: czas = 0.000002s, jakość = 0.900000
Uruchamianie algorytmu brute-force...
BruteForce: czas = 0.012048s, jakość = 1.000000
Uruchamianie algorytmu mrówkowego (ACO)...
ACO: czas = 0.004231s, jakość = 0.950000
```

## Install Dependencies

Before running the project, ensure you have the necessary Python dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
