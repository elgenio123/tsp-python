# Travelling Salesman Problem (TSP) Solver

## Overview
This Python implementation solves the Travelling Salesman Problem (TSP), which is a classic optimization problem. The goal is to find the shortest possible route that visits a set of cities and returns to the origin city.

## Features
- **Genetic Algorithm:** The solution is based on a genetic algorithm, a powerful optimization technique inspired by natural selection.

- **Matrix Representation:** The cities and distances are represented using an adjacency matrix, providing an efficient way to model the TSP.

- **Population Initialization:** The initial population of solutions is generated intelligently using [explain how you generate the initial population].

- **Crossover and Mutation:** Genetic operators such as crossover and mutation are applied to evolve the population towards better solutions. Mutqtion operqtion in this program is done AI guarded using tensorflow and keras

## Getting Started
### Prerequisites
- Python 3.x

### Installation
1. Clone the repository.
   ```bash
   git clone https://github.com/elgenio123/tsp-python.git
   cd tsp-python
   pip install -r requirements.txt
### Usage
The code contains inside the main an adjacent matrix that represents a graph
Edit this matrix based on your own graph
