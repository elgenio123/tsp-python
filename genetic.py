import numpy as np
import random
from ai_model import *

# Function to calculate total distance of a path in a complete graph
def calculate_total_distance(path, graph):
    total_distance = 0
    for i in range(len(path)-1):
        total_distance += graph[path[i]][path[i+1]]
    total_distance += graph[path[-1]][path[0]]  # Back to first town
    return total_distance

# Selection function
def selection(population, graph):
    selected = []
    for _ in range(len(population)):
        # Use random.sample to select individuals without replacement
        tournament = random.sample(population, k=3)
        best_chromosome = min(tournament, key=lambda x: calculate_total_distance(x, graph))
        selected.append(best_chromosome)
    return selected
# Crossover function (Order Crossover)
def crossover(parent1, parent2):
    
    size = len(parent1)
    start, end = sorted(np.random.choice(size, 2, replace=False))

    # Initialize child with a copy of the segment from parent1
    child = parent1[start:end].copy()

    # Fill the remaining positions from parent2, avoiding duplicates
    parent2_idx = 0
    for i in range(size):
        if i < start or i >= end:
            gene = parent2[parent2_idx]
            while gene in child:
                parent2_idx += 1
                gene = parent2[parent2_idx]
            child.append(gene)

    return child


# # Mutation function(Swap)
def mutation(original_sequence,predicted_mutation):
        
        # Select the index with the highest probability
        max_prob_index = np.argmax(predicted_mutation)
        mutation_index = max_prob_index
        # print("Mutation index: {}".format(mutation_index))

        mutated_sequence = original_sequence.copy()
    
        sorted_indices = np.argsort(predicted_mutation[0])[::-1]

        # Select the second index
        second_index = sorted_indices[1]
        # print("Second Mutation index: {}".format(second_index))
        
        # Perform the swap
        mutated_sequence[mutation_index], mutated_sequence[second_index] = (
            mutated_sequence[second_index],
            mutated_sequence[mutation_index]
        )

        return mutated_sequence

# Generate random list
def generate_random_lists(n, num_lists):
    all_lists = []

    for _ in range(num_lists):
        random_list = random.sample(range(n), n)
        all_lists.append(random_list)

    return all_lists

# Genetic algorithm
def genetic_algorithm(graph, population_size, generations):
    # Population initialization
    population = generate_random_lists(len(graph[0]), population_size)
    model = create_model(sequence_length=len(graph[0]), num_genes=len(graph[0]), num_features=1)
    for generation in range(generations):
        # Selection
        selected_population = selection(population, graph)

        # Crossover
        new_population = []
        for i in range(0, len(selected_population)-1, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([child1, child2])

        # Mutation
        for i in range(len(new_population)):
            encoded_solution = np.array([new_population[i]])
            predicted_mutation = model.predict(encoded_solution)
            new_population[i] = mutation(new_population[i], predicted_mutation=predicted_mutation)

        # Fitness evaluation
        population = sorted(new_population, key=lambda x: calculate_total_distance(x, graph))

        # Display best solution for each generation
        best_solution = population[0]
        best_distance = calculate_total_distance(best_solution, graph)
        print(f"Generation {generation + 1}: Best solution = {best_solution}, Total distance = {best_distance}")

    return population[0]
