import numpy as np
from random import shuffle

# Supposons que num_cities soit le nombre total de villes
# num_cities = 10

# Fonction pour générer une solution initiale
def generate_initial_solution(num_cities):
    return list(range(num_cities))

# Fonction de mutation simple (échange aléatoire de deux villes)
def mutate_solution(solution):
    mutated_solution = solution.copy()
    idx1, idx2 = np.random.choice(len(solution), size=2, replace=False)
    mutated_solution[idx1], mutated_solution[idx2] = mutated_solution[idx2], mutated_solution[idx1]
    return mutated_solution

# Génération d'exemples d'entraînement
def generate_data(num_cities, num_examples):

    X_train = []
    y_train = []

    for _ in range(num_examples):
        initial_solution = generate_initial_solution(num_cities)
        mutated_solution = mutate_solution(initial_solution)

        X_train.append(initial_solution)
        y_train.append(mutated_solution)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train
