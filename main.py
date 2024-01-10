import networkx as nx
import matplotlib.pyplot as plt
from genetic import *

G = nx.Graph()

graph = [
    [0, 7, 6, 8, 3],
    [7, 0, 2, 4, 3],
    [6, 2, 0, 5, 6],
    [8, 4, 5, 0, 4],
    [3, 3, 6, 4, 0]
]
best_path = genetic_algorithm(graph, population_size=7, generations=15)
best_path_length = calculate_total_distance(best_path, graph)

print("\nBest found solution:", best_path)
print("Total distance's best solution:", best_path_length)

G.add_nodes_from(range(len(graph)))

path_nodes = best_path + [best_path[0]]
path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]

for i in range(len(graph)):
    for j in range(i + 1, len(graph[i])):
        weight = graph[i][j]
        if weight != 0:
            G.add_edge(i, j, weight=weight)

pos = nx.circular_layout(G) 
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10)

nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='skyblue', node_size=700)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='skyblue', width=2,arrowsize=20, connectionstyle='arc3,rad=0.1')


edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()