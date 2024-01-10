import networkx as nx
import matplotlib.pyplot as plt


def greedy_coloring(graph):
    colors = {}  # Dictionnaire pour stocker les couleurs attribuées à chaque nœud

    for node in graph:
        # Trouver les couleurs déjà utilisées par les voisins
        neighbor_colors = set(colors.get(neighbor) for neighbor in graph[node] if neighbor in colors)

        # Trouver la première couleur disponible
        color = 0
        while color in neighbor_colors:
            color += 1

        colors[node] = color

    return colors

exemple_graphe = {
    'A': ['C'],
    'B': ['C', 'D', 'E'],
    'C': ['A', 'B', 'D', 'E'],
    'D': ['B', 'C' ],
    'E': ['B', 'C' ]
}


resultat_coloriage = greedy_coloring(exemple_graphe)
print(resultat_coloriage)


G = nx.Graph(exemple_graphe)

node_colors = [resultat_coloriage[node] for node in G.nodes]


pos = nx.spring_layout(G) 
nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow, font_color='white')
plt.show()