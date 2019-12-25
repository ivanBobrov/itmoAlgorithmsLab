import random
import math
import timeit
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap

seed = 186630
random.seed(seed)
graph: nx.Graph = nx.gnm_random_graph(100, 2000, seed=seed)
for (u, v, data) in graph.edges(data=True):
    data['weight'] = random.randint(0, 100)

edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())

plt.figure(dpi=300)
plt.box(False)
nx_pos = nx.spring_layout(graph, seed=seed)
cmap = plt.get_cmap("Greys")
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)
nx.draw(graph, nx_pos, node_size=10, with_labels=False,
        node_color='#f00000', edge_list=edges, edge_color=weights, edge_cmap=my_cmap)



def bellman_ford(graph: nx.Graph, start):
    vertex_mark = [math.inf for i in range(graph.number_of_nodes())]
    parent_node = [None for i in range(graph.number_of_nodes())]

    vertex_mark[start] = 0
    for _ in range(graph.number_of_nodes() - 1):
        for s, d, w in graph.edges.data('weight'):
            if vertex_mark[s] + w < vertex_mark[d]:
                vertex_mark[d] = vertex_mark[s] + w
                parent_node[d] = s
            if vertex_mark[d] + w < vertex_mark[s]:
                vertex_mark[s] = vertex_mark[d] + w
                parent_node[s] = d


def dijkstra(graph: nx.Graph, start):
    vertex_mark = [math.inf for i in range(graph.number_of_nodes())]
    parent_node = [None for i in range(graph.number_of_nodes())]
    unvisited = list(graph.nodes())

    vertex_mark[start] = 0
    current_node = start

    while True:
        for neighbor in graph.neighbors(current_node):
            if neighbor in unvisited:
                weight = graph.get_edge_data(current_node, neighbor)['weight']
                if vertex_mark[current_node] + weight < vertex_mark[neighbor]:
                    vertex_mark[neighbor] = vertex_mark[current_node] + weight
                    parent_node[neighbor] = current_node

        unvisited.remove(current_node)
        if len(unvisited) != 0:
            idx, node = min(enumerate(unvisited), key=lambda t: vertex_mark[t[1]])
            current_node = node
        else:
            return parent_node


print("Bellman_Ford")
#print(timeit.timeit(stmt='bellman_ford(graph, 5)',
#                    number=100, globals=globals()) / 100)
# bellman_ford(graph, 5)

print("Dijkstra")
#print(timeit.timeit(stmt='dijkstra(graph, 5)',
#                    number=100, globals=globals()) / 100)
dijkstra_pn = dijkstra(graph, 5)
edg = [(dijkstra_pn[n], dijkstra_pn[n + 1]) for n in range(len(dijkstra_pn) - 1)]
nx.draw(graph, nx_pos, with_labels=False, nodelist=[], width=3,
        edgelist=[(1,5), (5, 47), (47, 93), (93, 6)], edge_color="#f00000")
plt.show()

# Task 2
maze = nx.grid_graph(dim=[10, 10])
maze.remove_node((5, 0))
maze.remove_node((5, 1))
maze.remove_node((5, 2))
maze.remove_node((5, 3))

maze.remove_node((3, 0))

maze.remove_node((2, 2))
maze.remove_node((2, 3))
maze.remove_node((3, 3))
maze.remove_node((3, 4))

maze.remove_node((7, 2))
maze.remove_node((7, 3))
maze.remove_node((7, 4))
maze.remove_node((7, 5))

maze.remove_node((5, 5))
maze.remove_node((5, 6))

maze.remove_node((7, 7))
maze.remove_node((7, 8))

maze.remove_node((8, 9))
maze.remove_node((9, 9))

maze.remove_node((0, 7))
maze.remove_node((0, 8))
maze.remove_node((0, 9))
maze.remove_node((1, 7))
maze.remove_node((1, 8))
maze.remove_node((1, 9))
maze.remove_node((2, 8))
maze.remove_node((2, 9))
maze.remove_node((3, 8))
maze.remove_node((3, 9))
maze.remove_node((4, 9))

plt.figure(dpi=300)
plt.box(False)
nx_pos = nx.spring_layout(maze, seed=seed)
nx.draw(maze, nx_pos, node_size=10, with_labels=False,
        edge_color='#00000011', node_color='#a00000')
plt.show()


def a_star(graph: nx.Graph, start: (int, int), finish: (int, int)):
    if start == finish:
        print("Start is finish")
        return
    vertex_mark = {node: math.inf for node in graph.nodes()}
    parent_node = {node: None for node in graph.nodes()}
    unvisited = list(graph.nodes())

    vertex_mark[start] = 0
    current_node = start

    def h(s: (int, int), d: (int, int)):
        return math.sqrt((s[0] - d[0]) ** 2 + (s[1] - d[1]) ** 2)

    while True:
        for neighbor in graph.neighbors(current_node):
            if neighbor in unvisited:
                if vertex_mark[current_node] + 1 + h(current_node, finish) < vertex_mark[neighbor]:
                    vertex_mark[neighbor] = vertex_mark[current_node] + 1 + h(current_node, finish)
                    parent_node[neighbor] = current_node

        unvisited.remove(current_node)
        idx, node = min(enumerate(unvisited), key=lambda t: vertex_mark[t[1]])
        if node == finish:
            break
        else:
            current_node = node


print(timeit.timeit(stmt='a_star(maze, random.choice(list(maze.nodes())), '
                         'random.choice(list(maze.nodes())))',
                    number=100, globals=globals()) / 100)
