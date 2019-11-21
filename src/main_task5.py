import queue
import networkx as nx
import matplotlib.pyplot as plt

print("Generating random graph")
graph = nx.gnm_random_graph(100, 2000, 84640)

plt.figure(dpi=300)
plt.box(False)
nx_pos = nx.spring_layout(graph, seed = 184673)
nx.draw(graph, nx_pos, node_size=30, with_labels=False,
        edge_color='#00000006', node_color='#a00000')
plt.show()


def find_path_bfs(graph, node_from, node_to):
    nodes_queue = queue.Queue()
    nodes_queue.put(node_from)

    node_parents = [None] * graph.number_of_nodes()
    while not nodes_queue.empty():
        node = nodes_queue.get()
        for neighbor in graph.neighbors(node):
            node_parents[neighbor] = node
            nodes_queue.put(neighbor)

            if neighbor == node_to:
                result = [node_to]
                parent = node
                while not parent == node_from:
                    result.append(parent)
                    parent = node_parents[parent]
                result.append(node_from)
                result.reverse()
                return result

    return []


print("Breadth first algorithm. Searching path")
bfs_path = find_path_bfs(graph, 1, 5)
bfs_edges = [(bfs_path[n], bfs_path[n + 1]) for n in range(len(bfs_path) - 1)]
print("Resulting path: ", bfs_path)

plt.figure(dpi=300)
plt.box(False)
nx.draw(graph, nx_pos, node_size=30, with_labels=False,
        edge_color='#00000006', node_color='#888888')
nx.draw(graph, nx_pos, node_size=30, with_labels=False,
        edge_color='#a00000', node_color='#a00000',
        nodelist=bfs_path, edgelist=bfs_edges)
plt.show()


def dfs_connected(graph, starting_node):
    stack = [starting_node]
    visited = [False] * graph.number_of_nodes()

    def not_visited_neighbors_count(node):
        count = 0
        for n in graph.neighbors(node):
            if not visited[n]:
                count += 1
        return count

    def get_next_not_visited(node):
        for n in graph.neighbors(node):
            if not visited[n]:
                return n

    node = starting_node
    visited[node] = True
    while len(stack) != 0:
        if not_visited_neighbors_count(node) != 0:
            node = get_next_not_visited(node)
            visited[node] = True
            stack.append(node)
            continue

        stack.pop()
        if len(stack) == 0:
            break

        node = stack[-1]

    return visited


print("Depth fisrt search. Searching for connected components in sparse graph")
sparse_graph = nx.gnm_random_graph(100, 60, 84640)
visited = dfs_connected(sparse_graph, 3)
visited_nodes = []
for i in range(len(visited)):
    if visited[i]:
        visited_nodes.append(i)
print("Graph connected component nodes: ", visited_nodes)

plt.figure(dpi=300)
plt.box(False)
nx_sparse_pos = nx.spring_layout(sparse_graph, seed = 13432)
nx.draw(sparse_graph, nx_sparse_pos, node_size=30, with_labels=False,
        edge_color='#00000060', node_color='#888888')
nx.draw(sparse_graph, nx_sparse_pos, node_size=30, with_labels=False,
        edge_color='#a00000', node_color='#a00000',
        nodelist=visited_nodes, edgelist=[])
plt.show()

