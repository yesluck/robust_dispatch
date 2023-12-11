import numpy as np
import osmnx as ox
import networkx as nx
import pickle
import cvxpy as cp
import time
import heapq
from collections import Counter
import matplotlib.pyplot as plt



def k_shortest_paths(G, start, end, k=5):
    res = []

    print(f"start={start}, end={end}")

    if not G.has_node(start) or not G.has_node(end):
        print("Invalid start or end node.")
        return

    start_time = time.time()
    P = []  # final set of paths
    count = {node: 0 for node in G.nodes()}
    B = [(0, [start])]  # heap with cost and path, initialize with the path from the start node

    while B and len(P) < k:
        C, pu = heapq.heappop(B)  # get the shortest path
        u = pu[-1]  # the last node in the path
        count[u] += 1

        if u == end:
            P.append(pu)

        if count[u] < k:
            for v in G.neighbors(u):
                if v not in pu:  # ensure no cycle
                    pv = list(pu)
                    pv.append(v)
                    cost = C + G[u][v][0]["length"]
                    heapq.heappush(B, (cost, pv))

    end_time = time.time()

    for path in P:
        # Convert path of nodes to path of edges
        edge_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        distance = sum(G[path[i]][path[i + 1]][0]["length"] for i in range(len(path) - 1))
        print(f"Path: {edge_path}, Distance: {distance}")
        # res.append((edge_path, distance))
        nodes = []
        for edge in edge_path:
            nodes.append(edge[0])
        nodes.append(edge_path[-1][1])
        res.append(nodes)

    print(f"Finding k shortest paths used: {round(end_time - start_time, 6)}s")
    print()

    return res


def k_shortest_paths_no_overlap(G, start, end, k=5):
    G = G.copy()
    res = []
    print(f"start={start}, end={end}")

    if not G.has_node(start) or not G.has_node(end):
        print("Invalid start or end node.")
        return

    # Start the timer
    start_time = time.time()

    for i in range(k):
        try:
            # Find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='length')
            # Store the path
            res.append(path)
            # Remove the edges from the graph
            for u, v in zip(path[:-1], path[1:]):
                threshold = 3
                if len(set(G.neighbors(u))) > threshold:  # Check if there are multiple edges between u and v
                    try:
                        G.remove_edge(u, v)
                    except:
                        print("Not in graph")
                    try:
                        G.remove_edge(v, u)
                    except:
                        print("Not in graph")
            # Visualize new using networkx
            print(len(G.edges))
        except nx.NetworkXNoPath:
            print(f"No more paths found after {i} iterations.")
            break

    # Stop the timer
    end_time = time.time()

    # Print the time taken
    print(f"Finding k shortest paths used: {round(end_time - start_time, 6)}s")
    print()

    return res


def load_graph(prefix):

    # Read the graph from the file
    with open(f"{prefix}_graph.pkl", "rb") as f:
        graph = pickle.load(f)

    print(len(graph.edges))

    # Dictionary to hold the count of neighbors for each node
    neighbor_count = {}

    for node in graph.nodes():
        neighbor_count[node] = len(set(graph.neighbors(node)))

    # Count the occurrences of each neighbor count
    count_stats = Counter(neighbor_count.values())

    # Now count_stats holds the information about how many nodes have a particular neighbor count
    print(count_stats)

    num_edges = len(graph.edges)

    # initiate c0 and edge_indices
    c0 = np.zeros(num_edges)
    for i, edge in enumerate(graph.edges(data=True)):
        c0[i] = graph[edge[0]][edge[1]][0]["length"]
    edge_indices = {(edge[0], edge[1]): index for index, edge in enumerate(graph.edges)}
    edges = {index: (edge[0], edge[1]) for index, edge in enumerate(graph.edges)}

    return graph, c0, edge_indices, edges


def get_nearest_node(graph, lat, lon):
    return ox.distance.nearest_nodes(graph, X=lon, Y=lat)


def get_shortest_paths(graph, start_lat, start_lon, end_lat, end_lon, c0, edge_indices, prefix, suffix="x", k=3):
    start_node = get_nearest_node(graph, start_lat, start_lon)
    end_node = get_nearest_node(graph, end_lat, end_lon)

    k_shortest_path = k_shortest_paths_no_overlap(graph, start_node, end_node, k=3)

    fig, ax = ox.plot_graph(graph, figsize=(50, 50), show=False, close=False,
                            node_color='blue', node_size=15, edge_color='black', bgcolor='white', dpi=1000)
    colors = ['red', 'green', 'purple']  # Different colors for different paths
    for i in range(min(k, len(k_shortest_path))):
        ox.plot_graph_route(graph, k_shortest_path[i], ax=ax, route_linewidth=6, route_color=colors[i % len(colors)],
                            orig_dest_size=50, node_size=15, edge_color='black', bgcolor='white')
    fig.savefig(f"{prefix}_k_shortest_paths_{suffix}.png")

    # Print nodes along the shortest path and lengths of the edges
    xs = []
    for i in range(min(k, len(k_shortest_path))):
        total_length = 0
        x = np.zeros(len(graph.edges))  # Initiate route vector (x or y)
        for j in range(len(k_shortest_path[i]) - 1):
            edge_index = edge_indices[(k_shortest_path[i][j], k_shortest_path[i][j + 1])]
            x[edge_index] = 1  # If the edge is in the route, then xi will be 1
            edge_length = c0[edge_index]
            total_length += edge_length

        xs.append((x, np.dot(c0, x)))

    return xs


def solve_optimization_for_diff_gamma(num_responders, xs, ys, ws, k=3, max_gamma=120):
    three_optimal_gamma = np.zeros(max_gamma + 1)
    two_optimal_gamma = np.zeros(max_gamma + 1)
    one_optimal_gamma = np.zeros(max_gamma + 1)
    x_gamma = np.zeros(max_gamma + 1)
    y_gamma = np.zeros(max_gamma + 1)
    w_gamma = np.zeros(max_gamma + 1)
    optimal_index = (0, 0)
    optimal_origin = "x"

    # fix delta = 1.0 * c0
    delta = 1.0
    for gamma in range(0, max_gamma + 1):
        # three responders:
        if num_responders >= 3:
            for i in range(k):
                for j in range(k):
                    for h in range(k):
                        x = xs[i][0]
                        y = ys[j][0]
                        w = ws[h][0]

                        # Define variables
                        c = cp.Variable(len(c0))
                        z = cp.Variable()

                        # Define constraints
                        constraints = [
                            z <= x.T @ c,
                            z <= y.T @ c,
                            z <= w.T @ c,
                            c >= c0,
                            c <= (1.0 + delta) * c0,
                            cp.sum((c - c0) / (c0 * delta)) <= gamma
                        ]

                        # Define the optimization problem
                        problem = cp.Problem(cp.Maximize(z), constraints)

                        # Solve the optimization problem
                        try:
                            start = time.time()
                            problem.solve(max_iters=10000)
                            end = time.time()
                            print("Three responders")
                            print("time used", end - start)

                            # Output result
                            print("Gamma: ", gamma)
                            print("(i, j, h): ", i, j, h)
                            print("Optimal value：", problem.value)
                            print()

                            if three_optimal_gamma[gamma] == 0 or problem.value < three_optimal_gamma[gamma]:
                                three_optimal_gamma[gamma] = problem.value
                                optimal_index = (i, j, h)
                                if (np.dot(c.value, x) < np.dot(c.value, y)) and (np.dot(c.value, x) < np.dot(c.value, w)):
                                    optimal_origin = "x"
                                elif (np.dot(c.value, y) < np.dot(c.value, x)) and (np.dot(c.value, y) < np.dot(c.value, w)):
                                    optimal_origin = "y"
                                else:
                                    optimal_origin = "w"
                            if x_gamma[gamma] == 0 or np.dot(c.value, x) < x_gamma[gamma]:
                                x_gamma[gamma] = np.dot(c.value, x)
                            if y_gamma[gamma] == 0 or np.dot(c.value, y) < y_gamma[gamma]:
                                y_gamma[gamma] = np.dot(c.value, y)
                            if w_gamma[gamma] == 0 or np.dot(c.value, y) < w_gamma[gamma]:
                                w_gamma[gamma] = np.dot(c.value, w)

                            if gamma == 20:
                                for idx in range(len(constraints)):
                                    print(f"Dual value {idx}: ", constraints[idx].dual_value)
                        except:
                            print(gamma, " failure")
            print("optimal_index: ", optimal_index, "; optimal origin: ", optimal_origin)
            print()

        # Two responders
        if num_responders >= 2:
            for i in range(k):
                for j in range(k):
                    x = xs[i][0]
                    y = ys[j][0]

                    # Define variables
                    c = cp.Variable(len(c0))
                    z = cp.Variable()

                    # Define constraints
                    constraints = [
                        z <= x.T @ c,
                        z <= y.T @ c,
                        c >= c0,
                        c <= (1.0 + delta) * c0,
                        cp.sum((c - c0) / (c0 * delta)) <= gamma
                    ]

                    # Define the optimization problem
                    problem = cp.Problem(cp.Maximize(z), constraints)

                    # Solve the optimization problem
                    try:
                        start = time.time()
                        problem.solve(max_iters=10000)
                        end = time.time()
                        print("Two responders")
                        print("time used", end - start)

                        # Output result
                        print("Gamma: ", gamma)
                        print("(i, j): ", i, j)
                        print()

                        if two_optimal_gamma[gamma] == 0 or problem.value < two_optimal_gamma[gamma]:
                            two_optimal_gamma[gamma] = problem.value
                            optimal_index = (i, j)
                            if (np.dot(c.value, x) < np.dot(c.value, y)):
                                optimal_origin = "x"
                            else:
                                optimal_origin = "y"
                        if x_gamma[gamma] == 0 or np.dot(c.value, x) < x_gamma[gamma]:
                            x_gamma[gamma] = np.dot(c.value, x)
                        if y_gamma[gamma] == 0 or np.dot(c.value, y) < y_gamma[gamma]:
                            y_gamma[gamma] = np.dot(c.value, y)

                        if gamma == 20:
                            for idx in range(len(constraints)):
                                print(f"Dual value {idx}: ", constraints[idx].dual_value)
                    except:
                        print(gamma, " failure")
            print("optimal_index: ", optimal_index, "; optimal origin: ", optimal_origin)
            print()

        # One responder
        if num_responders >= 1:
            for i in range(k):
                x = xs[i][0]

                # Define variables
                c = cp.Variable(len(c0))
                z = cp.Variable()

                # Define constraints
                constraints = [
                    z <= x.T @ c,
                    c >= c0,
                    c <= (1.0 + delta) * c0,
                    cp.sum((c - c0) / (c0 * delta)) <= gamma
                ]

                # Define the optimization problem
                problem = cp.Problem(cp.Maximize(z), constraints)

                # Solve the optimization problem
                try:
                    start = time.time()
                    problem.solve(max_iters=10000)
                    end = time.time()
                    print("One responder")
                    print("time used", end - start)

                    # Output result
                    print("Gamma: ", gamma)
                    print()

                    if one_optimal_gamma[gamma] == 0 or problem.value < one_optimal_gamma[gamma]:
                        one_optimal_gamma[gamma] = problem.value

                    if x_gamma[gamma] == 0 or np.dot(c.value, x) < x_gamma[gamma]:
                        x_gamma[gamma] = np.dot(c.value, x)

                    if gamma == 20:
                        for idx in range(len(constraints)):
                            print(f"Dual value {idx}: ", constraints[idx].dual_value)
                except:
                    print(gamma, " failure")
            print("optimal_index: ", optimal_index, "; optimal origin: ", optimal_origin)
            print()

    print(one_optimal_gamma)
    print(two_optimal_gamma)
    print(three_optimal_gamma)

    indices = np.arange(0, max_gamma + 1)

    # Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(indices, one_optimal_gamma, color='blue', label='One Responder')
    plt.plot(indices, two_optimal_gamma, color='red', label='Two Responders')
    plt.plot(indices, three_optimal_gamma, color='purple', label='Three Responders')
    plt.xlabel('Gamma')
    plt.ylabel('Value')
    plt.title('Optimal value as a function of Gamma')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    prefix = "kowloon"
    graph, c0, edge_indices, _ = load_graph(prefix)

    k = 3
    start_lat1, start_lon1, start_lat2, start_lon2, start_lat3, start_lon3, end_lat, end_lon = 0, 0, 0, 0, 0, 0, 0, 0

    if prefix == "kowloon":
        start_lat1, start_lon1 = 22.30827,114.18340
        start_lat2, start_lon2 = 22.31652,114.18069
        start_lat3, start_lon3 = 22.33606,114.18582
        end_lat, end_lon = 22.30932,114.23042

    num_responders = 0
    # sort to make sure xs, ys, ws are in order.
    # add if statements because maybe we only have 2 responders.
    if start_lat1 > 0 and end_lat > 0:
        xs = get_shortest_paths(graph, start_lat1, start_lon1, end_lat, end_lon, c0, edge_indices, prefix=prefix, suffix="x", k=k)
        num_responders += 1
    if start_lat2 > 0 and end_lat > 0:
        ys = get_shortest_paths(graph, start_lat2, start_lon2, end_lat, end_lon, c0, edge_indices, prefix=prefix, suffix="y", k=k)
        num_responders += 1
        lists = [(xs[0][1], 'xs', xs), (ys[0][1], 'ys', ys)]
        lists.sort(key=lambda x: x[0])
        xs, ys = lists[0][2], lists[1][2]
    if start_lat3 > 0 and end_lat > 0:
        ws = get_shortest_paths(graph, start_lat3, start_lon3, end_lat, end_lon, c0, edge_indices, prefix=prefix, suffix="w", k=k)
        num_responders += 1
        lists = [(xs[0][1], 'xs', xs), (ys[0][1], 'ys', ys), (ws[0][1], 'ws', ws)]
        lists.sort(key=lambda x: x[0])
        xs, ys, ws = lists[0][2], lists[1][2], lists[2][2]

    solve_optimization_for_diff_gamma(num_responders, xs, ys, ws, k=k, max_gamma=120)

