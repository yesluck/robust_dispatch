import osmnx as ox
import networkx as nx
import pickle

# Run only on the first time
# Get the road network of Kowloon
place_name = "Kowloon, Hong Kong"
graph = ox.graph_from_place(place_name, network_type='drive')
print(len(graph.nodes))
print(len(graph.edges))

# # Save the graph to a file
with open("kowloon_drive_graph.pkl", "wb") as f:
    pickle.dump(graph, f)

# Read the graph from the file
# with open("kowloon_graph.pkl", "rb") as f:
#     graph = pickle.load(f)

def get_nearest_node(graph, lat, lon):
    return ox.distance.nearest_nodes(graph, X=lon, Y=lat)

def get_shortest_path(start_lat, start_lon, end_lat, end_lon):
    start_node = get_nearest_node(graph, start_lat, start_lon)
    end_node = get_nearest_node(graph, end_lat, end_lon)

    # Visualize using networkx
    fig, ax = ox.plot_graph(graph, figsize=(50, 50), show=False, close=False,
                            node_color='blue', node_size=15, edge_color='black', bgcolor='white', dpi=1000)

    # Save the figure to a file
    fig.savefig("kowloon_network.png")

    # Compute the shortest path using networkx's shortest_path function
    shortest_path = nx.shortest_path(graph, source=start_node, target=end_node, weight='length')

    # Plot the shortest path
    fig, ax = ox.plot_graph_route(graph, shortest_path, figsize=(50, 50), route_linewidth=3, route_color='red',
                                  orig_dest_size=50, node_color='blue', node_size=15, edge_color='black', bgcolor='white', dpi=1000)

    # Save the figure with the shortest path to a file
    fig.savefig("kowloon_shortest_path.png")

    # Print nodes along the shortest path and lengths of the edges
    print("Shortest path by nodes:", shortest_path)
    print("\nEdge lengths along the path:")

    total_length = 0
    for i in range(len(shortest_path) - 1):
        edge_data = graph.get_edge_data(shortest_path[i], shortest_path[i + 1])[0]
        edge_length = edge_data['length']
        total_length += edge_length
        print(f"Edge from {shortest_path[i]} to {shortest_path[i + 1]}: {edge_length} meters")

    print("\nTotal path length:", total_length, "meters")


# Example: Two latitude and longitude coordinates in Kowloon, Hong Kong
start_lat, start_lon = 22.3167, 114.1839  # Rough center point of Kowloon City
end_lat, end_lon = 22.3282, 114.1881  # Rough center point of Kowloon Tong
get_shortest_path(start_lat, start_lon, end_lat, end_lon)
