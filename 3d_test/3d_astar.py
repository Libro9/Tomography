# A* algorithm applied to a set of points
# Used to find the shortest path between two points in a 3D steel along grain boundaries
# Uses a heuristic given by the straight line distance between current and end node
# Automatically selects start and end points closest to the coordinates given

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from sklearn.neighbors import NearestNeighbors

# Define a Graph class
class Graph:
    def __init__(self, points, max_distance=0.52):
        self.graph = {}
        self.points = points
        self.max_distance = max_distance
        self._build_graph()
        self.neighbors = NearestNeighbors(n_neighbors=1)  # Nearest neighbor model for finding closest points
        self.neighbors.fit(self.points[:, 0:3])  # Fit all three dimensions

    def _build_graph(self):
        """
        Efficiently build the graph by using k-d trees for finding neighbors.
        This method builds edges between points within a given maximum distance.
        """
        # Use k-d tree to efficiently find neighbors within max_distance
        nbrs = NearestNeighbors(radius=self.max_distance).fit(self.points)
        distances, indices = nbrs.radius_neighbors(self.points)
        
        # Build the graph
        for i, neighbors in enumerate(indices):
            for j, neighbor_index in enumerate(neighbors):
                if i != neighbor_index:  # Avoid self-loops
                    distance = distances[i][j]  # Use precomputed distance from 'distances'
                    self.add_edge(i, neighbor_index, distance)
    
    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph:
            self.graph[node1] = {}
        self.graph[node1][node2] = weight

    def heuristic(self, node, target):
        """
        Calculate the heuristic (Euclidean distance) from the node to the target.
        """
        return np.linalg.norm(self.points[node, :] - self.points[target, :])
    
    def a_star(self, source, target):
        """
        Compute the shortest path from source to target using A* algorithm.
        Returns the path and distances.
        """
        # Priority queue to store nodes with their f(n) values
        open_set = [(0, source)]  # (f(n), node)
        g_values = {node: float("inf") for node in self.graph}  # Cost from start node to current node
        g_values[source] = 0
        f_values = {node: float("inf") for node in self.graph}  # Estimated cost to reach the target
        f_values[source] = self.heuristic(source, target)
        predecessors = {node: None for node in self.graph}  # For path reconstruction

        visited = set()

        while open_set:
            current_f, current_node = heappop(open_set)

            if current_node in visited:
                continue
            visited.add(current_node)

            # If we reached the target, reconstruct the path
            if current_node == target:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = predecessors[current_node]
                path.reverse()
                return path, g_values

            # Explore neighbors
            for neighbor, weight in self.graph[current_node].items():
                if neighbor in visited:
                    continue
                
                tentative_g = g_values[current_node] + weight
                if tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_values[neighbor] = tentative_g + self.heuristic(neighbor, target)
                    predecessors[neighbor] = current_node
                    heappush(open_set, (f_values[neighbor], neighbor))

        return [], g_values  # Return empty path if target is not reachable

    def shortest_distances(self, source):
        """
        Compute shortest distances from the source node using Dijkstra's algorithm.
        Returns the distances to all nodes and a dictionary of predecessors for path reconstruction.
        """
        distances = {node: float("inf") for node in self.graph}
        distances[source] = 0
        pq = [(0, source)]
        visited = set()

        while pq:
            current_distance, current_node = heappop(pq)
            if current_node in visited:
                continue
            visited.add(current_node)
            
            for neighbor, weight in self.graph[current_node].items():
                tentative_distance = current_distance + weight
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    heappush(pq, (tentative_distance, neighbor))

        predecessors = {node: None for node in self.graph}
        for node, distance in distances.items():
            for neighbor, weight in self.graph[node].items():
                if distances[neighbor] == distance + weight:
                    predecessors[neighbor] = node

        return distances, predecessors
    
    def shortest_path(self, source, target):
        """
        Returns the shortest path from source to target using the predecessors dictionary.
        """
        distances, predecessors = self.shortest_distances(source)
        path = []
        current_node = target
        while current_node:
            path.append(current_node)
            current_node = predecessors[current_node]
        path.reverse()
        return path, distances
    
    def plot(self, path, start, end, show_edges=False):
        """
        Plots the points and the shortest path. If show_edges is True all connections between nodes will also be plotted.
        """
        # Extract the coordinates of the path for plotting
        path_points = self.points[[int(node) for node in path]]

        # Plot the points
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.points[0:6070, 0], self.points[:6070, 1], self.points[:6070, 2], s=1, c='k', label='Grain Boundaries')

        # Plot the path
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], c='r', label='Path')

        # Highlight the start and end points
        ax.scatter(self.points[start, 0], self.points[start, 1], self.points[start, 2], c='blue', label='Start')
        ax.scatter(self.points[end, 0], self.points[end, 1], self.points[end, 2], c='green', label='End')

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        ax.set_title('Shortest Path along the Grain Boundaries in 3D')

        # Save and show the plot
        fig.savefig('3d/3D_Shortest_Path.png')
        return fig
    
    def find_closest_point(self, coords):
        """
        Finds the index of the point closest to the given coordinates using Nearest Neighbors.
        """
        distances, indices = self.neighbors.kneighbors([coords])
        return indices[0][0]  # Return the closest point index

def main(start_coords, end_coords, astar=True):
    
    start_time = timer()

    # Load the coordinates from the CSV file
    points = np.genfromtxt('3d/3d_points.csv', delimiter=',', skip_header=1)

    print(f'{(timer() - start_time):2f}s points loaded')

    # Initialise the graph
    graph = Graph(points)
    start = graph.find_closest_point(start_coords)
    end = graph.find_closest_point(end_coords)
    print(f"Start point: {points[start]}")
    print(f"End point: {points[end]}")

    print(f'{(timer() - start_time):2f}s graph initialised')

    # Find the shortest path
    if astar:
        path, distances = graph.a_star(start, end)
    else:
        path, distances = graph.shortest_path(start, end)

    DISTANCE = distances[end]
    FIGURE = graph.plot(path, start, end)

    if path != []:
        print(f"The shortest path from {points[start]} to {points[end]} has a length of {DISTANCE:.1f} and is shown in the figure.")
    else:
        print(f'No path found between {points[start]} and {points[end]}')

    # Timing
    end_time = timer()
    print('Time elapsed in seconds:', end_time - start_time)
    plt.show()

# Set start and end points
START_COORDS = [40, 40, 0.3]
END_COORDS = [16, 57, 20]

main(START_COORDS, END_COORDS)