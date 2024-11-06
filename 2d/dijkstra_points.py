#Dijkstra's algorithm applied to a set of points
#Used to find the shortest path between two points on a 2D slice along grain boundaries

from timeit import default_timer as timer
from heapq import heapify, heappop, heappush
import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt('2d/points_1slice.csv', delimiter=',', skip_header=1) #import the coordinates of the points from a csv file into a numpy array

COORDS = True #Allows you to enter the coordinates of start and end points, if false the argument of the coordinates must be given
START_COORDS = [6.6, 46.5]
END_COORDS = [57.3, 55.5]

if COORDS:
    START = str(np.where((points[:, 0:2] == START_COORDS).all(axis=1))[0][0])
    END = str(np.where((points[:, 0:2] == END_COORDS).all(axis=1))[0][0])
else:
    START = 3
    END = 5000

start = timer()
graph = {}

class Graph:
    def __init__(self, graph):

        self.graph = graph

    def add_edge(self, node1, node2, weight):

        if node1 not in self.graph:  # Check if the node is already added
            self.graph[node1] = {}  # If not, create the node
        self.graph[node1][node2] = weight  # Else, add a connection to its neighbour

    def shortest_distances(self, source: str):

        # Initialize the values of all nodes with infinity
        distances = {node: float("inf") for node in self.graph}
        distances[source] = 0  # Set the source value to 0

        # Initialize a priority queue
        pq = [(0, source)]
        heapify(pq)

        # Create a set to hold visited nodes
        visited = set()

        while pq:  # While the priority queue isn't empty

            current_distance, current_node = heappop(pq)  # Get the node with the min distance
            if current_node in visited:
                continue  # Skip already visited nodes
            visited.add(current_node)  # Else, add the node to visited set
            
            for neighbor, weight in self.graph[current_node].items():
                # Calculate the distance from current_node to the neighbour
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
    
    def shortest_path(self, source: str, target: str):
        # Generate the predecessors dict
        distances, predecessors = self.shortest_distances(source)

        path = []
        current_node = target

        # Backtrack from the target node using predecessors
        while current_node:
            path.append(current_node)
            current_node = predecessors[current_node]

        # Reverse the path and return it
        path.reverse()

        return path, distances, predecessors

G = Graph(graph)

#Here I add all the points and paths between points to the graph
for index in range(len(points)):
    x = points[index, 0]
    y = points[index, 1]

    #I define the paths by joining all points within a certain proximity by a straight line
    neighbours = np.argwhere(((x - 0.4) < points[:, 0]) & (points[:, 0] < (x + 0.4)) & ((y - 0.4) < points[:, 1]) & (points[:, 1] < (y + 0.4)))
    for node in neighbours:
        if node == index:
            continue
        distance = np.sqrt((points[index, 0] - points[node[0], 0]) ** 2 + (points[index, 1] - points[node[0], 1]) ** 2)
        G.add_edge(str(index), str(node[0]), distance)


PATH, distances, predecessors = G.shortest_path(START, END)
DISTANCE = distances[END]
print('The shortest distance from {0} to {1} along the grain boundaries is {2:.1f} and the path taken is shown in the figure.'
      .format(points[int(START), 0:2], points[int(END), 0:2], DISTANCE), '\n')

path_points = np.empty((0, 2)) #Define an array where all the points on the path will be stored
for step in np.array(PATH, dtype=int):
    path_points = np.vstack((path_points, points[step, 0:2]))

#plot the shortest path
FIG = plt.figure(figsize=(7, 7))
AX = FIG.add_subplot(111)
AX.scatter(points[:, 0], points[:, 1], s=1, c='k', label='Grain Boundaries')
AX.plot(path_points[:, 0], path_points[:, 1], c='r', label='Path')
AX.scatter(path_points[0, 0], path_points[0, 1], c='blue', label='Start')
AX.scatter(path_points[-1, 0], path_points[-1, 1], c='green', label='End')
AX.set_xlabel('x')
AX.set_ylabel('y')
AX.legend()
AX.set_title('Shortest path between two points along the grain boundaries')
FIG.savefig('2d/2D Shortest Path.png')

end = timer()
print('Time elapsed in seconds: ', end - start)
plt.show()