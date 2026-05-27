import math


class Graph:
    """
    Stores the entire route finding problem — nodes, edges, origin, destinations.
    Also provides helper methods that the search algorithms need.
    Replaces parse.py and utils.py by bundling data and functions together.
    """

    def __init__(self):
        self.nodes        = {}   # { node_id -> (x, y) }
        self.edges        = {}   # { from_node -> [(to_node, cost), ...] }
        self.origin       = None
        self.destinations = []

    def load_from_file(self, filename):
        """Reads a problem file and populates this graph's attributes."""
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]

        current_section = None

        for line in lines:
            if line == 'Nodes:':
                current_section = 'nodes'
                continue
            elif line == 'Edges:':
                current_section = 'edges'
                continue
            elif line == 'Origin:':
                current_section = 'origin'
                continue
            elif line == 'Destinations:':
                current_section = 'destinations'
                continue

            if current_section == 'nodes':
                # Example line: "1: (4,1)"
                parts   = line.split(': ', 1)
                node_id = int(parts[0].strip())
                coords  = parts[1].strip().strip('()')
                x, y    = coords.split(',')
                self.nodes[node_id] = (int(x), int(y))

            elif current_section == 'edges':
                # Example line: "(2,1): 4"
                parts     = line.split(': ', 1)
                ends      = parts[0].strip().strip('()').split(',')
                from_node = int(ends[0])
                to_node   = int(ends[1])
                cost      = int(parts[1].strip())

                if from_node not in self.edges:
                    self.edges[from_node] = []
                self.edges[from_node].append((to_node, cost))

            elif current_section == 'origin':
                # Example line: "2"
                self.origin = int(line)

            elif current_section == 'destinations':
                # Example line: "5; 4"
                self.destinations = [int(d.strip()) for d in line.split(';')]

    def get_neighbours(self, node):
        """Returns neighbours sorted ascending by node ID for tie-breaking."""
        neighbours = self.edges.get(node, [])
        return sorted(neighbours, key=lambda x: x[0])

    def heuristic(self, node_id):
        """Minimum Euclidean distance from node to any destination."""
        x1, y1 = self.nodes[node_id]
        min_dist = float('inf')
        for dest in self.destinations:
            x2, y2 = self.nodes[dest]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            min_dist = min(min_dist, dist)
        return min_dist