from collections import deque
from algorithms.base import SearchAlgorithm


class BFS(SearchAlgorithm):
    """
    Breadth-First Search — explores level by level using a QUEUE (FIFO).
    Guaranteed to find the path with fewest steps but not lowest cost.
    """

    def search(self):
        origin       = self.graph.origin
        destinations = set(self.graph.destinations)

        # queue stores (current_node, path_so_far)
        queue = deque()
        queue.append((origin, [origin]))

        # track visited nodes so we don't go in circles
        visited = set([origin])

        # count nodes as they are CREATED
        nodes_create = 1

        while queue:
            current_node, path = queue.popleft()   # take from the FRONT (FIFO)

            # goal check
            if current_node in destinations:
                return current_node, nodes_create, path

            # expand neighbours sorted ascending for tie-breaking
            for neighbour, cost in self.graph.get_neighbours(current_node):
                if neighbour not in visited:
                    visited.add(neighbour)
                    nodes_create += 1
                    queue.append((neighbour, path + [neighbour]))

        # no solution found
        return None, nodes_create, []