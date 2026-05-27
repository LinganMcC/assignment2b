#     return goal, nodes_created, path
# EDIT: FOR OBJECTORIENTED IMPIMENTATION MAKE SURE IT COMPLIES WITH BASE.PY FILE AND GRAPH.PY FILE
import heapq
from algorithms.base import SearchAlgorithm


class GBFS(SearchAlgorithm):
    """
    Greedy Best-First Search — always expands the node closest to the goal.
    Uses only the heuristic (estimated cost to goal), ignores path cost so far.
    Not guaranteed to find the optimal path.
    """

    def search(self):
        origin       = self.graph.origin
        destinations = set(self.graph.destinations)

        if not destinations:
            return None, 0, []

        # priority queue stores (heuristic, node_id)
        frontier = []
        heapq.heappush(frontier, (self.graph.heuristic(origin), origin))

        came_from = {}
        visited   = set()
        nodes_created = 1   # origin counts

        while frontier:
            _, current = heapq.heappop(frontier)

            if current in visited:
                continue
            visited.add(current)

            # goal check
            if current in destinations:
                # reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(origin)
                path.reverse()
                return current, nodes_created, path

            for neighbour, _ in self.graph.edges.get(current, []):
                if neighbour not in visited:
                    heapq.heappush(frontier, (self.graph.heuristic(neighbour), neighbour))
                    if neighbour not in came_from:
                        came_from[neighbour] = current
                        nodes_created += 1

        return None, nodes_created, []