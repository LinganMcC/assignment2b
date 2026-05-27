#     return goal, nodes_created, path
# EDIT: FOR OBJECTORIENTED IMPIMENTATION MAKE SURE IT COMPLIES WITH BASE.PY FILE AND GRAPH.PY FILE

import heapq
from collections import defaultdict
from algorithms.base import SearchAlgorithm


class AStar(SearchAlgorithm):
    """
    A* Search — uses both cost so far (g) and estimated cost to goal (h).
    f = g + h. Guaranteed to find the optimal (lowest cost) path.
    """

    def search(self):
        origin       = self.graph.origin
        destinations = set(self.graph.destinations)

        if not destinations:
            return None, 0, []

        # priority queue stores (f_score, g_score, node_id)
        open_set = []
        heapq.heappush(open_set, (self.graph.heuristic(origin), 0, origin))

        came_from = {}
        g_score   = defaultdict(lambda: float('inf'))
        g_score[origin] = 0

        visited       = set()
        nodes_created = 1

        while open_set:
            f, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            # goal check
            if current in destinations:
                path         = []
                goal_reached = current
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(origin)
                path.reverse()
                return goal_reached, nodes_created, path

            for neighbour, cost in self.graph.edges.get(current, []):
                tentative_g = g_score[current] + cost

                if tentative_g < g_score[neighbour]:
                    came_from[neighbour]  = current
                    g_score[neighbour]    = tentative_g
                    f_score               = tentative_g + self.graph.heuristic(neighbour)

                    if neighbour not in visited:
                        heapq.heappush(open_set, (f_score, tentative_g, neighbour))
                        nodes_created += 1

        return None, nodes_created, []