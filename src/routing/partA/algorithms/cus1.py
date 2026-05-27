from collections import deque
from algorithms.base import SearchAlgorithm


class CUS1(SearchAlgorithm):
    """
    Custom Uninformed Search 1 — Bidirectional BFS (Directed-safe).
    Uses forward edges from origin and reverse edges from goal.
    """

    def build_reverse_graph(self):
        """
        Builds a mapping: node -> list of predecessors
        so we can traverse incoming edges efficiently.
        """
        reverse = {}

        for node in self.graph.nodes:
            reverse[node] = []

        for node in self.graph.nodes:
            for neighbour, cost in self.graph.get_neighbours(node):
                reverse.setdefault(neighbour, []).append((node, cost))

        return reverse

    def search(self):
        origin       = self.graph.origin
        destinations = set(self.graph.destinations)

        if not destinations:
            return None, 0, []

        if origin in destinations:
            return origin, 1, [origin]

        # Build reverse graph ONCE
        reverse_graph = self.build_reverse_graph()

        nodes_created = 1

        for goal in destinations:

            forward_queue   = deque([(origin, [origin])])
            forward_visited = {origin: [origin]}

            backward_queue   = deque([(goal, [goal])])
            backward_visited = {goal: [goal]}

            nodes_created = 2

            while forward_queue and backward_queue:

                # ---- Forward expansion ----
                current_f, path_f = forward_queue.popleft()

                for neighbour, cost in self.graph.get_neighbours(current_f):
                    if neighbour not in forward_visited:
                        new_path_f = path_f + [neighbour]
                        forward_visited[neighbour] = new_path_f
                        forward_queue.append((neighbour, new_path_f))
                        nodes_created += 1

                        if neighbour in backward_visited:
                            backward_path = backward_visited[neighbour]
                            return goal, nodes_created, new_path_f + backward_path[::-1][1:]

                # ---- Backward expansion (NOW CORRECT) ----
                current_b, path_b = backward_queue.popleft()

                for neighbour, cost in reverse_graph.get(current_b, []):
                    if neighbour not in backward_visited:
                        new_path_b = path_b + [neighbour]
                        backward_visited[neighbour] = new_path_b
                        backward_queue.append((neighbour, new_path_b))
                        nodes_created += 1

                        if neighbour in forward_visited:
                            forward_path = forward_visited[neighbour]
                            return goal, nodes_created, forward_path + new_path_b[::-1][1:]

        return None, nodes_created, []
