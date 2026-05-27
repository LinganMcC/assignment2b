#     return goal, nodes_created, path
# EDIT: FOR OBJECTORIENTED IMPIMENTATION MAKE SURE IT COMPLIES WITH BASE.PY FILE AND GRAPH.PY FILE

from algorithms.base import SearchAlgorithm


class DFS(SearchAlgorithm):
    """
    Depth-First Search — dives deep using a STACK (LIFO).
    Not guaranteed to find shortest or cheapest path.
    """

    def search(self):
        origin       = self.graph.origin
        destinations = set(self.graph.destinations)

        # stack stores (current_node, path_so_far)
        stack = [(origin, [origin])]

        # track visited nodes to avoid cycles
        visited = set([origin])

        # count nodes as they are CREATED
        nodes_create = 1

        while stack:
            current_node, path = stack.pop()   # take from the TOP (LIFO)

            # goal check
            if current_node in destinations:
                return current_node, nodes_create, path

            # reverse sorted so smallest node ends up on top of stack and explored first
            for neighbour, cost in reversed(self.graph.get_neighbours(current_node)):
                if neighbour not in visited:
                    visited.add(neighbour)
                    nodes_create += 1
                    stack.append((neighbour, path + [neighbour]))

        # no solution found
        return None, nodes_create, []