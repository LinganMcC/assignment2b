#     return goal, nodes_created, path
# EDIT: FOR OBJECTORIENTED IMPIMENTATION MAKE SURE IT COMPLIES WITH BASE.PY FILE AND GRAPH.PY FILE

from algorithms.base import SearchAlgorithm
 
 
class CUS2(SearchAlgorithm):
    def search(self):

        origin       = self.graph.origin
        destinations = set(self.graph.destinations)
 
        if not destinations:
            return None, 0, []
 
        # if origin is already a goal
        if origin in destinations:
            return origin, 1, [origin]
 
        self.nodes_created = 1
        bound = self.graph.heuristic(origin)
        path = [origin]
 
        while True:
            # search with current f-cost bound
            result, goal, final_path = self._search_recursive(path, 0, bound, destinations)
 
            if goal is not None:
                return goal, self.nodes_created, final_path
 
            if result == float('inf'):
                return None, self.nodes_created, []
 
            bound = result
 
    def _search_recursive(self, path, g_cost, bound, destinations):
        """
        Recursive helper for IDA*.
        
        Performs depth-first search down from current node,
        but prunes branches where f = g + h exceeds the bound.

        Returns:
            (min_f_exceeding_bound, goal_node_or_none, final_path_or_none)
        """
        current = path[-1]
        f_cost = g_cost + self.graph.heuristic(current)
 
        # if f exceeds bound, return this f value so we know new bound for next iteration
        if f_cost > bound:
            return f_cost, None, None
 
        # goal check
        if current in destinations:
            return f_cost, current, path
 
        # track the minimum f value that exceeded bound
        # (this becomes the new bound for next iteration)
        minimum = float('inf')
 
        # expand neighbours
        for neighbour, cost in self.graph.get_neighbours(current):
            if neighbour not in path:  # avoid cycles in tree search
                self.nodes_created += 1
                new_path = path + [neighbour]
                result, goal, final_path = self._search_recursive(
                    new_path,
                    g_cost + cost,
                    bound,
                    destinations
                )
 
                # if we found the goal, return immediately
                if goal is not None:
                    return result, goal, final_path
 
                # track the minimum f value that exceeded the bound
                if result < minimum:
                    minimum = result
 
        # no solution found in this branch, return min f that exceeded bound
        return minimum, None, None