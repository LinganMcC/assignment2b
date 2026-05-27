from graph import Graph


class SearchAlgorithm:
    """
    Parent class all search algorithms inherit from.
    Provides shared setup so each algorithm only needs to implement search().
    """

    def __init__(self, graph: Graph):
        self.graph         = graph
        self.nodes_created = 0
        self.visited       = set()

    def search(self):
        """
        Run the search. Every child class must implement this.
        Returns: (goal, nodes_created, path)
        """
        raise NotImplementedError("Each algorithm must implement search()")