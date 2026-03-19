# dag.py
# -----------------------------------------------------------------------------
# Project: Clinical Bayesian Network for Heart Disease Prediction
# Role: Member 1 - System Architect
# Implements: Bayesian Network Structure (DAG), Cycle Detection, and Topological Sorting
# -----------------------------------------------------------------------------

from collections import deque


class BayesNetStructure:
    """
    Defines the structural properties of the Bayesian Network.

    This class manages the Directed Acyclic Graph (DAG) representing causal
    relationships between clinical features and heart disease. It provides
    utilities for graph validation and sequence ordering for inference.
    """

    def __init__(self):
        """
        Initializes the network with standardized domains and causal structures.
        """
        # 1. Variable Domains: Defined states for each discrete node.
        # These keys serve as the global 'Source of Truth' for the project.
        self.domains = {
            "Age":            ["young", "middle", "senior"],
            "Sex":            [0, 1],
            "Slope":          [1, 2, 3],
            "Thal":           [3, 6, 7],
            "FastingBS":      [0, 1],
            "BloodPressure":  ["low", "medium", "high"],
            "Cholesterol":    ["low", "medium", "high"],
            "HeartDisease":   [0, 1],
            "Cp":             [1, 2, 3, 4],
            "RestingECG":     [0, 1, 2],
            "Thalach":        ["low", "medium", "high"],
            "ExerciseAngina": [0, 1],
            "Oldpeak":        ["low", "high"],
            "Ca":             [0, 1, 2, 3]
        }
        self.nodes = list(self.domains.keys())

        # 2. Causal Adjacency List: { Child: [Parents] }
        # The structure follows a 'Waterfall' clinical model:
        # Demographics -> Bio-markers -> Disease -> Symptoms/Tests
        self.structure = {
            "Age":            [],
            "Sex":            [],
            "Slope":          [],
            "Thal":           [],
            "FastingBS":      ["Age"],
            "BloodPressure":  ["Age", "Sex"],
            "Cholesterol":    ["Age", "Sex"],
            "HeartDisease":   ["Age", "Sex", "BloodPressure", "Cholesterol", "FastingBS"],
            "Cp":             ["HeartDisease"],
            "RestingECG":     ["HeartDisease"],
            "Thalach":        ["HeartDisease"],
            "ExerciseAngina": ["HeartDisease"],
            "Oldpeak":        ["HeartDisease", "Slope"],
            "Ca":             ["HeartDisease"]
        }

    def get_parents(self, node: str) -> list:
        """
        Retrieves the parent nodes for a given child node.

        Args:
            node (str): The name of the variable.
            Returns:
                list: A list of immediate causal parent names.
                """
        return self.structure.get(node, [])

    def topological_sort(self) -> list:
        """
        Computes a linear ordering of nodes using Kahn's Algorithm.

        This ordering ensures that parents are processed before their children,
        which is a requirement for the Variable Elimination algorithm.

        Returns:
            list: A list of node names in topological order.
            """
        # Compute in-degrees (number of parents) for each node
        in_degree = {node: len(self.get_parents(node)) for node in self.nodes}

        # Initialize queue with root nodes (those with no parents)
        queue = deque(
            [node for node in self.nodes if in_degree[node] == 0])
        sorted_list = []

        while queue:
            current = queue.popleft()
            sorted_list.append(current)

            # Reduce in-degree for children and add to queue if they become roots
            for child in self.nodes:
                if current in self.get_parents(child):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        return sorted_list

    def check_for_cycles(self) -> bool:
        """
        Validates the graph's acyclic property using Depth First Search (DFS).

        Returns:
            bool: True if the graph is a valid DAG (no cycles), False otherwise.
            """
        visited = set()
        stack = set()

        def visit(node):
            if node in stack:
                return True  # Cycle detected via back-edge
            if node in visited:
                return False

            visited.add(node)
            stack.add(node)

            # Recursive check through all parental lineages
            for parent in self.get_parents(node):
                if visit(parent):
                    return True

            stack.remove(node)  # Backtrack: remove from current path
            return False

        # Iterate through all nodes to ensure disconnected components are checked
        for n in self.nodes:
            if n not in visited:
                if visit(n):
                    return False  # Cycle found in the graph
        return True
# -----------------------------------------------------------------------------
# Unit Testing / Validation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    dag = BayesNetStructure()
    print("=== Architect's DAG Validation ===")

    if dag.check_for_cycles():
        print(" [STATUS] Graph Check: Valid DAG (No Cycles detected).")
        order = dag.topological_sort()
        print(f" [STATUS] Sorting Check: {order}")
    else:
        print(" [ERROR] Graph Check: Cycle Detected! Structure is invalid.")
