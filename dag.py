# dag.py
# Implements: BayesNetStructure, get_parents, topological_sort, check_for_cycles

from collections import deque
from inspect import stack


class BayesNetStructure:
    def __init__(self):
        # 1. Domains (Shared Contract with Team)
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

        # 2. Causality Structure (The "Waterfall" Model)
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

    def get_parents(self, node):
        return self.structure.get(node, [])

    def topological_sort(self):
        """Returns a valid ordering of nodes for Member 2's engine."""
        in_degree = {node: len(self.get_parents(node)) for node in self.nodes}
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        sorted_list = []

        while queue:
            current = queue.popleft()
            sorted_list.append(current)
            for child in self.nodes:
                if current in self.get_parents(child):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        return sorted_list

    def check_for_cycles(self) -> bool:
        """Uses DFS to ensure the graph is a Directed Acyclic Graph (DAG)."""
        visited = set()
        stack = set()


        def visit(node):
            if node in stack:
                return True  # Cycle detected
            if node in visited:
                return False

            visited.add(node)
            stack.add(node)

            # Check all parents of this node
            for parent in self.get_parents(node):
                if visit(parent):
                    return True

            stack.remove(node)  # Backtrack
            return False

        for n in self.nodes:
            if n not in visited:
                if visit(n):
                    return False  # Found a cycle
        return True  # No cycles found


if __name__ == "__main__":
    dag = BayesNetStructure()
    print("=== Architect's DAG Validation ===")

    if dag.check_for_cycles():
        print(" Graph Check: Valid DAG (No Cycles).")
        order = dag.topological_sort()
        print(f" Sorting Check: {order}")
    else:
        print(" Graph Check: Cycle Detected!")
