
# Implements: Factor, pointwise_product, marginalize, variable_elimination, normalize

from itertools import product as cartesian_product


class Factor:
    """
    Represents a factor (probability table) over a set of variables.
    
    Attributes:
        variables (list): Ordered list of variable names in this factor.
        domains  (dict):  Maps each variable name to its list of possible states.
                          e.g. {"Cancer": [True, False], "Smoking": [True, False]}
        table    (dict):  Maps tuples of states (aligned with self.variables)
                          to a non-negative float.
                          e.g. {(True, True): 0.9, (True, False): 0.1, ...}
    """

    def __init__(self, variables: list, domains: dict, table: dict):
        self.variables = list(variables)
        self.domains = domains          # shared reference to the global domain map
        self.table = dict(table)

    def __repr__(self):
        lines = [f"Factor({self.variables})"]
        for assignment, val in sorted(self.table.items()):
            variable_names = self.variables
            current_values = assignment
            paired_data = zip(variable_names, current_values)
            row_dictionary = dict(paired_data)
            
            formatted_line = f"  {row_dictionary} -> {val:.6f}"

            lines.append(formatted_line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Operations
# ---------------------------------------------------------------------------

def restrict(factor: Factor, evidence: dict) -> Factor:
    """
    Operation: f(X, E=e)
    ~Throws away everything that contradicts what we know to be true

    Filters a factor by fixing evidence variables to their observed values.
    Evidence variables are removed from the factor's scope.

    Args:
        factor:   The factor to restrict.
        evidence: Dict mapping variable names to observed values.
                  e.g. {"Cough": True}

    Returns:
        A new Factor with evidence variables removed from scope.
    """
    
    # the evidence variables that appear in the Factor
    ev_in_factor = {}
    for v in factor.variables:
        if v in evidence:
            val = evidence[v]
            ev_in_factor[v] = val

    if not ev_in_factor:
        return factor  # Nothing to restrict

    # Indices of non-evidence variables (kept) and evidence variables (fixed)
    kept_vars = [v for v in factor.variables if v not in ev_in_factor]
    ev_indices = {v: factor.variables.index(v) for v in ev_in_factor}


    # restrict the factor's table
    new_table = {}
    for assignment, value in factor.table.items():
        # Check if this row matches all evidence
        if all(assignment[ev_indices[v]] == ev_val
               for v, ev_val in ev_in_factor.items()):
            # Build new key with only non-evidence variables
            new_key = tuple(assignment[factor.variables.index(v)]
                            for v in kept_vars)
            new_table[new_key] = value
    
    
    # return factor with new_table, only the rows matching the evidence
    return Factor(kept_vars, factor.domains, new_table)



def pointwise_product(f1: Factor, f2: Factor) -> Factor:
    """
    Operation: ψ(X∪Y∪Z) = φ₁(X,Y) · φ₂(Y,Z)

    Combines two factors into one

    Multiplies two factors entry-by-entry over shared variables.
    The result spans the union of both variable sets.

    Args:
        f1: First factor.
        f2: Second factor.

    Returns:
        A new Factor over the union of f1 and f2's variables.
    """
    # Union of variables, preserving order (f1 first, then f2 extras)
    union_vars = list(f1.variables)
    for v in f2.variables:
        if v not in union_vars:
            union_vars.append(v)

    new_table = {}
    # Enumerate all combinations of the union domain
    union_domains = [f1.domains[v] for v in union_vars]

    for combo in cartesian_product(*union_domains):
        assignment = dict(zip(union_vars, combo))

        # Build lookup keys for f1 and f2
        key1 = tuple(assignment[v] for v in f1.variables)
        key2 = tuple(assignment[v] for v in f2.variables)

        val1 = f1.table.get(key1, 0.0)
        val2 = f2.table.get(key2, 0.0)

        new_table[combo] = val1 * val2

    return Factor(union_vars, f1.domains, new_table)


def marginalize(factor: Factor, variable: str) -> Factor:
    """
    Operation: τ(X) = Σ_y φ(X, y)
    Sums out a variable from a factor, reducing its scope by one.

    Args:
        factor:   The factor containing the variable to eliminate.
        variable: The variable name to sum out.

    Returns:
        A new Factor with `variable` removed, values summed over its states.
    
    Raises:
        ValueError: If the variable is not in the factor.
    """
    if variable not in factor.variables:
        raise ValueError(f"Variable '{variable}' not in factor scope {factor.variables}")

    var_idx = factor.variables.index(variable)
    remaining_vars = [v for v in factor.variables if v != variable]

    new_table = {}
    for assignment, value in factor.table.items():
        # Key without the marginalized variable
        new_key = tuple(v for i, v in enumerate(assignment) if i != var_idx)
        # Sum out the variables
        new_table[new_key] = new_table.get(new_key, 0.0) + value

    return Factor(remaining_vars, factor.domains, new_table)


def normalize(factor: Factor) -> Factor:
    """
    Operation: P(X) = φ(X) / Σ_x φ(x)
    Normalizes a factor so its values sum to 1.0.

    Args:
        factor: The factor to normalize (must have non-zero sum).

    Returns:
        A new normalized Factor.

    Raises:
        ZeroDivisionError: If all factor values are zero.
    """
    total = sum(factor.table.values())
    if total == 0.0:
        raise ZeroDivisionError("Cannot normalize a factor with all-zero values.")
    
    normalized_table = {k: v / total for k, v in factor.table.items()}
    return Factor(factor.variables, factor.domains, normalized_table)


# ---------------------------------------------------------------------------
# Variable Elimination (Core Query Engine)
# ---------------------------------------------------------------------------

def variable_elimination(
    target: str,
    evidence: dict,
    cpts: dict,
    domains: dict,
    elimination_order: list = None
) -> Factor:
    """
    Variable Elimination Algorithm.
    Computes P(Target | Evidence) using factor operations.

    Args:
        target:            The query variable name. e.g. "Cancer"
        evidence:          Dict of observed variables. e.g. {"Cough": True}
        cpts:              Dict mapping each variable name to its Factor (CPT).
                           Built by Member 3 (Data Scientist).
        domains:           Dict mapping variable names to their possible states.
                           e.g. {"Cancer": [True, False], "Cough": [True, False]}
        elimination_order: Optional list of hidden variables in elimination order.
                           If None, uses the natural order (all non-target, non-evidence).

    Returns:
        Normalized Factor over [target] representing P(Target | Evidence).

    Algorithm:
        1. Convert all CPTs to factors.
        2. Restrict factors using evidence.
        3. For each hidden variable H:
           a. Collect factors containing H.
           b. Multiply them together.
           c. Sum out H.
           d. Return result to factor pool.
        4. Multiply remaining factors.
        5. Normalize.
    """
    # Step 1: Initialize factor pool from CPTs
    factors = []
    for var_name, cpt in cpts.items():
        if isinstance(cpt, Factor):
            factors.append(Factor(cpt.variables, domains, dict(cpt.table)))
        else:
            raise TypeError(f"CPT for '{var_name}' must be a Factor instance.")

    # Step 2: Apply evidence restrictions
    factors = [restrict(f, evidence) for f in factors]
    # Remove degenerate factors (empty scope / scalar) if needed
    factors = [f for f in factors if f.table]

    # Step 3: Determine hidden variables
    all_vars = set(domains.keys())
    observed = set(evidence.keys())
    hidden_vars = all_vars - {target} - observed

    if elimination_order is None:
        elimination_order = list(hidden_vars)

    # Step 4: Eliminate hidden variables one by one
    for h_var in elimination_order:
        if h_var not in hidden_vars:
            continue  # Skip if already eliminated or not applicable

        # Collect all factors that mention h_var
        relevant = [f for f in factors if h_var in f.variables]
        irrelevant = [f for f in factors if h_var not in f.variables]

        if not relevant:
            continue

        # Multiply all relevant factors together
        product_factor = relevant[0]
        for f in relevant[1:]:
            product_factor = pointwise_product(product_factor, f)

        # Sum out the hidden variable
        summed_out = marginalize(product_factor, h_var)

        # Return new factor to the pool
        factors = irrelevant + [summed_out]

    # Step 5: Multiply all remaining factors
    if not factors:
        raise RuntimeError("No factors remaining after elimination.")

    result = factors[0]
    for f in factors[1:]:
        result = pointwise_product(result, f)

    # Restrict to target variable only (in case extras remain)
    non_target = [v for v in result.variables if v != target]
    for v in non_target:
        result = marginalize(result, v)

    # Step 6: Normalize
    return normalize(result)