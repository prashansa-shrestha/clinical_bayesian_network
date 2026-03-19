# inference_engine.py — README
**Owner:** Member 2 — Lead Mathematician / Backend Engineer  
**Module:** `inference_engine.py`  
**Algorithm:** Variable Elimination (VE)

---

## What This Module Does

This is the core query engine of the Bayesian Network. It takes a **target variable** (e.g. `"HeartDisease"`), a set of **observed evidence** (e.g. `{"Age": "middle", "Cholesterol": "high"}`), and the **CPT tables** built from the dataset — and returns a normalized probability distribution over the target.

It does not read CSVs, build a DAG, or evaluate accuracy. It only does inference.

---

## Files Overview

| File | Owner | Purpose |
|---|---|---|
| `inference_engine.py` | Member 2 | Factor class + all VE operations — **this file** |
| `dag.py` | Member 1 | DAG definition, parent structure, cycle detection |
| `data_pipeline.py` | Member 3 | CSV loading, discretization, CPT fitting (MLE + Laplace) |
| `domains.py` | **All members** | Shared dict of variable names and their states |
| `main.py` | All members | Wires everything together, runs evaluation |
| `test_mini_med.py` | Member 2 | Verification test: Smoking → Cancer → Cough |

---

## Shared Contract — `domains.py`

**This file must be agreed on by all three members before integration.** It is a plain Python dict mapping every variable name to its list of possible discrete states. Every state name is case-sensitive — `"high"` and `"High"` are different values and will cause silent bugs.

```python
# domains.py
domains = {
    "HeartDisease": [0, 1],
    "Age":          ["young", "middle", "senior"],
    "Cholesterol":  ["low", "medium", "high"],
    "BloodPressure":["low", "medium", "high"],
    # ... all other variables
}
```

Import it everywhere:
```python
from domains import domains
```

---

## Core Data Structure — `Factor`

A `Factor` is a probability table stored as a Python dict. It is the only data structure this engine uses. Every CPT is a Factor.

```python
Factor(
    variables = ["Cancer", "Smoking"],  # node first, then parents
    domains   = domains,                # the shared dict from domains.py
    table     = {
        (True,  True):  0.90,   # Cancer=True,  Smoking=True
        (False, True):  0.10,   # Cancer=False, Smoking=True
        (True,  False): 0.05,   # Cancer=True,  Smoking=False
        (False, False): 0.95,   # Cancer=False, Smoking=False
    }
)
```

**Key rule:** the order of values in each tuple key must exactly match the order of names in `variables`.

---

## API Reference

### `restrict(factor, evidence) → Factor`
Filters a factor by fixing evidence variables to their observed values. Rows that contradict the evidence are deleted. Evidence variables are removed from scope.

```python
# Factor(["Cough", "Cancer"]) with 4 rows
# → Factor(["Cancer"]) with 2 rows (only Cough=True rows kept)
restricted = restrict(cpt_cough, {"Cough": True})
```

---

### `pointwise_product(f1, f2) → Factor`
Multiplies two factors together. Result covers the union of both variable scopes. Entries are matched on shared variables and multiplied.

```python
# f1 covers (Smoking), f2 covers (Cancer, Smoking)
# result covers (Cancer, Smoking)
product = pointwise_product(cpt_smoking, cpt_cancer)
```

---

### `marginalize(factor, variable) → Factor`
Sums out a variable from a factor by adding up all rows where that variable differs. Reduces scope by one variable.

```python
# factor covers (Cancer, Smoking)
# result covers (Cancer) — Smoking is gone
marginal = marginalize(product_factor, "Smoking")
```

---

### `normalize(factor) → Factor`
Divides all values by their total so they sum to 1.0. Always the last step.

```python
# {(True,): 0.244, (False,): 0.0695}
# → {(True,): 0.778, (False,): 0.222}
final = normalize(result_factor)
```

---

### `variable_elimination(target, evidence, cpts, domains, elimination_order=None) → Factor`
The main query function. Runs the full VE algorithm and returns `P(target | evidence)` as a normalized Factor.

```python
from inference_engine import variable_elimination

result = variable_elimination(
    target            = "HeartDisease",
    evidence          = {"Age": "middle", "Cholesterol": "high"},
    cpts              = fitted_cpts,   # dict from Member 3's fit()
    domains           = domains,       # from domains.py
    elimination_order = None           # auto-derived if not provided
)

# result.table = {(0,): 0.73, (1,): 0.27}
# → 27% probability of heart disease given the evidence
```

To get the predicted class for evaluation:
```python
predicted_state = max(result.table, key=result.table.get)[0]
```

---

## Integration Guide

### For Member 1 — The Architect

Your DAG defines the parent structure. This determines how Member 3 builds each Factor's `variables` list. The convention is:

```
Factor.variables = [node_itself, parent_1, parent_2, ...]
```

If you expose a topological sort, Member 2 can pass the reverse as `elimination_order`:
```python
order = list(reversed(dag.topological_sort()))
result = variable_elimination("HeartDisease", evidence, cpts, domains, order)
```

If you don't, the engine auto-derives an order from the hidden variables — it will work correctly either way.

---

### For Member 3 — The Data Scientist

Your `fit()` method must produce a `cpts` dict where every value is a `Factor` instance. Import the class directly:

```python
from inference_engine import Factor
```

Build each CPT like this:
```python
# Example: P(HeartDisease | Age, Cholesterol)
cpts["HeartDisease"] = Factor(
    variables = ["HeartDisease", "Age", "Cholesterol"],
    domains   = domains,
    table     = {
        (0, "young", "low"):  0.95,
        (1, "young", "low"):  0.05,
        (0, "young", "medium"): 0.90,
        (1, "young", "medium"): 0.10,
        # ... all combinations
    }
)
```

Run evaluation like this:
```python
from inference_engine import variable_elimination

for patient in test_data:
    evidence  = {v: patient[v] for v in observed_vars}
    result    = variable_elimination("HeartDisease", evidence, cpts, domains)
    predicted = max(result.table, key=result.table.get)[0]
    # compare predicted vs patient["HeartDisease"]
```

**Checklist before handing off `cpts`:**
- [ ] Every value in `cpts` is a `Factor` instance (not a plain dict)
- [ ] `Factor.variables` starts with the node, followed by its parents
- [ ] All state names match `domains.py` exactly (case-sensitive)
- [ ] Each CPT sums to 1.0 per parent configuration (Laplace smoothing applied)
- [ ] No zero probabilities anywhere in any table

---

## Verification Test

Run this before integration to confirm the engine is mathematically correct:

```bash
python test_mini_med.py
```

**Network:** `Smoking → Cancer → Cough`  
**Query:** `P(Cancer | Cough=True)`  
**Expected output:**
```
=== P(Cancer | Cough=True) ===
Factor(['Cancer'])
  {'Cancer': True}  -> 0.778470
  {'Cancer': False} -> 0.221530
```

**Hand-calculated verification:**

| Step | Calculation | Result |
|---|---|---|
| P(Cancer=T) | 0.9×0.3 + 0.05×0.7 | 0.305 |
| P(Cancer=F) | 0.1×0.3 + 0.95×0.7 | 0.695 |
| P(Cancer=T, Cough=T) | 0.80 × 0.305 | 0.244 |
| P(Cancer=F, Cough=T) | 0.10 × 0.695 | 0.0695 |
| Z (normalizing constant) | 0.244 + 0.0695 | 0.3135 |
| P(Cancer=T \| Cough=T) | 0.244 / 0.3135 | ≈ 0.7783 |
| P(Cancer=F \| Cough=T) | 0.0695 / 0.3135 | ≈ 0.2217 |

---

## Algorithm Steps (Quick Reference)

```
1. Load all CPTs as Factors
2. Restrict each Factor using evidence          → restrict()
3. Identify hidden vars = All − Target − Evidence
4. For each hidden variable H:
     a. Collect all factors that mention H
     b. Multiply them together                  → pointwise_product()
     c. Sum out H                               → marginalize()
     d. Put result back in the factor pool
5. Multiply all remaining factors               → pointwise_product()
6. Normalize                                    → normalize()
```

---

## Notes

- **Elimination order** affects performance, not correctness. For ~10 variables it doesn't matter.
- **Zero probabilities** will cause `normalize()` to raise `ZeroDivisionError`. This is intentional — fix it upstream with Laplace smoothing.
- **The engine does not validate CPTs.** If a CPT doesn't sum to 1.0 per parent config, results will be wrong silently. That's Member 3's responsibility.
