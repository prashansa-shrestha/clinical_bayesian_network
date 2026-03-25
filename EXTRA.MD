# P(Cancer | Smoking) stored as a Factor
table = {
    (True,  True):  0.90,   # Cancer=True,  Smoking=True
    (False, True):  0.10,   # Cancer=False, Smoking=True
    (True,  False): 0.05,   # Cancer=True,  Smoking=False
    (False, False): 0.95,   # Cancer=False, Smoking=False
}

The tuple key `(True, True)` means: Cancer=True AND Smoking=True. The order of values in the tuple matches the order of variables in the Factor's `variables` list.

Every CPT is a Factor. But during computation, you also create *intermediate* Factors that don't directly correspond to any single CPT — they're the result of multiplying and summing tables together.

---

# Part 4: The Core Question — What are We Computing?

You're given a **query** like:

> *"What is the probability of Cancer, given that the patient is Coughing?"*

Written mathematically: **P(Cancer | Cough=True)**

This is called **probabilistic inference** — using known evidence to update our belief about an unknown variable.

The naive approach would be to look at every possible combination of all variables and sum up. That becomes impossibly slow with 10+ variables. That's why we use **Variable Elimination**.

---

# Part 5: Variable Elimination — The Big Picture

Variable Elimination (VE) is a smart algorithm that answers the query **P(Target | Evidence)** efficiently by eliminating one variable at a time.

Here's the intuition: instead of considering every combination of every variable simultaneously, we process variables one at a time, collapsing the problem as we go.

The steps are:
```
1. Start with all CPT tables (as Factors)
2. Lock in the evidence  →  "Cough=True" means we delete all 
                            table rows where Cough=False
3. Identify hidden variables  →  variables that are neither 
                                  the target nor evidence
4. Eliminate each hidden variable one by one:
     a. Collect all factors that mention this variable
     b. Multiply them together into one big factor
     c. Sum out the variable (collapse it away)
5. Multiply whatever factors remain
6. Normalize so probabilities sum to 1
```

Let's walk through the Mini-Med example in full detail.

---

# Part 6: Full Worked Example (Mini-Med)

**Network:** Smoking → Cancer → Cough

**Query:** P(Cancer | Cough=True)

**CPTs:**
```
P(Smoking=True)  = 0.30,  P(Smoking=False) = 0.70

P(Cancer=T | Smoking=T) = 0.90,  P(Cancer=F | Smoking=T) = 0.10
P(Cancer=T | Smoking=F) = 0.05,  P(Cancer=F | Smoking=F) = 0.95

P(Cough=T | Cancer=T) = 0.80,  P(Cough=F | Cancer=T) = 0.20
P(Cough=T | Cancer=F) = 0.10,  P(Cough=F | Cancer=F) = 0.90
```

### Step 1 — Start with 3 factors
- φ₁ = P(Smoking)
- φ₂ = P(Cancer | Smoking)
- φ₃ = P(Cough | Cancer)

### Step 2 — Apply evidence: Cough=True
Filter φ₃ to only keep rows where Cough=True:
```
φ₃ becomes:
  Cancer=True  → 0.80
  Cancer=False → 0.10
```

This factor now only talks about Cancer (Cough is gone, fixed at True).

### Step 3 — Identify hidden variables
- Target = Cancer
- Evidence = {Cough=True}
- All variables = {Smoking, Cancer, Cough}
- Hidden = All − Target − Evidence = **{Smoking}**

### Step 4 — Eliminate Smoking

**4a.** Which factors mention Smoking? → φ₁ and φ₂

**4b.** Multiply φ₁ × φ₂ (pointwise product):

φ₁ talks about (Smoking). φ₂ talks about (Cancer, Smoking). The product talks about (Cancer, Smoking):
```
(Cancer=T, Smoking=T): 0.90 × 0.30 = 0.270
(Cancer=F, Smoking=T): 0.10 × 0.30 = 0.030
(Cancer=T, Smoking=F): 0.05 × 0.70 = 0.035
(Cancer=F, Smoking=F): 0.95 × 0.70 = 0.665
```

**4c.** Sum out Smoking (add up rows where Cancer value is the same):
```
Cancer=True:  0.270 + 0.035 = 0.305
Cancer=False: 0.030 + 0.665 = 0.695
```

Smoking is gone. We now have a new factor φ₄(Cancer) = {True: 0.305, False: 0.695}. This is actually just **P(Cancer)** — the marginal probability.

### Step 5 — Multiply remaining factors

Remaining factors: φ₃ (restricted) and φ₄
```
Cancer=True:  0.80 × 0.305 = 0.244
Cancer=False: 0.10 × 0.695 = 0.0695
```

### Step 6 — Normalize
```
Total Z = 0.244 + 0.0695 = 0.3135

P(Cancer=True  | Cough=True) = 0.244  / 0.3135 ≈ 0.778
P(Cancer=False | Cough=True) = 0.0695 / 0.3135 ≈ 0.222
```

**Result:** Given the patient is coughing, there is a **77.8% chance they have cancer.** That's the engine working correctly.

---

# Part 7: Why Each Function Exists in Your Code

Now the code should make complete sense:

| Function | What it does | When it's called |
|---|---|---|
| `Factor` class | Stores a probability table as a dict | Always — it's the base data structure |
| `restrict()` | Applies evidence — deletes rows that don't match | Step 2 of VE |
| `pointwise_product()` | Multiplies two tables together over shared variables | Step 4b of VE |
| `marginalize()` | Sums out one variable from a table | Step 4c of VE |
| `normalize()` | Divides everything by total so it sums to 1 | Final step of VE |
| `variable_elimination()` | Orchestrates all of the above in the right order | The main query function called by Member 3 |

---

# Part 8: End-to-End Project Flow

Here's how all three members connect:
```
UCI Heart Disease Dataset (CSV)
         │
         ▼
  Member 3 reads CSV, 
  bins continuous values 
  (e.g. Age: 45 → "middle"),
  counts occurrences,
  applies Laplace smoothing,
  builds a dict of Factor objects
         │
         ▼ cpts = {"HeartDisease": Factor, "Age": Factor, ...}
         │
         ▼
  Member 1 defines the DAG:
  which variables are parents 
  of which. This determines 
  what variables appear in 
  each Factor's scope.
         │
         ▼ structure = {"Cancer": ["Smoking"], "Cough": ["Cancer"]}
         │
         ▼
  YOUR ENGINE (Member 2):
  Takes (target, evidence, cpts, domains)
  Runs Variable Elimination
  Returns P(HeartDisease | patient's test results)
         │
         ▼
  Member 3 evaluates: 
  runs this on test data,
  computes accuracy / confusion matrix