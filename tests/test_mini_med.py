# test_mini_med.py

from inference_engine import Factor, variable_elimination

# --- Domain Definitions ---
domains = {
    "Smoking": [True, False],
    "Cancer":  [True, False],
    "Cough":   [True, False],
}

# --- CPTs (built by Member 3, or hardcoded here for testing) ---

# P(Smoking)
cpt_smoking = Factor(
    variables=["Smoking"],
    domains=domains,
    table={
        (True,):  0.30,
        (False,): 0.70,
    }
)

# P(Cancer | Smoking)
cpt_cancer = Factor(
    variables=["Cancer", "Smoking"],
    domains=domains,
    table={
        (True,  True):  0.90,  # P(Cancer=T | Smoking=T)
        (False, True):  0.10,
        (True,  False): 0.05,  # P(Cancer=T | Smoking=F)
        (False, False): 0.95,
    }
)

# P(Cough | Cancer)
cpt_cough = Factor(
    variables=["Cough", "Cancer"],
    domains=domains,
    table={
        (True,  True):  0.80,  # P(Cough=T | Cancer=T)
        (False, True):  0.20,
        (True,  False): 0.10,  # P(Cough=T | Cancer=F)
        (False, False): 0.90,
    }
)

cpts = {
    "Smoking": cpt_smoking,
    "Cancer":  cpt_cancer,
    "Cough":   cpt_cough,
}

# --- Query: P(Cancer | Cough=True) ---
result = variable_elimination(
    target="Cancer",
    evidence={"Cough": True},
    cpts=cpts,
    domains=domains,
    elimination_order=["Smoking"]  # Smoking is the only hidden variable
)

print("=== P(Cancer | Cough=True) ===")
print(result)

# Expected (hand-calculated):
# P(Cancer=T, Cough=T) = P(Cough=T|Cancer=T)*P(Cancer=T)
#   where P(Cancer=T) = P(Cancer=T|Sm=T)*P(Sm=T) + P(Cancer=T|Sm=F)*P(Sm=F)
#                     = 0.9*0.3 + 0.05*0.7 = 0.27 + 0.035 = 0.305
# P(Cancer=T, Cough=T) = 0.80 * 0.305 = 0.244
# P(Cancer=F, Cough=T) = 0.10 * 0.695 = 0.0695
# Z = 0.244 + 0.0695 = 0.3135
# P(Cancer=T | Cough=T) ≈ 0.244 / 0.3135 ≈ 0.778
# P(Cancer=F | Cough=T) ≈ 0.0695 / 0.3135 ≈ 0.222
