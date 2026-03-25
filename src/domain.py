# domain.py
# -----------------------------------------------------------------------------
# Global Configuration: Defines the discrete states for all clinical variables.
# This is the 'Source of Truth' for Data Processing (M3) and Inference (M2).
# -----------------------------------------------------------------------------

DOMAINS = {
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

# Helpful for looping through columns in the CSV
VARIABLE_LIST = list(DOMAINS.keys())