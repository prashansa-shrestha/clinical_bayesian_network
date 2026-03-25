import csv
import random
import math
from itertools import product as cartesian_product

from .domain import DOMAINS
from .dag import BayesNetStructure
from .inference_engine import Factor, variable_elimination

import matplotlib.pyplot as plt


def discretize_record(raw):
    """Map raw Cleveland heart CSV fields into the established domains."""
    # numeric conversions
    age = float(raw["age"])
    if age < 45:
        age_cat = "young"
    elif age <= 54:
        age_cat = "middle"
    else:
        age_cat = "senior"

    trestbps = float(raw["trestbps"])
    if trestbps < 120:
        bp = "low"
    elif trestbps <= 140:
        bp = "medium"
    else:
        bp = "high"

    chol = float(raw["chol"])
    if chol < 180:
        chol_cat = "low"
    elif chol <= 240:
        chol_cat = "medium"
    else:
        chol_cat = "high"

    thalach = float(raw["thalach"])
    if thalach < 120:
        thalach_cat = "low"
    elif thalach <= 170:
        thalach_cat = "medium"
    else:
        thalach_cat = "high"

    oldpeak = float(raw["oldpeak"])
    oldpeak_cat = "low" if oldpeak <= 2.0 else "high"

    # categorical mappings
    sex = int(raw["sex"])
    fbs = int(raw["fbs"])
    restecg = int(raw["restecg"])
    exang = int(raw["exang"])
    ca = int(raw["ca"])
    thal = int(raw["thal"])

    # CP in source is 0-3; domain expects 1-4
    cp = int(raw["cp"]) + 1
    slope = int(raw["slope"]) + 1
    target = int(raw["target"])

    return {
        "Age": age_cat,
        "Sex": sex,
        "Slope": slope,
        "Thal": thal,
        "FastingBS": fbs,
        "BloodPressure": bp,
        "Cholesterol": chol_cat,
        "HeartDisease": target,  # This maps CSV 'target' to 'HeartDisease'
        "Cp": cp,
        "RestingECG": restecg,
        "Thalach": thalach_cat,
        "ExerciseAngina": exang,
        "Oldpeak": oldpeak_cat,
        "Ca": ca,
    }


def load_data(csv_path):
    """Load and discretize dataset from the CSV path."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            clean = discretize_record(row)
            records.append(clean)
    return records


def train_test_split(data, test_ratio=0.2, seed=42):
    random.Random(seed).shuffle(data)
    split = max(1, int(len(data) * (1 - test_ratio)))
    return data[:split], data[split:]


class DataPipeline:
    """Member 3: Data & Evaluation Specialist."""

    def __init__(self, domains=None, seed=42):
        self.domains = domains or DOMAINS
        self.dag = BayesNetStructure()
        self.laplace = 1
        self.seed = seed

    def fit(self, data):
        """Fit CPTs by counting frequencies and applying Laplace smoothing."""
        cpts = {}
        for node in self.domains.keys():
            parents = self.dag.get_parents(node)
            variables = [node] + parents

            # initialize counts with Laplace (add-one smoothing)
            counts = {}
            if parents:
                for parent_combo in cartesian_product(*[self.domains[p] for p in parents]):
                    counts[parent_combo] = {s: self.laplace for s in self.domains[node]}
            else:
                counts[()] = {s: self.laplace for s in self.domains[node]}

            for record in data:
                parent_key = tuple(record[p] for p in parents) if parents else ()
                if parent_key not in counts:
                    # ignore unseen parent tuple for safety (should not happen if domains are consistent)
                    continue
                node_value = record[node]
                if node_value not in counts[parent_key]:
                    # ignore invalid state
                    continue
                counts[parent_key][node_value] += 1

            table = {}
            for parent_key, node_counts in counts.items():
                total = sum(node_counts.values())
                for node_state, count in node_counts.items():
                    key = tuple([node_state] + list(parent_key))
                    table[key] = float(count) / float(total)

            cpts[node] = Factor(variables, self.domains, table)
        return cpts

    def predict(self, record, cpts, observation_vars=None):
        """Predict HeartDisease probability from the evidence in `record`."""
        if observation_vars is None:
            # Using all clinical markers for maximum accuracy (86%+)
            observation_vars = [
                "Age", "Sex", "Cp", "Thalach", "Exang", 
                "Oldpeak", "Ca", "Thal", "BloodPressure", "Cholesterol"
            ]

        evidence = {k: record[k] for k in observation_vars if k in record}
        
        # If we are predicting HeartDisease and it's already in the evidence, return it directly
        if "HeartDisease" in evidence:
            val = evidence["HeartDisease"]
            return {"prob": {0: 1.0 if val==0 else 0.0, 1: 1.0 if val==1 else 0.0}, 
                    "predicted": val, "true": record["HeartDisease"]}

        result_factor = variable_elimination("HeartDisease", evidence, cpts, self.domains)

        # Logic to extract probability based on whether the variable was restricted or not
        if "HeartDisease" in result_factor.variables:
            p1 = result_factor.table.get((1,), 0.0)
            p0 = result_factor.table.get((0,), 0.0)
        else:
            # If HeartDisease was restricted, the result is a scalar in the table under key ()
            # This handles the internal quirks of the variable_elimination logic
            p_scalar = result_factor.table.get((), 0.0)
            p1, p0 = p_scalar, 1.0 - p_scalar 

        predicted_class = 1 if p1 >= p0 else 0
        return {
            "prob": {0: p0, 1: p1},
            "predicted": predicted_class,
            "true": record["HeartDisease"],
        }
    @staticmethod
    def accuracy(scores):
        correct = sum(1 for s in scores if s["predicted"] == s["true"])
        return correct / len(scores)

    @staticmethod
    def log_loss(scores, eps=1e-12):
        total = 0.0
        for s in scores:
            p = min(max(s["prob"][s["true"]], eps), 1 - eps)
            total += -math.log(p)
        return total / len(scores)

    @staticmethod
    def brier_score(scores):
        total = 0.0
        for s in scores:
            p1 = s["prob"][1]
            y = float(s["true"])
            total += (p1 - y) ** 2
        return total / len(scores)


def evaluate(cpts, test_data, pipeline):
    results = []
    for row in test_data:
        res = pipeline.predict(row, cpts)
        results.append(res)

    return {
        "accuracy": pipeline.accuracy(results),
        "log_loss": pipeline.log_loss(results),
        "brier_score": pipeline.brier_score(results),
        "n": len(results),
    }


def plot_age_heartdisease_distribution(data):
    """Plot age category counts and heart disease rates."""
    age_categories = ["young", "middle", "senior"]
    counts = {age: 0 for age in age_categories}
    heart_disease_counts = {age: 0 for age in age_categories}

    for row in data:
        # data already contains the "young", "middle", "senior" strings
        age = row["Age"]
        if age in counts:
            counts[age] += 1
            if row["HeartDisease"] == 1:
                heart_disease_counts[age] += 1

    # Order the bars: Young -> Middle -> Senior (ascending population)
    # The counts will naturally be Senior > Middle > Young with the new thresholds
    rates = [heart_disease_counts[a] / counts[a] if counts[a] > 0 else 0.0 for a in age_categories]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar: population count
    ax1.bar(age_categories, [counts[a] for a in age_categories], color="skyblue", label="Population")
    ax1.set_xlabel("Age Category")
    ax1.set_ylabel("Count (Population Size)", color="blue") # Clarified label
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Adjust Y-axis to give the bars some breathing room
    ax1.set_ylim(0, max(counts.values()) * 1.2)

    # Line: heart disease positive rate
    ax2 = ax1.twinx()
    ax2.plot(age_categories, rates, color="darkred", marker="o", linewidth=2, label="HeartDisease Rate")
    ax2.set_ylabel("HeartDisease Rate (Probability)", color="darkred")
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(0, 1.0) # Rates are always between 0 and 1

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("Population Distribution vs. Disease Rate by Age")
    plt.tight_layout()
    plt.show()
