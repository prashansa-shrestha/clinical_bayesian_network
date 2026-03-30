from src.data_pipeline import load_data, train_test_split, DataPipeline, evaluate
from src.inference_engine import variable_elimination

def main():
    # 1. Load and preprocess 
    data = load_data("data/heart.csv")
    train_data, test_data = train_test_split(data, test_ratio=0.2, seed=123)

    # 2. Setup Pipeline and Fit CPTs 
    pipeline = DataPipeline() 
    # Note: DataPipeline internally creates BayesNetStructure() from dag.py
    cpts = pipeline.fit(train_data)

    # 3. Standard Evaluation
    metrics = evaluate(cpts, test_data, pipeline)

    print("=== FINAL PROJECT RESULTS ===")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Brier Score: {metrics['brier_score']:.4f}")

    # ==========================================
    # FINAL VALIDATION SUITE
    # ==========================================
    # --- Performing Sanity Queries ---
    print("\n--- Performing Sanity Queries ---")

    # TEST 1: Marginal Probability
    marginal = variable_elimination("HeartDisease", {}, cpts, pipeline.domains)
    print(f"Population Risk: {marginal.table.get((1,), 0):.4f}")

    # TEST 2: High Risk Profile
    high_risk_ev = {"Age": "senior", "Cholesterol": "high"}
    risk_res = variable_elimination("HeartDisease", high_risk_ev, cpts, pipeline.domains)
    print(f"[TEST 2] Senior + High Chol Risk: {risk_res.table.get((1,), 0):.4f}")

    # TEST 3: Identity Check (The Fix)
    identity = variable_elimination("HeartDisease", {"HeartDisease": 1}, cpts, pipeline.domains)
    # Because HD is in evidence, the engine removes it from the table keys.
    # The value 1.0 is now stored under the empty tuple key ().
    identity_val = identity.table.get((), 0) 
    print(f"Identity Check P(HD=1|HD=1): {identity_val:.4f}")

    # 4. Visualization
    from data_pipeline import plot_age_heartdisease_distribution
    plot_age_heartdisease_distribution(train_data)

if __name__ == "__main__":
    main()