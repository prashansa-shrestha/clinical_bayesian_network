import streamlit as st
import pandas as pd
import graphviz
from main import DataPipeline, variable_elimination, load_data, train_test_split, evaluate
from src.dag import BayesNetStructure
from src.domain import DOMAINS

# 1. Page Configuration
st.set_page_config(page_title="Heart Disease AI Predictor", layout="wide")

# Custom CSS for Light Theme (White Background)
st.markdown("""
    <style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 15px; border-radius: 10px; border: 1px solid #EEEEEE; }
    h1, h2, h3 { color: #1E1E1E; font-family: 'Segoe UI', sans-serif; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: #555; text-align: center; padding: 10px; border-top: 1px solid #EEEEEE; font-size: 14px; z-index: 100; }
    .stTable { background-color: white; border: 1px solid #EEE; }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Data & Model ---
@st.cache_resource
def initialize_system():
    # 1. Update the path to include the 'data/' folder
    raw_list = load_data("data/heart.csv") 
    df = pd.DataFrame(raw_list)
    # 2. Split data
    train_data, test_data = train_test_split(raw_list, test_ratio=0.2, seed=123)
    
    # 3. Setup Pipeline
    pipeline = DataPipeline()
    
    # FIX: pipeline.fit(train_data) already returns a dictionary { "VarName": FactorObject }
    # Assign it directly to cpts
    cpts = pipeline.fit(train_data)
    
    # 4. Calculate performance metrics
    metrics = evaluate(cpts, test_data, pipeline)
    
    # 5. Initialize DAG structure for visualization
    structure = BayesNetStructure()
    
    return pipeline, cpts, metrics, df, structure


# Load resources
pipeline, cpts, metrics, df_raw, bn_structure = initialize_system()

# --- Sidebar ---
st.sidebar.header("Patient Clinical Profile")
user_evidence = {}

# Map technical variable names to user-friendly display names
variable_labels = {
    "Age": "Age Group",
    "Sex": "Sex",
    "Cholesterol": "Cholesterol Level",
    "BloodPressure": "Blood Pressure",
    "Thalach": "Max Heart Rate Achieved",
    "Cp": "Chest Pain Type",
    "ExerciseAngina": "Chest Pain after Exercise"
}

for var in ["Age", "Sex", "Cholesterol", "BloodPressure", "Thalach", "Cp", "ExerciseAngina"]:
    user_evidence[var] = st.sidebar.selectbox(variable_labels[var], DOMAINS[var])

# --- Main UI ---
st.title("Clinical Bayesian Network")
st.write("An Explainable AI (XAI) approach to Heart Disease Risk Assessment.")

tabs = st.tabs(["Live Predictor", "Project Details", "Network Structure", "Behind the Scenes", "References"])

# --- TAB 1: Live Predictor ---
with tabs[0]:
    st.subheader("Inference Result")
    
    # This now receives the correct 'cpt_dict'
    result = variable_elimination("HeartDisease", user_evidence, cpts, pipeline.domains)
    prob_val = result.table.get((1,), 0)
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.metric(label="Heart Disease Probability", value=f"{prob_val:.2%}")
        if prob_val > 0.5:
            st.error("Status: High Risk Identified")
        else:
            st.success("Status: Low Risk Identified")
            
    with col2:
        st.write("**Active Evidence Summary**")
        evidence_data = [{"Clinical Variable": k, "Selected Value": v} for k, v in user_evidence.items()]
        st.table(pd.DataFrame(evidence_data))

# --- TAB 2: Project Details ---
with tabs[1]:
    st.subheader("Project Overview")
    st.markdown(f"""
    **Dataset Information** The model is trained on the [UCI Heart Disease Dataset (Cleveland Clinic Foundation)](https://archive.ics.uci.edu/dataset/45/heart+disease).  
    - Population Size: {len(df_raw)} patients  
    - Attributes: 14 clinical features
    
    **Model Performance** - Accuracy Score: {metrics['accuracy']*100:.2f}%  
    - Brier Score: {metrics['brier_score']:.4f}
    """)
    st.divider()
    st.write("### Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

# --- TAB 3: Network Structure ---
with tabs[2]:
    st.subheader("Causal Graph Architecture")
    dot = graphviz.Digraph()
    dot.attr(bgcolor='white')
    for child, parents in bn_structure.structure.items():
        dot.node(child, child, style="filled", color="#F0F2F6", fontcolor="#1E1E1E", shape="box")
        for p in parents:
            dot.edge(p, child, color="#BDC3C7")
    st.graphviz_chart(dot)

# --- TAB 4: Behind the Scenes ---
with tabs[3]:
    st.subheader("Mathematical Foundations")
    st.write("This system uses Variable Elimination to compute exact posterior probabilities.")
    st.latex(r"P(Target | Evidence) = \frac{\sum_{Hidden} \prod P(Node_i | Parents_i)}{P(Evidence)}")
    st.markdown(r"""
    1. **Factor Restriction**: Slicing CPTs based on patient symptoms.
    2. **Marginalization**: Summing out variables not present in the query.
    3. **Normalization**: Ensuring the results sum to 1.
    """)

# --- TAB 5: References ---
with tabs[4]:
    st.subheader("References")
    st.markdown("""
        - **Koller, D., & Friedman, N. (2009).** *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
        - **Detrano, R., et al. (1989).** *International application of a new probability algorithm for the diagnosis of coronary artery disease*. American Journal of Cardiology.
        - **Russell, S., & Norvig, P.** *Artificial Intelligence: A Modern Approach*. Pearson (Bayesian Inference Chapters).
        - **UCI Machine Learning Repository.** *Heart Disease Dataset*. [Link to Repository](https://archive.ics.uci.edu/dataset/45/heart+disease). 
    """)

# --- Footer ---
# --- Footer ---
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #555;
        text-align: center; /* This handles the horizontal centering */
        padding: 15px 0;
        border-top: 1px solid #EEEEEE;
        font-size: 14px;
        z-index: 100;
        line-height: 1.6;
    }
    </style>
    <div class="footer">
        Developed by Prashansa, Sadina, and Vaibhav<br>
        Fulfillment of Coursework for Artificial Intelligence [CT653]
    </div>
    """, unsafe_allow_html=True)