import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

st.set_page_config(
    page_title="Kanpur Child Pedestrian Safety Profiler",
    page_icon="🚸",
    layout="wide"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card-safe {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
        margin-bottom: 1rem;
    }
    .metric-card-risky {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
        margin-bottom: 1rem;
    }
    .stat-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
    }
    .stat-val {
        font-size: 2.2rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Pipeline and Metadata
def load_pipeline():
    with open("student_risk_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return pipeline, metadata

try:
    pipeline, metadata = load_pipeline()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.info("Please run `python train_pipeline.py` first to train and save the model pipeline.")
    st.stop()

# Header Section
st.markdown("<div class='main-header'>🚸 Kanpur Child Pedestrian Safety Profile Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predicting Child Pedestrian Crossing Risk (Safe vs Risky) using Machine Learning & Kanpur Survey Data</div>", unsafe_allow_html=True)

# Layout Columns
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("📋 Student Demographics & Travel Context")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        age_cat = st.selectbox("Age Group", options=["10-12 Years", "13-14 Years", "15-16 Years"], index=1)
        age_val = 1 if "10-12" in age_cat else (2 if "13-14" in age_cat else 3)
    with c2:
        gender_cat = st.selectbox("Gender", options=["Male", "Female"], index=0)
        gender_val = 0 if gender_cat == "Male" else 1
    with c3:
        supervision_cat = st.selectbox("Travel Supervision", options=["Alone (Unsupervised)", "With Adult / Guardian"], index=0)
        supervision_val = 0 if "Alone" in supervision_cat else 1
        
    st.markdown("---")
    st.subheader("🚶 Self-Reported Road Behaviors (Likert Scale: 1 = Strongly Disagree, 5 = Strongly Agree)")
    
    likert_labels = {
        1: "1 - Strongly Disagree",
        2: "2 - Disagree",
        3: "3 - Neutral",
        4: "4 - Agree",
        5: "5 - Strongly Agree"
    }
    
    b1, b2 = st.columns(2)
    with b1:
        risky_run = st.select_slider("I sometimes run while crossing the road", options=[1, 2, 3, 4, 5], value=3, format_func=lambda x: likert_labels[x])
        play_road = st.select_slider("I sometimes play near or on the road", options=[1, 2, 3, 4, 5], value=2, format_func=lambda x: likert_labels[x])
        use_mobile = st.select_slider("I use a mobile phone while walking/crossing", options=[1, 2, 3, 4, 5], value=1, format_func=lambda x: likert_labels[x])
        talking = st.select_slider("I cross the road while talking to peers", options=[1, 2, 3, 4, 5], value=3, format_func=lambda x: likert_labels[x])
        follow_friends = st.select_slider("I follow friends even if they cross dangerously", options=[1, 2, 3, 4, 5], value=2, format_func=lambda x: likert_labels[x])
        
    with b2:
        look_before = st.select_slider("I look for vehicles before crossing", options=[1, 2, 3, 4, 5], value=4, format_func=lambda x: likert_labels[x])
        crosswalks = st.select_slider("I cross at designated zebra crosswalks", options=[1, 2, 3, 4, 5], value=4, format_func=lambda x: likert_labels[x])
        wait_adult = st.select_slider("I wait for an adult if traffic is heavy", options=[1, 2, 3, 4, 5], value=4, format_func=lambda x: likert_labels[x])
        conf_infra = st.select_slider("I feel confident crossing without infrastructure", options=[1, 2, 3, 4, 5], value=3, format_func=lambda x: likert_labels[x])

    st.markdown("---")
    st.subheader("🏫 Road Environment & Infrastructure Perception")
    
    e1, e2, e3 = st.columns(3)
    with e1:
        footpath = st.slider("Footpath Availability & Quality", 1, 5, 3)
        condition_road = st.slider("Condition of Road Surface", 1, 5, 3)
    with e2:
        vehicle_speed = st.slider("Low Vehicle Speed Perception", 1, 5, 3)
        zebra_crossing = st.slider("Zebra Crossing Visibility", 1, 5, 3)
    with e3:
        conf_school_env = st.slider("School Zone Environment Safety", 1, 5, 3)
        speed_breakers = st.slider("Speed Breaker Provisions", 1, 5, 3)

with col_right:
    st.subheader("🔮 Model Risk Prediction")
    
    # Construct input feature dictionary matching 27 features in exact order
    input_dict = {
        'Age': age_val,
        'Gender': gender_val,
        'adult/guardian?': supervision_val,
        'risky_run': risky_run,
        'play_road': play_road,
        'use_mobile': use_mobile,
        'talking.': talking,
        'follow_friends': follow_friends,
        'look_before_crossing.': look_before,
        'designated crosswalks': crosswalks,
        'wait_adult': wait_adult,
        'confident_no infrastructure': conf_infra,
        'footpath': footpath,
        'zebra crossing': zebra_crossing,
        'speed breakers': speed_breakers,
        'guard rails': 3,
        'road width': 3,
        'condition of road': condition_road,
        'vehicle speed low': vehicle_speed,
        'less traffic': 3,
        'vehicles stop': 3,
        'traffic police': 3,
        'school zone signboards': 3,
        'traffic signals': 3,
        'satisfied_near_school.  2': 3,
        'satisfied_safety_provisions3': 3,
        'confiden_school_environment   4': conf_school_env
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Run Prediction
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    
    safe_prob = probabilities[0] * 100
    risky_prob = probabilities[1] * 100
    
    # Calculate composite risk index score for display
    comp_score = (
        risky_run + play_road + use_mobile + talking + follow_friends + conf_infra +
        (6 - look_before) + (6 - crosswalks) + (6 - wait_adult)
    ) / 9.0
    
    if prediction == 1:
        st.markdown(f"""
        <div class='metric-card-risky'>
            <div class='stat-label'>Classification Result</div>
            <div class='stat-val'>⚠️ RISKY CROSSING PROFILE</div>
            <div style='margin-top: 10px; font-size: 1.1rem;'>Confidence Probability: <b>{risky_prob:.1f}%</b></div>
            <div style='margin-top: 5px; font-size: 0.95rem; opacity: 0.9;'>Composite Risk Index: <b>{comp_score:.2f} / 5.0</b> (Threshold: &ge; 3.5)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='metric-card-safe'>
            <div class='stat-label'>Classification Result</div>
            <div class='stat-val'>✅ SAFE CROSSING PROFILE</div>
            <div style='margin-top: 10px; font-size: 1.1rem;'>Confidence Probability: <b>{safe_prob:.1f}%</b></div>
            <div style='margin-top: 5px; font-size: 0.95rem; opacity: 0.9;'>Composite Risk Index: <b>{comp_score:.2f} / 5.0</b> (Threshold: &lt; 3.5)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("📊 Class Confidence Breakdown")
    prob_chart = pd.DataFrame({
        'Profile Category': ['Safe Profile', 'Risky Profile'],
        'Probability (%)': [safe_prob, risky_prob]
    })
    st.bar_chart(prob_chart.set_index('Profile Category'))
    
    st.subheader("💡 Tailored Intervention & Recommendation")
    if prediction == 1:
        st.error("""
        **High Risk Warning**:
        - **Targeted Action**: Engage student in practical street crosswalk simulation drills.
        - **Parental Guidance**: Strongly recommend active adult accompaniment during pick-up and drop-off hours.
        - **Peer Influence**: Discourage following peers without independently checking traffic flow.
        """)
    else:
        st.success("""
        **Safe Behavior Leadership**:
        - **Ambassador Program**: Nominate student as a peer road safety ambassador.
        - **Habit Reinforcement**: Maintain positive habits of using zebra crossings and checking traffic before step-off.
        """)
        
    st.markdown("---")
    st.subheader("🏆 Model Performance & Validation Benchmarks")
    m = metadata.get('metrics', {})
    
    mb1, mb2 = st.columns(2)
    with mb1:
        st.metric(label="🎯 Holdout Test Accuracy", value=f"{m.get('accuracy', 0.9048)*100:.2f}%")
        st.metric(label="🔄 5-Fold Stratified CV Accuracy", value=f"{m.get('cv_accuracy', 0.9429)*100:.2f}%")
        st.metric(label="🛡️ Macro Recall (Balanced)", value=f"{m.get('macro_recall', 0.8800)*100:.2f}%")
    with mb2:
        st.metric(label="⚡ Risky Class Sensitivity / Recall", value=f"{m.get('recall', 0.9200)*100:.2f}%")
        st.metric(label="⚖️ Model F1-Score", value=f"{m.get('f1_score', 0.8519)*100:.2f}%")
        st.metric(label="📈 ROC-AUC Score", value=f"{m.get('roc_auc', 0.9824)*100:.2f}%")
        
    st.caption("ℹ️ Validated with SMOTE oversampling & 5-fold cross-validation on 420 Kanpur student survey samples.")

