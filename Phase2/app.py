from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/yadyar_risk_model.joblib")

st.set_page_config(page_title="YadYar Lite", page_icon="🎓", layout="centered")
st.title("🎓 YadYar Lite")
st.caption("Early warning demo for at-risk student prediction")

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. First run: "
        "`python src/phase2_pipeline.py --data-dir data --module AAA --presentation 2013J`"
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = float(bundle["threshold"])
options = bundle["dataset_options"]
model_display_name = bundle.get("model_display_name", "Selected model")


def select(label: str, column: str, default: str = "Unknown") -> str:
    values = options.get(column) or [default]
    return st.selectbox(label, values)


with st.form("student_form"):
    code_module = select("Module", "code_module")
    code_presentation = select("Presentation", "code_presentation")
    gender = select("Gender", "gender")
    region = select("Region", "region")
    highest_education = select("Highest education", "highest_education")
    imd_band = select("Deprivation band", "imd_band")
    age_band = select("Age band", "age_band")
    disability = select("Disability recorded", "disability")

    num_of_prev_attempts = st.number_input("Previous attempts", 0, 10, 0)
    studied_credits = st.number_input("Studied credits", 0, 360, 60)
    total_clicks_30d = st.number_input("Total clicks in first 30 days", 0, 100000, 100)
    active_days_30d = st.number_input("Active days in first 30 days", 0, 31, 10)
    unique_sites_30d = st.number_input("Unique learning resources", 0, 10000, 20)
    assessment_count_30d = st.number_input("Assessments submitted in first 30 days", 0, 20, 1)
    avg_score_30d = st.number_input("Average early assessment score", 0.0, 100.0, 60.0)

    avg_clicks_per_active_day = (
        total_clicks_30d / active_days_30d if active_days_30d else 0.0
    )
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    row = pd.DataFrame([
        {
            "code_module": code_module,
            "code_presentation": code_presentation,
            "gender": gender,
            "region": region,
            "highest_education": highest_education,
            "imd_band": imd_band,
            "age_band": age_band,
            "disability": disability,
            "num_of_prev_attempts": num_of_prev_attempts,
            "studied_credits": studied_credits,
            "total_clicks_30d": total_clicks_30d,
            "active_days_30d": active_days_30d,
            "unique_sites_30d": unique_sites_30d,
            "avg_clicks_per_active_day": avg_clicks_per_active_day,
            "assessment_count_30d": assessment_count_30d,
            "avg_score_30d": avg_score_30d,
        }
    ])
    probability = float(model.predict_proba(row)[0, 1])
    at_risk = probability >= threshold

    st.write(f"Model: **{model_display_name}**")
    st.metric("Predicted at-risk probability", f"{probability:.1%}")
    st.write(f"Decision threshold: **{threshold:.0%}**")
    if at_risk:
        st.warning("At-risk flag: recommend a supportive human review or early intervention.")
    else:
        st.success("No at-risk flag at the selected operating point.")
    st.info(
        "This is an early-warning estimate, not a final judgment. It should support, "
        "not replace, educational staff decisions."
    )
