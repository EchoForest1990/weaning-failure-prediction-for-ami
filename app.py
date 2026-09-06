
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AMI Weaning Failure Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# 加载训练时保存的对象
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('MICE_Random Forest_final.pkl')
        scaler = joblib.load('scaler.pkl')
        final_features = joblib.load('final_features.pkl')
        continuous_cols = joblib.load('continuous_cols.pkl')
        return model, scaler, final_features, continuous_cols
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}\nEnsure all .pkl files are in the same directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Load failed: {str(e)}")
        return None, None, None, None

model, scaler, final_features, continuous_cols = load_artifacts()
if model is None:
    st.stop()

# ------------------------------------------------------------
# 提取子缩放参数（针对最终选中的连续特征）
# ------------------------------------------------------------
final_continuous = [f for f in final_features if f in continuous_cols]
cont_indices_in_scaler = [continuous_cols.index(f) for f in final_continuous]
sub_means = scaler.mean_[cont_indices_in_scaler]
sub_scales = scaler.scale_[cont_indices_in_scaler]

# ------------------------------------------------------------
# 特征定义
# ------------------------------------------------------------
display_features = [
    'Vasopressor', 'BUN', 'HR', 'OI', 'APTT', 'RR', 'pH',
    'Bicarbonate', 'SBP', 'UO', 'QRS width', 'Chloride',
    'Platelet', 'PT'
]
actual_features = [
    'vasopressor', 'bun', 'heart_rate', 'oi', 'aptt', 'respiratory_rate',
    'ph', 'bicarbonate', 'systolic_bp', 'uo_24h', 'qrs_width_mean',
    'chloride', 'platelet_count', 'pt'
]

# 各特征的显示范围（用于输入控件）
feature_ranges = {
    'BUN': (0.0, 200.0, 'mg/dL', 0.01),
    'HR': (1, 250, 'bpm', 1),          
    'OI': (0.0, 800.0, '', 0.01),
    'APTT': (0.0, 200.0, 'sec', 0.01),
    'RR': (1, 80, 'breaths/min', 1),   
    'pH': (0.0, 7.8, '', 0.01),
    'Bicarbonate': (0.0, 50.0, 'mmol/L', 0.01),
    'SBP': (1, 300, 'mmHg', 1),        
    'UO': (1, 5000, 'mL/24h', 1),      
    'QRS width': (0.0, 2000.0, 'ms', 0.01),
    'Chloride': (0.0, 150.0, 'mmol/L', 0.01),
    'Platelet': (1, 1000, 'x10⁹/L', 1), 
    'PT': (0.0, 100.0, 'sec', 0.01)
}

# ------------------------------------------------------------
# 页面标题
# ------------------------------------------------------------
st.title("AMI Weaning Failure Predictor")
st.markdown("---")

# ------------------------------------------------------------
# 输入表单
# ------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Information")
    with st.form("input_form"):
        input_col1, input_col2 = st.columns(2)
        input_values = {}

        other_features = [f for f in display_features if f != 'Vasopressor']
        half = len(other_features) // 2

        with input_col1:
            
            vaso_choice = st.selectbox(
                "Vasopressor (Yes/No)",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="vasopressor_select"
            )
            input_values['Vasopressor'] = float(vaso_choice)

            for feat in other_features[:half]:
                min_val, max_val, unit, step = feature_ranges[feat]
                is_int = (step == 1 and min_val >= 1)
                label = f"{feat} ({unit})" if unit else feat
                if is_int:
                    label += " (integer)"
                else:
                    label += " (positive, 2 decimals)"
                input_values[feat] = st.number_input(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=float(step),
                    key=feat
                )

        with input_col2:
            for feat in other_features[half:]:
                min_val, max_val, unit, step = feature_ranges[feat]
                is_int = (step == 1 and min_val >= 1)
                label = f"{feat} ({unit})" if unit else feat
                if is_int:
                    label += " (integer)"
                else:
                    label += " (positive, 2 decimals)"
                input_values[feat] = st.number_input(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=float(step),
                    key=feat
                )

        submitted = st.form_submit_button("Predict")

        if submitted:
            
            raw_dict = {actual: float(input_values[display]) 
                        for display, actual in zip(display_features, actual_features)}

            X_list = []
            for feat in final_features:
                val = raw_dict[feat]
                if feat in final_continuous:
                    idx = final_continuous.index(feat)
                    scaled_val = (val - sub_means[idx]) / sub_scales[idx]
                    X_list.append(scaled_val)
                else:
                    X_list.append(val)

            X_final = np.array(X_list).reshape(1, -1)

            try:
                proba = model.predict_proba(X_final)[0][1] * 100
                st.success("Prediction completed!")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                proba = None

# ------------------------------------------------------------
# 结果显示区域
# ------------------------------------------------------------
with col2:
    st.header("Prediction Result")
    if submitted and 'proba' in locals() and proba is not None:
        st.metric(
            label="Probability of Weaning Failure",
            value=f"{proba:.1f}%"
        )

# ------------------------------------------------------------
# 说明信息
# ------------------------------------------------------------
st.markdown("---")
st.header("About This Calculator")

about_text = f"""
This tool predicts the **risk of weaning failure** for patients with **Acute Myocardial Infarction (AMI)** up to **6 hours in advance**.

The underlying model is a **Random Forest** classifier, trained with **MICE imputation** and feature selection (Lasso + Boruta).  
It uses **{len(final_features)} selected features** and was externally validated.

---

**Definition of weaning failure (WF)**:  
Reintubation, need for non‑invasive ventilation, or death within 48 hours following extubation.

**Input features**

| Clinical Parameter           | Notes |
|------------------------------|-------|
| Vasopressor                  | Binary (Yes/No): whether any of norepinephrine, dopamine, or dobutamine was being administered **at the 6‑hour time point prior to weaning**. |
| BUN (Blood Urea Nitrogen)    | Last value before the 6‑hour time point prior to weaning|
| HR (Heart Rate)              | Last value before the 6‑hour time point prior to weaning|
| OI (Oxygenation Index)       | Last value before the 6‑hour time point prior to weaning|
| APTT                         | Last value before the 6‑hour time point prior to weaning|
| RR (Respiratory Rate)        | Last value before the 6‑hour time point prior to weaning|
| pH                           | Last value before the 6‑hour time point prior to weaning|
| Bicarbonate                  | Last value before the 6‑hour time point prior to weaning|
| SBP (Systolic Blood Pressure)| Last value before the 6‑hour time point prior to weaning|
| UO (Urine Output)            | from 30 hours to 6 hours prior to weaning (mL/24 h)|
| QRS width                    | Last value before the 6‑hour time point prior to weaning|
| Chloride                     | Last value before the 6‑hour time point prior to weaning|
| Platelet                     | Last value before the 6‑hour time point prior to weaning|
| PT                           | Last value before the 6‑hour time point prior to weaning|

---

**Interpretation**:
- The model outputs the probability (0–100%) that the patient will experience **weaning failure**.
- The optimal probability threshold determined in the original study is **38%**.

**Disclaimer**: This tool is for clinical decision support only and does not replace comprehensive physician assessment.
"""

st.markdown(about_text)
st.caption("© 2026 – AMI Weaning Failure Predictor | Original Random Forest model (threshold 0.38)")
