
import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_all():
    with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    models = {}
    for target, target_models in metadata["models"].items():
        models[target] = {}
        for name, relative_path in target_models.items():
            models[target][name] = joblib.load(BASE_DIR / relative_path)

    return metadata, models


def linear_formula_f1(B, L, H, E, rho):
    return (
        -1.234
        + 0.233 * B
        + 0.531 * L
        - 0.738 * H
        + 3.144 * math.sqrt(E / rho)
    )


def symbolic_formula_f1(B, L, H, E, rho):
    return (
        math.sqrt(E / rho)
        * (1.273 * L / H - 0.395)
        / ((L / (B + 5.428)) ** 4 + 0.278)
    )


metadata, models = load_all()
roles = metadata["roles"]
ranges = metadata["ranges"]

st.set_page_config(
    page_title="Masonry Frequency Predictor",
    page_icon="🏛️",
    layout="wide",
)

st.title("Fundamental Frequency Prediction of Masonry Buildings")
st.write(
    "Enter the geometric and material properties to obtain predictions "
    "from the trained machine-learning models and the explicit equations."
)

st.info(
    "Use this application only within the parameter ranges adopted in the study."
)

left, right = st.columns(2)

with left:
    B = st.number_input(
        "Plan width, B (m)",
        min_value=float(ranges[roles["B"]]["min"]),
        max_value=float(ranges[roles["B"]]["max"]),
        value=float(ranges[roles["B"]]["default"]),
        step=0.10,
    )

    L = st.number_input(
        "Plan length, L (m)",
        min_value=float(ranges[roles["L"]]["min"]),
        max_value=float(ranges[roles["L"]]["max"]),
        value=float(ranges[roles["L"]]["default"]),
        step=0.10,
    )

    H = st.number_input(
        "Structural height, H (m)",
        min_value=float(ranges[roles["H"]]["min"]),
        max_value=float(ranges[roles["H"]]["max"]),
        value=float(ranges[roles["H"]]["default"]),
        step=0.10,
    )

with right:
    E = st.number_input(
        "Elastic modulus, E (MPa)",
        min_value=float(ranges[roles["E"]]["min"]),
        max_value=float(ranges[roles["E"]]["max"]),
        value=float(ranges[roles["E"]]["default"]),
        step=50.0,
    )

    rho = st.number_input(
        "Wall density, ρ (kg/m³)",
        min_value=float(ranges[roles["rho"]]["min"]),
        max_value=float(ranges[roles["rho"]]["max"]),
        value=float(ranges[roles["rho"]]["default"]),
        step=50.0,
    )

input_df = pd.DataFrame(
    [{
        roles["B"]: B,
        roles["L"]: L,
        roles["H"]: H,
        roles["E"]: E,
        roles["rho"]: rho,
    }],
    columns=metadata["feature_columns"],
)

if st.button("Predict", type="primary"):
    rows = []

    for target, target_models in models.items():
        for model_name, model in target_models.items():
            prediction = float(model.predict(input_df)[0])
            rows.append({
                "Target": target,
                "Method": model_name,
                "Predicted frequency (Hz)": prediction,
            })

    rows.extend([
        {
            "Target": "f1 (Hz)",
            "Method": "Linear Regression Equation",
            "Predicted frequency (Hz)": linear_formula_f1(
                B, L, H, E, rho
            ),
        },
        {
            "Target": "f1 (Hz)",
            "Method": "Symbolic Regression Equation",
            "Predicted frequency (Hz)": symbolic_formula_f1(
                B, L, H, E, rho
            ),
        },
    ])

    results = pd.DataFrame(rows)
    results["Predicted frequency (Hz)"] = (
        results["Predicted frequency (Hz)"].round(4)
    )

    st.subheader("Prediction results")
    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    f1_results = results[results["Target"] == "f1 (Hz)"]
    if not f1_results.empty:
        st.subheader("f1 model comparison")
        st.bar_chart(
            f1_results.set_index("Method")["Predicted frequency (Hz)"]
        )

with st.expander("Show equations"):
    st.markdown("**Linear regression**")
    st.latex(
        r"""
        f_1=-1.234+0.233B+0.531L-0.738H+
        3.144\sqrt{\frac{E}{\rho}}
        """
    )

    st.markdown("**Symbolic regression**")
    st.latex(
        r"""
        f_1=
        \sqrt{\frac{E}{\rho}}
        \frac{1.273L/H-0.395}
        {\left(L/(B+5.428)\right)^4+0.278}
        """
    )

    st.caption(
        "B, L, and H are in m; E is in MPa; "
        "ρ is in kg/m³; f1 is in Hz."
    )

st.markdown("---")
st.caption(
    "Machine-learning predictions are generated using the final models "
    "fitted to the complete dataset after group-based hyperparameter selection."
)
