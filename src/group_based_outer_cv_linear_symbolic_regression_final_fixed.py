
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pysr import PySRRegressor

warnings.filterwarnings("ignore")
plt.ioff()

# =========================================================
# USER SETTINGS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATH = BASE_DIR / "data" / "Kombinations.xlsx"
SHEET_NAME = 0

TARGET_COL = "f1 (Hz)"
GEOMETRY_COLS = ["B (m)", "L (m)", "H (m)"]

E_COL = "E (MPa)"
RHO_COL_CANDIDATES = ["Ro (kg/m^3)", "Ro (kg/m3)", "ρ (kg/m3)", "rho (kg/m3)"]

BASE_FEATURE_COLS = ["B (m)", "L (m)", "H (m)"]
TRANSFORMED_FEATURE_NAME = "sqrt_E_over_rho_m_s"

N_OUTER_SPLITS = 5
RANDOM_STATE = 42

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_XLSX = OUTPUT_DIR / "group_based_linear_symbolic_regression_results.xlsx"
SUMMARY_TXT = OUTPUT_DIR / "group_based_linear_symbolic_regression_summary.txt"

# =========================================================
# PySR SETTINGS
# =========================================================
PYSR_MODEL_KWARGS = dict(
    niterations=300,
    populations=30,
    population_size=50,
    maxsize=25,
    maxdepth=10,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    model_selection="best",
    loss="loss(prediction, target) = (prediction - target)^2",
    parsimony=1e-4,
    random_state=RANDOM_STATE,
    deterministic=True,
    parallelism="serial",
    verbosity=0,
    progress=False,
)

# =========================================================
# HELPERS
# =========================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE_Hz": mean_absolute_error(y_true, y_pred),
        "RMSE_Hz": rmse(y_true, y_pred),
    }


def find_rho_column(columns):
    for candidate in RHO_COL_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Density column was not found. Checked: "
        + ", ".join(RHO_COL_CANDIDATES)
    )


def make_group_ids(dataframe):
    parts = []
    for col in GEOMETRY_COLS:
        values = pd.to_numeric(dataframe[col], errors="coerce").round(8)
        parts.append(values.astype(str))

    groups = parts[0]
    for part in parts[1:]:
        groups = groups + "_" + part
    return groups


def make_parity_plot(y_true, y_pred, title, save_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    margin = 0.05 * (lim_max - lim_min if lim_max != lim_min else 1.0)

    metrics = evaluate_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    ax.scatter(y_true, y_pred, alpha=0.8)
    ax.plot(
        [lim_min - margin, lim_max + margin],
        [lim_min - margin, lim_max + margin],
        linestyle="--",
        linewidth=1.8
    )

    ax.text(
        0.05, 0.95,
        f"$R^2$ = {metrics['R2']:.4f}\n"
        f"MAE = {metrics['MAE_Hz']:.4f} Hz\n"
        f"RMSE = {metrics['RMSE_Hz']:.4f} Hz",
        transform=ax.transAxes,
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"Actual $f_1$ (Hz)", fontsize=15)
    ax.set_ylabel(r"Predicted $f_1$ (Hz)", fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lim_min - margin, lim_max + margin)
    ax.set_ylim(lim_min - margin, lim_max + margin)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def linear_equation_string(model, feature_names):
    terms = [f"{model.intercept_:.10g}"]
    for coef, feature in zip(model.coef_, feature_names):
        sign = "+" if coef >= 0 else "-"
        terms.append(f" {sign} {abs(coef):.10g}*{feature}")
    return "f1_Hz =" + "".join(terms)


def selected_symbolic_equation(model):
    try:
        return str(model.sympy())
    except Exception:
        return str(model.get_best()["sympy_format"])


# =========================================================
# READ AND PREPARE DATA
# =========================================================
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]

required_cols = set(BASE_FEATURE_COLS + [E_COL, TARGET_COL])
missing_required = [col for col in required_cols if col not in df.columns]
if missing_required:
    raise ValueError(
        "Missing required columns: " + ", ".join(missing_required)
    )

rho_col = find_rho_column(df.columns)

numeric_cols = list(required_cols) + [rho_col]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols).reset_index(drop=True)

if (df[E_COL] <= 0).any():
    raise ValueError("Elastic modulus values must be positive.")
if (df[rho_col] <= 0).any():
    raise ValueError("Density values must be positive.")

# E is converted from MPa to Pa, so sqrt(E/rho) has units of m/s.
df[TRANSFORMED_FEATURE_NAME] = np.sqrt(
    df[E_COL] * 1_000_000.0 / df[rho_col]
)

feature_cols = BASE_FEATURE_COLS + [TRANSFORMED_FEATURE_NAME]

# PySR accepts only alphanumeric variable names and underscores.
# Therefore, the original dataframe columns are renamed to safe aliases.
SAFE_FEATURE_NAMES = ["B", "L", "H", "sqrt_E_over_rho"]

X = df[feature_cols].copy()
X.columns = SAFE_FEATURE_NAMES
y = df[TARGET_COL].copy()

groups = make_group_ids(df)
n_groups = groups.nunique()

if n_groups < N_OUTER_SPLITS:
    raise ValueError(
        f"Only {n_groups} geometry groups are available, "
        f"but {N_OUTER_SPLITS} folds were requested."
    )

print(f"Unique geometry groups: {n_groups}")
print(f"Original features: {feature_cols}")
print(f"PySR-safe feature names: {SAFE_FEATURE_NAMES}")

# =========================================================
# GROUP-BASED OUTER CV
# =========================================================
outer_cv = GroupKFold(n_splits=N_OUTER_SPLITS)

linear_oof = np.full(len(df), np.nan)
symbolic_oof = np.full(len(df), np.nan)

fold_records = []
symbolic_fold_equations = []
split_records = []

for fold_no, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1
):
    print(f"Outer fold {fold_no}/{N_OUTER_SPLITS}")

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    overlap = set(groups_train).intersection(set(groups_test))
    if overlap:
        raise RuntimeError(
            f"Geometry-group leakage detected in fold {fold_no}: {overlap}"
        )

    split_records.append({
        "Outer_Fold": fold_no,
        "Train_Rows": len(train_idx),
        "Test_Rows": len(test_idx),
        "Train_Groups": groups_train.nunique(),
        "Test_Groups": groups_test.nunique(),
        "Group_Overlap_Count": len(overlap),
        "Test_Group_IDs": " | ".join(sorted(map(str, set(groups_test)))),
    })

    # Linear regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_oof[test_idx] = linear_pred

    linear_metrics = evaluate_metrics(y_test, linear_pred)
    fold_records.append({
        "Model": "Linear Regression",
        "Outer_Fold": fold_no,
        **linear_metrics,
    })

    # Symbolic regression
    symbolic_model = PySRRegressor(**PYSR_MODEL_KWARGS)
    symbolic_model.fit(X_train, y_train)

    symbolic_pred = np.asarray(
        symbolic_model.predict(X_test),
        dtype=float
    ).reshape(-1)

    symbolic_oof[test_idx] = symbolic_pred
    symbolic_metrics = evaluate_metrics(y_test, symbolic_pred)

    fold_records.append({
        "Model": "Symbolic Regression",
        "Outer_Fold": fold_no,
        **symbolic_metrics,
    })

    symbolic_fold_equations.append({
        "Outer_Fold": fold_no,
        "Selected_Equation": selected_symbolic_equation(symbolic_model),
        "Outer_Test_R2": symbolic_metrics["R2"],
        "Outer_Test_MAE_Hz": symbolic_metrics["MAE_Hz"],
        "Outer_Test_RMSE_Hz": symbolic_metrics["RMSE_Hz"],
    })

if np.isnan(linear_oof).any() or np.isnan(symbolic_oof).any():
    raise RuntimeError("Some outer-fold predictions are missing.")

# =========================================================
# SUMMARIES
# =========================================================
fold_df = pd.DataFrame(fold_records)

summary_records = []
for model_name, predictions in [
    ("Linear Regression", linear_oof),
    ("Symbolic Regression", symbolic_oof),
]:
    model_fold_df = fold_df[fold_df["Model"] == model_name]
    pooled_metrics = evaluate_metrics(y, predictions)

    summary_records.append({
        "Model": model_name,
        "Outer_Test_R2_Mean": model_fold_df["R2"].mean(),
        "Outer_Test_R2_SD": model_fold_df["R2"].std(ddof=1),
        "Outer_Test_MAE_Mean_Hz": model_fold_df["MAE_Hz"].mean(),
        "Outer_Test_MAE_SD_Hz": model_fold_df["MAE_Hz"].std(ddof=1),
        "Outer_Test_RMSE_Mean_Hz": model_fold_df["RMSE_Hz"].mean(),
        "Outer_Test_RMSE_SD_Hz": model_fold_df["RMSE_Hz"].std(ddof=1),
        "Overall_Grouped_OOF_R2": pooled_metrics["R2"],
        "Overall_Grouped_OOF_MAE_Hz": pooled_metrics["MAE_Hz"],
        "Overall_Grouped_OOF_RMSE_Hz": pooled_metrics["RMSE_Hz"],
    })

summary_df = pd.DataFrame(summary_records)

paper_ready_df = pd.DataFrame({
    "Model": summary_df["Model"],
    "Test R2 (mean ± SD)": [
        f"{r.Outer_Test_R2_Mean:.4f} ± {r.Outer_Test_R2_SD:.4f}"
        for r in summary_df.itertuples()
    ],
    "Test MAE (Hz, mean ± SD)": [
        f"{r.Outer_Test_MAE_Mean_Hz:.4f} ± {r.Outer_Test_MAE_SD_Hz:.4f}"
        for r in summary_df.itertuples()
    ],
    "Test RMSE (Hz, mean ± SD)": [
        f"{r.Outer_Test_RMSE_Mean_Hz:.4f} ± {r.Outer_Test_RMSE_SD_Hz:.4f}"
        for r in summary_df.itertuples()
    ],
    "Overall grouped out-of-fold R2": [
        f"{r.Overall_Grouped_OOF_R2:.4f}"
        for r in summary_df.itertuples()
    ],
})

# =========================================================
# FINAL FULL-DATA EQUATIONS
# =========================================================
final_linear = LinearRegression()
final_linear.fit(X, y)
final_linear_equation = linear_equation_string(final_linear, SAFE_FEATURE_NAMES)

final_symbolic = PySRRegressor(**PYSR_MODEL_KWARGS)
final_symbolic.fit(X, y)
final_symbolic_equation = selected_symbolic_equation(final_symbolic)

final_equations_df = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "Final_Full_Data_Equation": final_linear_equation,
        "Purpose": "Final practical equation after grouped outer-CV evaluation",
    },
    {
        "Model": "Symbolic Regression",
        "Final_Full_Data_Equation": final_symbolic_equation,
        "Purpose": "Final practical equation after grouped outer-CV evaluation",
    },
])

# =========================================================
# SAVE OUTPUTS
# =========================================================
predictions_df = pd.DataFrame({
    "Actual_f1_Hz": y,
    "Geometry_Group": groups,
    "Pred_Linear_Grouped_OOF": linear_oof,
    "Residual_Linear_Grouped_OOF": y - linear_oof,
    "Pred_Symbolic_Grouped_OOF": symbolic_oof,
    "Residual_Symbolic_Grouped_OOF": y - symbolic_oof,
})

make_parity_plot(
    y,
    linear_oof,
    "Linear Regression - Grouped OOF",
    PLOTS_DIR / "linear_grouped_oof_parity.png"
)

make_parity_plot(
    y,
    symbolic_oof,
    "Symbolic Regression - Grouped OOF",
    PLOTS_DIR / "symbolic_grouped_oof_parity.png"
)

with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
    paper_ready_df.to_excel(writer, sheet_name="Paper_Ready_Table", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    fold_df.to_excel(writer, sheet_name="Outer_Fold_Metrics", index=False)
    pd.DataFrame(symbolic_fold_equations).to_excel(
        writer, sheet_name="Symbolic_Fold_Equations", index=False
    )
    final_equations_df.to_excel(
        writer, sheet_name="Final_Equations", index=False
    )
    predictions_df.to_excel(
        writer, sheet_name="Grouped_OOF_Predictions", index=False
    )
    pd.DataFrame(split_records).to_excel(
        writer, sheet_name="Group_Splits", index=False
    )
    df.to_excel(writer, sheet_name="Prepared_Data", index=False)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("GROUP-BASED OUTER CROSS-VALIDATION\n")
    f.write("=" * 70 + "\n")
    f.write(f"Unique geometry groups: {n_groups}\n")
    f.write(
        "Predictors: B, L, H, and sqrt(E/rho), "
        "with E converted from MPa to Pa. "
        "PySR-safe variable names: B, L, H, sqrt_E_over_rho.\n\n"
    )
    f.write("PAPER-READY PERFORMANCE TABLE\n")
    f.write(paper_ready_df.to_string(index=False))
    f.write("\n\n")
    f.write("FINAL FULL-DATA LINEAR EQUATION\n")
    f.write(final_linear_equation + "\n\n")
    f.write("FINAL FULL-DATA SYMBOLIC EQUATION\n")
    f.write(final_symbolic_equation + "\n\n")
    f.write(
        "Interpretation:\n"
        "- Performance metrics are based on unseen geometry groups in the outer folds.\n"
        "- A separate symbolic equation was discovered in each outer training fold.\n"
        "- Final equations were derived using the complete dataset only after "
        "cross-validated performance assessment.\n"
    )

print("\nPAPER-READY GROUP-BASED REGRESSION TABLE")
print(paper_ready_df.to_string(index=False))

print("\nFinal linear equation:")
print(final_linear_equation)

print("\nFinal symbolic equation:")
print(final_symbolic_equation)

print(f"\nExcel results saved to: {RESULTS_XLSX}")
print(f"Text summary saved to: {SUMMARY_TXT}")
print(f"Plots saved to: {PLOTS_DIR}")
