"""
HAB Prediction: Regressor Suite to Predict log_abundance
=========================================================
Features:  is_s2, b2 (blue), b3 (green), b4 (red), b5 (red_edge), ndci, ndwi
Target:    log_abundance  (log-transformed cyanobacteria cell concentration)

Models:
  1. AdaBoost Regressor
  2. Linear Regression
  3. Ridge / Lasso (reported together)
  4. ElasticNet Regressor

Strategy:
  - Drop rows missing ndci / red_edge (non-S2 rows) as a separate
    "S2-only" dataset; keep full dataset with ndci/ndwi as NaN-imputed (0).
  - Use 5-Fold cross-validation (random split) + hold-out test set (20%)
  - Report R², RMSE, MAE per model on both CV and test set
  - Feature importance / coefficient plots for interpretability
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

df = pd.read_csv("caml_satellite_matchup_clean.csv")

# Rename bands to match paper convention
df = df.rename(columns={
    "blue":     "b2",
    "green":    "b3",
    "red":      "b4",
    "red_edge": "b5",
    "log_abundance": "log_abun"
})

FEATURES = ["is_s2", "b2", "b3", "b4", "b5", "ndci", "ndwi"]
TARGET   = "log_abun"

# Fill missing b5/ndci with 0 for non-S2 rows (is_s2==0 rows have no red_edge/ndci)
df[FEATURES] = df[FEATURES].fillna(0)

X = df[FEATURES].values
y = df[TARGET].values

feature_names = FEATURES

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Model definitions
models = {
    "AdaBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=200, learning_rate=0.1, random_state=42
        ))
    ]),
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=0.01, max_iter=5000))
    ]),
    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000))
    ]),
}

# Cross-validation (5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

print("Running 5-Fold Cross-Validation …")
for name, pipe in models.items():
    cv = cross_validate(pipe, X_train, y_train, cv=kf,
                        scoring=scoring, return_train_score=False)
    cv_results[name] = {
        "cv_r2_mean":    cv["test_r2"].mean(),
        "cv_r2_std":     cv["test_r2"].std(),
        "cv_rmse_mean":  -cv["test_neg_root_mean_squared_error"].mean(),
        "cv_rmse_std":   cv["test_neg_root_mean_squared_error"].std(),
        "cv_mae_mean":   -cv["test_neg_mean_absolute_error"].mean(),
        "cv_mae_std":    cv["test_neg_mean_absolute_error"].std(),
    }
    print(f"  {name:20s}  R²={cv_results[name]['cv_r2_mean']:.4f} ± {cv_results[name]['cv_r2_std']:.4f}"
          f"  RMSE={cv_results[name]['cv_rmse_mean']:.4f}")

# Fit on full train, evaluate on test
print("\nTest-Set Performance:")
test_results = {}
fitted_models = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    test_results[name] = {"r2": r2, "rmse": rmse, "mae": mae}
    fitted_models[name] = pipe
    print(f"  {name:20s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# Summary DataFrame
rows = []
for name in models:
    r = cv_results[name]
    t = test_results[name]
    rows.append({
        "Model": name,
        "CV R²": f"{r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}",
        "CV RMSE": f"{r['cv_rmse_mean']:.4f} ± {r['cv_rmse_std']:.4f}",
        "CV MAE": f"{r['cv_mae_mean']:.4f} ± {r['cv_mae_std']:.4f}",
        "Test R²": f"{t['r2']:.4f}",
        "Test RMSE": f"{t['rmse']:.4f}",
        "Test MAE": f"{t['mae']:.4f}",
    })
summary_df = pd.DataFrame(rows)
print("\n", summary_df.to_string(index=False))

# Extract coefficients / importances
coef_data = {}

# Linear models → coefficients
for name in ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]:
    pipe = fitted_models[name]
    scaler = pipe.named_steps["scaler"]
    coef   = pipe.named_steps["model"].coef_
    # Un-scale for interpretability (effect per unit of original feature)
    raw_coef = coef / scaler.scale_
    coef_data[name] = raw_coef

# AdaBoost → feature importances
ada_pipe = fitted_models["AdaBoost"]
importances = ada_pipe.named_steps["model"].feature_importances_
coef_data["AdaBoost (importance)"] = importances

# Plotting
PALETTE = {
    "AdaBoost":          "#4C72B0",
    "Linear Regression": "#DD8452",
    "Ridge":             "#55A868",
    "Lasso":             "#C44E52",
    "ElasticNet":        "#8172B3",
}
COLORS = list(PALETTE.values())

fig = plt.figure(figsize=(20, 22), facecolor="#F7F9FC")
fig.suptitle("Log-Abundance Regressor Comparison", fontsize=18,
             fontweight="bold", y=0.98, color="#1a1a2e")

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.95, bottom=0.04, left=0.07, right=0.97)

model_names = list(models.keys())

# Bar chart: Test R²
ax1 = fig.add_subplot(gs[0, 0])
r2_vals  = [test_results[n]["r2"]  for n in model_names]
rmse_vals= [test_results[n]["rmse"] for n in model_names]

bars = ax1.barh(model_names, r2_vals, color=COLORS, edgecolor="white",
                height=0.55, zorder=3)
for bar, val in zip(bars, r2_vals):
    ax1.text(max(val + 0.01, 0.01), bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=9, color="#333")
ax1.set_xlim(0, 1.08)
ax1.set_xlabel("R²")
ax1.set_title("Test R²  (higher = better)", fontsize=11, fontweight="bold")
ax1.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
ax1.set_facecolor("#FAFAFA"); ax1.grid(axis="x", alpha=0.3, zorder=0)

# Bar chart: Test RMSE
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(model_names, rmse_vals, color=COLORS, edgecolor="white",
                 height=0.55, zorder=3)
for bar, val in zip(bars2, rmse_vals):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=9, color="#333")
ax2.set_xlabel("RMSE (log units)")
ax2.set_title("Test RMSE  (lower = better)", fontsize=11, fontweight="bold")
ax2.set_facecolor("#FAFAFA"); ax2.grid(axis="x", alpha=0.3, zorder=0)

# CV R² fold performance (box-like using mean ± std)
ax3 = fig.add_subplot(gs[1, 0])
cv_means = [cv_results[n]["cv_r2_mean"] for n in model_names]
cv_stds  = [cv_results[n]["cv_r2_std"]  for n in model_names]
x_pos = np.arange(len(model_names))
ax3.bar(x_pos, cv_means, yerr=cv_stds, color=COLORS, edgecolor="white",
        capsize=5, width=0.55, zorder=3)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
ax3.set_ylabel("R²")
ax3.set_title("5-Fold CV R²  (mean ± std)", fontsize=11, fontweight="bold")
ax3.set_ylim(0, 1.0)
ax3.set_facecolor("#FAFAFA"); ax3.grid(axis="y", alpha=0.3, zorder=0)

# Feature importances – AdaBoost
ax5 = fig.add_subplot(gs[2, 0])
imp = coef_data["AdaBoost (importance)"]
order = np.argsort(imp)
ax5.barh(np.array(feature_names)[order], imp[order],
         color="#4C72B0", edgecolor="white", height=0.55, zorder=3)
ax5.set_xlabel("Feature Importance")
ax5.set_title("AdaBoost – Feature Importances", fontsize=11, fontweight="bold")
ax5.set_facecolor("#FAFAFA"); ax5.grid(axis="x", alpha=0.3, zorder=0)

# Coefficient heatmap – linear models
ax6 = fig.add_subplot(gs[2, 1])
lin_names = ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]
coef_matrix = np.array([coef_data[n] for n in lin_names])

# normalise each row by its max abs for visual comparison
norm_matrix = coef_matrix / (np.abs(coef_matrix).max(axis=1, keepdims=True) + 1e-12)

im = ax6.imshow(norm_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax6.set_xticks(range(len(feature_names)))
ax6.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=9)
ax6.set_yticks(range(len(lin_names)))
ax6.set_yticklabels(lin_names, fontsize=9)
ax6.set_title("Linear Model Coefficients\n(row-normalised: red=+, blue=–)",
              fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax6, fraction=0.04, pad=0.02)

for i in range(len(lin_names)):
    for j in range(len(feature_names)):
        ax6.text(j, i, f"{coef_matrix[i, j]:.2f}",
                 ha="center", va="center", fontsize=6.5,
                 color="white" if abs(norm_matrix[i, j]) > 0.6 else "black")

plt.savefig("/Users/brianna/Documents/Data Science/MachineLearningClimateChange/FinalProject/log_abun_regressors.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nFigure saved → log_abun_regressors.png")

# Save summary CSV
summary_df.to_csv("/Users/brianna/Documents/Data Science/MachineLearningClimateChange/FinalProject/regressor_performance_summary.csv", index=False)
print("Summary CSV saved → regressor_performance_summary.csv")

plt.show()
