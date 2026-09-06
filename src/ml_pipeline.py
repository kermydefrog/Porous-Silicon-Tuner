"""
Machine learning pipeline to predict porosity (%) and thickness (nm)
from porous silicon etching parameters.

Features : HF ratio, current density, etch time, resistivity,
           sac current, sac time
Targets  : porosity (%) and thickness (nm) — each predicted by its own pipeline

Models   : Random Forest, Gradient Boosting, Ridge regression (baseline)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
TSV_PATH = DATA_DIR / "AllData2025-2026.tsv"


# ---------------------------------------------------------------------------
# 1. Load & clean data
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0)
    df.columns = ["HF", "current_density", "etch_time",
                  "resistivity", "porosity", "thickness", "sac_current", "sac_time"]

    df["HF"] = df["HF"].str.strip()

    # Parse porosity: "60.7%" → 60.7
    df["porosity"] = (
        df["porosity"].astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    for col in ["current_density", "etch_time", "resistivity",
                "thickness", "sac_current", "sac_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["current_density", "etch_time", "HF"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Schema
# ---------------------------------------------------------------------------

FEATURES     = ["HF", "current_density", "etch_time",
                "resistivity", "sac_current", "sac_time"]
TARGETS      = ["porosity", "thickness"]
CAT_FEATURES = ["HF"]
NUM_FEATURES = ["current_density", "etch_time", "resistivity",
                "sac_current", "sac_time"]

# ---------------------------------------------------------------------------
# HF feature engineering
# ---------------------------------------------------------------------------
# "3:2:1" = HF:acetic acid:ethanol  →  φ_HF = 3/6 = 0.5, is_acetic = 1
# "1:1"   = HF:ethanol              →  φ_HF = 1/2 = 0.5, is_acetic = 0
# "3:1"   = HF:ethanol              →  φ_HF = 3/4 = 0.75, is_acetic = 0
# "1:3"   = HF:ethanol              →  φ_HF = 1/4 = 0.25, is_acetic = 0

def hf_to_fraction(X: pd.DataFrame) -> np.ndarray:
    """Scalar HF volume fraction (kept for backward compat)."""
    def parse(s):
        parts = [float(p) for p in str(s).split(":")]
        return parts[0] / sum(parts)
    return X["HF"].apply(parse).values.reshape(-1, 1)


def hf_to_features(X: pd.DataFrame) -> np.ndarray:
    """Return [φ_HF, is_acetic] for each row.

    φ_HF    — HF volume fraction (first part / sum of all parts)
    is_acetic — 1 if electrolyte contains acetic acid (3-component ratio), else 0
    """
    phi, is_acetic = [], []
    for s in X["HF"]:
        parts = [float(p) for p in str(s).split(":")]
        phi.append(parts[0] / sum(parts))
        is_acetic.append(1.0 if len(parts) == 3 else 0.0)
    return np.column_stack([phi, is_acetic])


def add_interactions(X: np.ndarray) -> np.ndarray:
    """Append φ_HF×J, φ_HF×t_etch, J×t_etch to the feature matrix.

    Expected column order after ColumnTransformer:
      0: φ_HF  1: is_acetic  2: current_density  3: etch_time
      4: resistivity  5: sac_current  6: sac_time
    """
    phi = X[:, 0:1]
    J   = X[:, 2:3]
    t   = X[:, 3:4]
    return np.hstack([X, phi * J, phi * t, J * t])


def make_preprocessor() -> ColumnTransformer:
    """OHE for tree models (unchanged)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])


def make_ridge_preprocessor() -> Pipeline:
    """φ_HF + is_acetic + numeric features + interaction terms, then StandardScaler."""
    ct = ColumnTransformer([
        ("hf",  FunctionTransformer(hf_to_features, validate=False), ["HF"]),
        ("num", "passthrough", NUM_FEATURES),
    ])
    return Pipeline([
        ("encode",       ct),
        ("interactions", FunctionTransformer(add_interactions, validate=False)),
        ("scale",        StandardScaler()),
    ])


# Features where log transform helps (wide, positive, right-skewed ranges)
LOG_FEATURES = ["current_density", "etch_time"]
LIN_FEATURES = ["resistivity", "sac_current", "sac_time"]


def make_mlp_preprocessor(log_transform: bool = False) -> ColumnTransformer:
    """φ_HF + is_acetic + StandardScaler; optionally log1p on skewed features."""
    if log_transform:
        return ColumnTransformer([
            ("hf",  FunctionTransformer(hf_to_features, validate=False), ["HF"]),
            ("log", Pipeline([
                ("log1p", FunctionTransformer(np.log1p, validate=True)),
                ("scale", StandardScaler()),
            ]), LOG_FEATURES),
            ("lin", StandardScaler(), LIN_FEATURES),
        ])
    return ColumnTransformer([
        ("hf",  FunctionTransformer(hf_to_features, validate=False), ["HF"]),
        ("num", StandardScaler(), NUM_FEATURES),
    ])


# ---------------------------------------------------------------------------
# 3. Build single-target pipelines for one target
# ---------------------------------------------------------------------------

def build_pipelines_for(target: str) -> dict:
    """Return three named sklearn Pipelines that each predict `target`."""
    return {
        "Ridge": Pipeline([
            ("pre", make_ridge_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("pre", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=150, max_features="sqrt",
                                            min_samples_leaf=5,
                                            random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", make_preprocessor()),
            ("model", GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                learning_rate=0.05, subsample=0.8,
                                                random_state=42)),
        ]),
        # MLP: numeric HF fraction, smaller network, strong L2 (alpha),
        # target scaled to zero-mean/unit-variance for stable convergence
        "MLP": Pipeline([
            ("pre", make_mlp_preprocessor(log_transform=(target == "porosity"))),
            ("model", TransformedTargetRegressor(
                regressor=MLPRegressor(hidden_layer_sizes=(64, 32),
                                       activation="relu",
                                       solver="adam" if target == "porosity" else "lbfgs",
                                       alpha=0.1,
                                       max_iter=2000,
                                       **({"early_stopping": True,
                                           "validation_fraction": 0.15,
                                           "n_iter_no_change": 30}
                                          if target == "porosity" else {}),
                                       random_state=42),
                transformer=StandardScaler(),
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# 4. Cross-validate a set of pipelines for one target
# ---------------------------------------------------------------------------

def cross_validate_pipelines(pipelines: dict, X: pd.DataFrame,
                              y: pd.Series, cv: int = 5) -> pd.DataFrame:
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    records = []
    for name, pipe in pipelines.items():
        r2s, maes = [], []
        for tr_idx, val_idx in kf.split(X):
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            y_pred = pipe.predict(X.iloc[val_idx])
            r2s.append(r2_score(y.iloc[val_idx], y_pred))
            maes.append(mean_absolute_error(y.iloc[val_idx], y_pred))
        records.append({
            "model":    name,
            "R2_mean":  np.mean(r2s),
            "R2_std":   np.std(r2s),
            "MAE_mean": np.mean(maes),
            "MAE_std":  np.std(maes),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Train final model on train split, evaluate on hold-out
# ---------------------------------------------------------------------------

def train_and_evaluate(pipelines: dict, X: pd.DataFrame, y: pd.Series,
                       test_frac: float = 0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=42
    )
    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        results[name] = {
            "pipe":   pipe,
            "y_te":   y_te,
            "y_pred": y_pred,
            "R2":     r2_score(y_te, y_pred),
            "MAE":    mean_absolute_error(y_te, y_pred),
            "RMSE":   root_mean_squared_error(y_te, y_pred),
        }
    return results


# ---------------------------------------------------------------------------
# 6. Feature importance from a fitted Random Forest pipeline
# ---------------------------------------------------------------------------

def feature_importance_series(pipe: Pipeline) -> pd.Series:
    ohe_names = (pipe.named_steps["pre"]
                 .named_transformers_["cat"]
                 .get_feature_names_out(CAT_FEATURES).tolist())
    feat_names = ohe_names + NUM_FEATURES
    imp = pipe.named_steps["model"].feature_importances_
    return pd.Series(imp, index=feat_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "Ridge (baseline)": "#888888",
    "Random Forest":    "#2196F3",
    "Gradient Boosting":"#FF9800",
    "MLP":              "#4CAF50",
}
MODEL_MARKERS = {
    "Ridge (baseline)": "s",
    "Random Forest":    "o",
    "Gradient Boosting":"^",
    "MLP":              "D",
}


def plot_actual_vs_predicted(porosity_results: dict, thickness_results: dict,
                              save_dir: Path):
    """4×2 grid: one row per model, columns = porosity / thickness."""
    models = list(porosity_results.keys())
    fig, axes = plt.subplots(len(models), 2,
                              figsize=(11, 4 * len(models)),
                              squeeze=False)

    for row, model in enumerate(models):
        for col, (target, res_dict) in enumerate([("porosity (%)", porosity_results),
                                                   ("thickness (nm)", thickness_results)]):
            ax = axes[row][col]
            res = res_dict[model]
            color  = MODEL_COLORS[model]
            marker = MODEL_MARKERS[model]

            ax.scatter(res["y_te"], res["y_pred"],
                       color=color, marker=marker,
                       alpha=0.75, edgecolors="k", linewidths=0.4, s=45)

            lo = min(res["y_te"].min(), res["y_pred"].min())
            hi = max(res["y_te"].max(), res["y_pred"].max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="ideal")

            ax.set_xlabel(f"Actual {target}", fontsize=10)
            ax.set_ylabel(f"Predicted {target}", fontsize=10)
            ax.set_title(
                f"{model} — {target}\n"
                f"R²={res['R2']:.3f}  MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}",
                fontsize=10,
            )
            ax.grid(alpha=0.3)

    fig.suptitle("Predicted vs Actual — all models (hold-out test set)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = save_dir / "actual_vs_predicted.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_feature_importance(por_imp: pd.Series, thk_imp: pd.Series,
                             save_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, imp) in zip(axes, [("porosity", por_imp),
                                        ("thickness", thk_imp)]):
        col = imp.sort_values(ascending=True)
        ax.barh(col.index, col.values)
        ax.set_title(f"Feature importance — {label}")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = save_dir / "feature_importance.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# 8. Prediction helpers
# ---------------------------------------------------------------------------

def _input_row(hf, current_density, etch_time, resistivity, sac_current, sac_time):
    return pd.DataFrame([{
        "HF": hf,
        "current_density": current_density,
        "etch_time": etch_time,
        "resistivity": resistivity,
        "sac_current": sac_current,
        "sac_time": sac_time,
    }])


def predict_porosity(pipe: Pipeline,
                     hf: str,
                     current_density: float,
                     etch_time: float,
                     resistivity: float = 1.0,
                     sac_current: float = 200.0,
                     sac_time: float = 30.0) -> float:
    """Return predicted porosity (%) for the given etching parameters."""
    row = _input_row(hf, current_density, etch_time, resistivity, sac_current, sac_time)
    return round(float(pipe.predict(row)[0]), 2)


def predict_thickness(pipe: Pipeline,
                      hf: str,
                      current_density: float,
                      etch_time: float,
                      resistivity: float = 1.0,
                      sac_current: float = 200.0,
                      sac_time: float = 30.0) -> float:
    """Return predicted thickness (nm) for the given etching parameters."""
    row = _input_row(hf, current_density, etch_time, resistivity, sac_current, sac_time)
    return round(float(pipe.predict(row)[0]), 1)


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Porous Silicon — ML Pipeline (per-target)")
    print("=" * 65)

    df = load_data(TSV_PATH)
    print(f"\nLoaded {len(df)} total samples.\n")

    X = df[FEATURES]

    for target in TARGETS:
        y = df[target].dropna()
        X_t = X.loc[y.index]

        print(f"\n{'='*65}")
        print(f"TARGET: {target}  ({len(y)} samples with valid labels)")
        print(f"{'='*65}")

        pipes = build_pipelines_for(target)

        # Cross-validation
        print("\n  5-Fold Cross-Validation:")
        cv_df = cross_validate_pipelines(pipes, X_t, y)
        print(cv_df.to_string(index=False))

        # Hold-out evaluation
        print("\n  Hold-out Test Set (20%):")
        results = train_and_evaluate(pipes, X_t, y)
        for name, res in results.items():
            print(f"    {name:20s}  R²={res['R2']:.3f}  "
                  f"MAE={res['MAE']:.2f}  RMSE={res['RMSE']:.2f}")

        # Feature importance
        rf_imp = feature_importance_series(results["Random Forest"]["pipe"])
        print(f"\n  Feature Importance (Random Forest):")
        print(rf_imp.round(4).to_string())

        # Store for plots
        if target == "porosity":
            porosity_results = results
            por_imp = rf_imp
        else:
            thickness_results = results
            thk_imp = rf_imp

    # Plots
    save_dir = DATA_DIR
    plot_actual_vs_predicted(porosity_results, thickness_results, save_dir)
    plot_feature_importance(por_imp, thk_imp, save_dir)

    # Example predictions using the best model (Random Forest)
    print("\n--- Example Predictions (Random Forest) ---")
    por_pipe = porosity_results["Random Forest"]["pipe"]
    thk_pipe = thickness_results["Random Forest"]["pipe"]

    cases = [
        ("3:1", 100, 180),
        ("1:1",  75, 120),
        ("3:1", 200, 120),
    ]
    for hf, j, t in cases:
        p = predict_porosity(por_pipe, hf=hf, current_density=j, etch_time=t)
        h = predict_thickness(thk_pipe, hf=hf, current_density=j, etch_time=t)
        print(f"  {hf} HF, {j:>3} mA/cm², {t:>3} s  →  porosity={p}%  thickness={h} nm")


    # Example prediction using trained models
    hf = "3:1"
    j = 100
    model = "Random Forest"
    por_pipe = porosity_results[model]["pipe"]
    thk_pipe = thickness_results[model]["pipe"]
    for t in range(10,410,10):
      p = predict_porosity(por_pipe, hf=hf, current_density=j, etch_time=t)
      h = predict_thickness(thk_pipe, hf=hf, current_density=j, etch_time=t)
      print(f"  {hf} HF, {j:>3} mA/cm², {t:>3} s  →  porosity={p}%  thickness={h} nm")

    t = 120
    for j in range(10,410,10):
      p = predict_porosity(por_pipe, hf=hf, current_density=j, etch_time=t)
      h = predict_thickness(thk_pipe, hf=hf, current_density=j, etch_time=t)
      print(f"  {hf} HF, {j:>3} mA/cm², {t:>3} s  →  porosity={p}%  thickness={h} nm")
    print("\nDone.")

    return porosity_results, thickness_results

if __name__ == "__main__":
    main()
