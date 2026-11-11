"""
Script 03: Machine Learning Model Training, Comparison, and Selection (Deliverables 4 & 5)
Evaluates 4 candidate models x 4 cases:
1. Multi-Layer Perceptron (MLP)
2. XGBoost Regressor
3. LightGBM Regressor
4. Random Forest Regressor
Generates Deliverable 4 (Model comparison table) & Deliverable 5 (Chosen model parity plots & saved models)
"""
import sys
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, SAVED_MODELS_DIR, RESULTS_DIR, RANDOM_SEED, CASES
from src.model_wrapper import LogTransformedModel, FLOOR_VAL

sns.set_theme(style="whitegrid", font_scale=1.1)

FEATURE_COLS = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb', 'vth0', 'toxe', 'ndep', 'u0', 'xj']
TARGET_COLS  = ['id', 'ig', 'is', 'ib']


def compute_metrics(y_true, y_pred):
    """Computes R2 on log scale and median & 95th% relative errors on currents."""
    y_true_abs = np.abs(y_true)
    y_true_log = np.log10(np.maximum(y_true_abs, FLOOR_VAL))
    y_pred_log = np.log10(np.maximum(np.abs(y_pred), FLOOR_VAL))

    r2 = r2_score(y_true_log, y_pred_log)
    denom = np.maximum(y_true_abs, 1e-15)
    rel_errors = np.abs(y_pred - y_true_abs) / denom * 100.0

    med_err = np.median(rel_errors)
    p95_err = np.percentile(rel_errors, 95)
    return r2, med_err, p95_err


def get_candidate_models():
    """Returns the 4 candidate model architectures."""
    return {
        "MLP (Neural Net)": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=RANDOM_SEED
        ),
        "XGBoost": MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        ),
        "LightGBM": MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }


def main():
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comp_dir = RESULTS_DIR / "model_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    comparison_records = []
    chosen_models = {}

    print("=================================================================")
    print("      TRAINING & EVALUATING 4 ML MODELS ACROSS 4 CASES           ")
    print("=================================================================")

    for case in CASES:
        csv_file = DATA_DIR / f"{case}.csv"
        print(f"\n---> Case: {case} <---")
        df = pd.read_csv(csv_file)

        X = df[FEATURE_COLS]
        y = df[TARGET_COLS]

        # 70% Train, 15% Val, 15% Test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED)

        candidates = get_candidate_models()
        best_val_err = float('inf')
        best_name = None
        best_model = None

        for name, estimator in candidates.items():
            pipeline = LogTransformedModel(estimator)
            pipeline.fit(X_train, y_train)

            # Validation evaluation
            y_val_pred = pipeline.predict(X_val)
            r2, med_err, p95_err = compute_metrics(y_val.values, y_val_pred)

            print(f"  [{name:18s}] Val R² = {r2:.4f} | Med Err = {med_err:.2f}% | 95th Err = {p95_err:.2f}%")

            comparison_records.append({
                'case': case,
                'model': name,
                'val_r2': r2,
                'val_median_error_%': med_err,
                'val_95th_percentile_error_%': p95_err
            })

            if med_err < best_val_err:
                best_val_err = med_err
                best_name = name
                best_model = pipeline

        print(f"  >>> Best model selected for {case}: {best_name}")

        # Evaluate best model on unseen Test Set
        y_test_pred = best_model.predict(X_test)
        test_r2, test_med, test_p95 = compute_metrics(y_test.values, y_test_pred)
        print(f"  >>> Test Set: R² = {test_r2:.4f} | Med Err = {test_med:.2f}% | 95th Err = {test_p95:.2f}%")

        # Save chosen model
        model_out = SAVED_MODELS_DIR / f"{case}_best_model.joblib"
        joblib.dump(best_model, model_out)
        chosen_models[case] = best_model

        # Deliverable 5: Predicted vs SPICE Parity Plot for test set (Id)
        fig, ax = plt.subplots(figsize=(6, 6))
        y_t_id = np.abs(y_test['id'].values)
        y_p_id = np.abs(y_test_pred[:, 0])

        ax.scatter(y_t_id, y_p_id, alpha=0.3, s=15, color='royalblue', label='Test Samples')
        min_v = max(1e-15, min(y_t_id.min(), y_p_id.min()))
        max_v = max(y_t_id.max(), y_p_id.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='Ideal 1:1')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("SPICE |Id| (A)")
        ax.set_ylabel("ML Predicted |Id| (A)")
        ax.set_title(f"{case} Test Parity ({best_name})\nMed Err={test_med:.2f}%, 95th%={test_p95:.2f}%")
        ax.legend()
        plt.tight_layout()
        plt.savefig(comp_dir / f"parity_test_{case}.png", dpi=300)
        plt.close()

    # Save Deliverable 4 Model Comparison Table
    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(comp_dir / "model_comparison_table.csv", index=False)
    print(f"\n[+] Saved Deliverable 4 table to {comp_dir / 'model_comparison_table.csv'}")


if __name__ == "__main__":
    main()
