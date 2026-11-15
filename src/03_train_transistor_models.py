"""

Script 03: Machine Learning Model Training, Comparison, and Selection (Deliverables 4 & 5)



For each of the 4 cases (NMOS_ON/OFF, PMOS_ON/OFF) it trains 4 model TYPES (default:

MLP, XGBoost, LightGBM, RandomForest). Every model type is a SignedCurrentModel: one

regressor per terminal (id, ig, is, ib) on standardised log10|I| plus a sign model.



Metrics (computed per terminal, on SIGNED currents, so a wrong sign counts as an error):

  * R2           : R2 of log10|I|

  * median / p95 : relative error |pred - true| / |true| in %, over samples with |true| >= EVAL_FLOOR

                   (below that the SPICE value is numerical noise at the gmin/abstol level)

The comparison table reports the mean of the per-terminal values; the full per-terminal

detail goes to model_comparison_per_terminal.csv.



Selection: score = mean_t(val median err) + 0.1 * mean_t(val p95 err), lower is better.

Outputs: model_comparison_table.csv, model_comparison_per_terminal.csv,

         chosen_models_test_metrics.csv, parity_test_<case>.png (all 4 terminals), saved models.

"""

import sys

import time

from pathlib import Path



import joblib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



from config import DATA_DIR, SAVED_MODELS_DIR, RESULTS_DIR, RANDOM_SEED, CASES

from src.model_wrapper import (FEATURE_COLS, TARGET_COLS, FLOOR_VAL, EVAL_FLOOR,

                               SignedCurrentModel, available_model_types)





def terminal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:

    """y_true, y_pred: (n, 4) signed currents. Returns one row per terminal."""

    rows = []

    for j, t in enumerate(TARGET_COLS):

        yt, yp = y_true[:, j], y_pred[:, j]

        mask = np.abs(yt) >= EVAL_FLOOR

        lt = np.log10(np.maximum(np.abs(yt), FLOOR_VAL))[mask]

        lp = np.log10(np.maximum(np.abs(yp), FLOOR_VAL))[mask]

        rel = np.abs(yp[mask] - yt[mask]) / np.abs(yt[mask]) * 100.0

        rows.append({

            'terminal': t,

            'r2_log': r2_score(lt, lp),

            'median_err_%': np.median(rel),

            'p95_err_%': np.percentile(rel, 95),

            'sign_acc_%': 100.0 * np.mean(np.sign(yt[mask]) == np.sign(yp[mask])),

            'frac_evaluated': mask.mean(),

        })

    return pd.DataFrame(rows)





def pooled(m: pd.DataFrame):

    return m['r2_log'].mean(), m['median_err_%'].mean(), m['p95_err_%'].mean()





def selection_score(m: pd.DataFrame) -> float:

    return m['median_err_%'].mean() + 0.1 * m['p95_err_%'].mean()





def parity_plot(case, model_name, y_true, y_pred, test_m, out_path):

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for j, (ax, t) in enumerate(zip(axes.ravel(), TARGET_COLS)):

        yt = np.maximum(np.abs(y_true[:, j]), FLOOR_VAL)

        yp = np.maximum(np.abs(y_pred[:, j]), FLOOR_VAL)

        ax.scatter(yt, yp, s=8, alpha=0.3, color='royalblue')

        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())

        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2)

        ax.set_xscale('log'); ax.set_yscale('log')

        r = test_m.iloc[j]

        ax.set_title(f"{t.upper()}  med={r['median_err_%']:.2f}%  p95={r['p95_err_%']:.1f}%", fontsize=10)

        ax.set_xlabel(f"SPICE |{t}| (A)"); ax.set_ylabel(f"ML |{t}| (A)")

    fig.suptitle(f"{case} test-set parity ({model_name})")

    plt.tight_layout()

    plt.savefig(out_path, dpi=200)

    plt.close()





def main():

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    comp_dir = RESULTS_DIR / "model_comparison"

    comp_dir.mkdir(parents=True, exist_ok=True)



    model_types = available_model_types()

    if len(model_types) < 4:

        print(f"[!] Only {len(model_types)} model types available: {model_types}")

    print("Model types:", model_types)



    comparison, per_terminal, chosen_rows = [], [], []



    for case in CASES:

        print(f"\n---> Case: {case} <---")

        df = pd.read_csv(DATA_DIR / f"{case}.csv")

        X, y = df[FEATURE_COLS], df[TARGET_COLS]



        # 70 / 15 / 15 with a fixed seed

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED)

        X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED)



        best = (None, np.inf, None)

        for name in model_types:

            t0 = time.time()

            model = SignedCurrentModel(name, seed=RANDOM_SEED).fit(X_tr, y_tr)

            m = terminal_metrics(y_va.values, model.predict(X_va))

            r2, med, p95 = pooled(m)

            print(f"  [{name:13s}] val R2={r2:.4f} | median={med:6.2f}% | p95={p95:7.2f}% "

                  f"| {time.time() - t0:5.0f}s")

            comparison.append({'case': case, 'model': name, 'val_r2': r2,

                               'val_median_error_%': med, 'val_95th_percentile_error_%': p95})

            for _, r in m.iterrows():

                per_terminal.append({'case': case, 'model': name, **r.to_dict()})

            sc = selection_score(m)

            if sc < best[1]:

                best = (name, sc, model)



        best_name, _, best_model = best

        y_te_pred = best_model.predict(X_te)

        test_m = terminal_metrics(y_te.values, y_te_pred)

        r2, med, p95 = pooled(test_m)

        print(f"  >>> Selected {best_name} | TEST R2={r2:.4f} median={med:.2f}% p95={p95:.2f}%")

        print(test_m.to_string(index=False, float_format=lambda v: f"{v:.3f}"))



        chosen_rows.append({'case': case, 'model': best_name, 'test_r2': r2,

                            'test_median_error_%': med, 'test_95th_percentile_error_%': p95,

                            **{f"{r['terminal']}_median_%": r['median_err_%'] for _, r in test_m.iterrows()},

                            **{f"{r['terminal']}_p95_%": r['p95_err_%'] for _, r in test_m.iterrows()}})



        joblib.dump(best_model, SAVED_MODELS_DIR / f"{case}_best_model.joblib")

        parity_plot(case, best_name, y_te.values, y_te_pred, test_m, comp_dir / f"parity_test_{case}.png")



    pd.DataFrame(comparison).to_csv(comp_dir / "model_comparison_table.csv", index=False)

    pd.DataFrame(per_terminal).to_csv(comp_dir / "model_comparison_per_terminal.csv", index=False)

    pd.DataFrame(chosen_rows).to_csv(comp_dir / "chosen_models_test_metrics.csv", index=False)

    print(f"\n[+] Saved comparison tables and chosen-model test metrics to {comp_dir}")





if __name__ == "__main__":

    main()
