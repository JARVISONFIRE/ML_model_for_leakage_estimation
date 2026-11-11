"""
Script 05: Gate-Level ML Evaluation and Comparison (Deliverables 6 & 7)
Uses trained transistor models to predict leakage for 18 gate cases across 200 corners.
Generates:
- Deliverable 6: Gate Results Table (SPICE median, ML median, bias, med err, 95th err)
- Deliverable 7: Gate Parity Plot on log axes
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAVED_MODELS_DIR, RESULTS_DIR, CASES
from src.model_wrapper import LogTransformedModel

sns.set_theme(style="whitegrid", font_scale=1.1)

FEATURE_COLS = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb', 'vth0', 'toxe', 'ndep', 'u0', 'xj']


def load_models():
    models = {}
    for c in CASES:
        p = SAVED_MODELS_DIR / f"{c}_best_model.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model file {p} not found. Run 03_train_transistor_models.py first.")
        models[c] = joblib.load(p)
    return models


def evaluate_gate_ml(row: pd.Series, models: dict) -> float:
    """
    Predicts static CMOS gate leakage current.
    In static CMOS, DC leakage drawn from the supply rail (VDD) is determined by the
    non-conducting (OFF) transistor network connecting the output/internal nodes to ground,
    or the OFF pull-up network connecting to VDD when output is low.
    """
    gate = row['gate']
    in_bits = str(row['input']).zfill(2) if gate != 'INV' else str(row['input'])
    vdd = float(row['vdd'])
    temp = float(row['temp'])
    wn = float(row['wn'])
    wp = float(row['wp'])

    v_out = float(row['v_out'])
    v_x = float(row['v_x'])
    v_y = float(row['v_y'])
    v_nout = float(row['v_nout'])

    # Process params
    vth0_n, toxe_n, ndep_n, u0_n, xj_n = row['vth0_n'], row['toxe_n'], row['ndep_n'], row['u0_n'], row['xj_n']
    vth0_p, toxe_p, ndep_p, u0_p, xj_p = row['vth0_p'], row['toxe_p'], row['ndep_p'], row['u0_p'], row['xj_p']

    def get_noff(vd, vs):
        feat = pd.DataFrame([[vdd, temp, wn, vd, vs, 0.0, 0.0, vth0_n, toxe_n, ndep_n, u0_n, xj_n]],
                            columns=FEATURE_COLS)
        pred = models['NMOS_OFF'].predict(feat)[0]
        return abs(float(pred[0]))

    def get_poff(vd, vs):
        feat = pd.DataFrame([[vdd, temp, wp, vd, vs, vdd, vdd, vth0_p, toxe_p, ndep_p, u0_p, xj_p]],
                            columns=FEATURE_COLS)
        pred = models['PMOS_OFF'].predict(feat)[0]
        return abs(float(pred[0]))

    if gate == "INV":
        if in_bits == '0':
            return get_noff(vd=v_out, vs=0.0)
        else:
            return get_poff(vd=v_out, vs=vdd)

    elif gate == "NAND2":
        if in_bits == '00':
            return get_noff(vd=v_out, vs=v_x)
        elif in_bits == '01':
            return get_noff(vd=v_out, vs=v_x)
        elif in_bits == '10':
            return get_noff(vd=v_x, vs=0.0)
        elif in_bits == '11':
            return 2.0 * get_poff(vd=v_out, vs=vdd)

    elif gate == "NOR2":
        if in_bits == '00':
            return 2.0 * get_noff(vd=v_out, vs=0.0)
        elif in_bits == '01':
            return get_poff(vd=v_out, vs=v_y)
        elif in_bits == '10':
            return get_poff(vd=v_y, vs=vdd)
        elif in_bits == '11':
            return get_poff(vd=v_y, vs=vdd)

    elif gate == "AND2":
        if in_bits == '00':
            n_leak = get_noff(vd=v_nout, vs=v_x)
        elif in_bits == '01':
            n_leak = get_noff(vd=v_nout, vs=v_x)
        elif in_bits == '10':
            n_leak = get_noff(vd=v_x, vs=0.0)
        else:
            n_leak = 2.0 * get_poff(vd=v_nout, vs=vdd)

        inv_in_is_one = (in_bits != '11')
        inv_leak = get_poff(vd=v_out, vs=vdd) if inv_in_is_one else get_noff(vd=v_out, vs=0.0)
        return n_leak + inv_leak

    elif gate == "OR2":
        if in_bits == '00':
            nr_leak = 2.0 * get_noff(vd=v_nout, vs=0.0)
        elif in_bits == '01':
            nr_leak = get_poff(vd=v_nout, vs=v_y)
        elif in_bits == '10':
            nr_leak = get_poff(vd=v_y, vs=vdd)
        else:
            nr_leak = get_poff(vd=v_y, vs=vdd)

        inv_in_is_one = (in_bits == '00')
        inv_leak = get_poff(vd=v_out, vs=vdd) if inv_in_is_one else get_noff(vd=v_out, vs=0.0)
        return nr_leak + inv_leak

    return 0.0


def main():
    gate_res_dir = RESULTS_DIR / "gate_results"
    ref_csv = gate_res_dir / "gate_spice_reference.csv"
    if not ref_csv.exists():
        print(f"[!] Error: {ref_csv} not found. Please run 04_simulate_gates.py first.")
        return

    df_spice = pd.read_csv(ref_csv)
    models = load_models()

    print("=== Computing Gate ML Leakage Predictions ===")
    ml_leakages = []
    for _, row in df_spice.iterrows():
        ml_leak = evaluate_gate_ml(row, models)
        ml_leakages.append(ml_leak)

    df_spice['ml_leakage'] = ml_leakages
    df_spice.to_csv(gate_res_dir / "gate_ml_vs_spice_full.csv", index=False)

    # Deliverable 6: Gate Results Table
    table_records = []
    grouped = df_spice.groupby(['gate', 'input'])

    for (gate, in_bits), group in grouped:
        spice_vals = np.abs(group['spice_leakage'].values)
        ml_vals = np.abs(group['ml_leakage'].values)

        spice_med = np.median(spice_vals)
        ml_med = np.median(ml_vals)
        bias = np.mean(ml_vals - spice_vals)

        rel_errors = np.abs(ml_vals - spice_vals) / np.maximum(spice_vals, 1e-15) * 100.0
        med_err = np.median(rel_errors)
        p95_err = np.percentile(rel_errors, 95)

        table_records.append({
            'gate': gate,
            'input': str(in_bits),
            'spice_median_A': spice_med,
            'ml_median_A': ml_med,
            'bias_A': bias,
            'median_error_%': med_err,
            '95th_percentile_error_%': p95_err
        })

    df_table = pd.DataFrame(table_records)
    table_csv = gate_res_dir / "gate_results_table.csv"
    df_table.to_csv(table_csv, index=False)
    print(f"\n[+] Saved Deliverable 6 Gate Results Table to {table_csv}")
    print(df_table.to_string(index=False))

    # Deliverable 7: Gate Parity Plot (log-log)
    fig, ax = plt.subplots(figsize=(7, 7))
    sp_all = np.abs(df_spice['spice_leakage'].values)
    ml_all = np.abs(df_spice['ml_leakage'].values)

    ax.scatter(sp_all, ml_all, alpha=0.35, s=20, color='darkorchid', label='Gate Simulations (18 cases x 200 corners)')
    min_v = max(1e-13, min(sp_all.min(), ml_all.min()))
    max_v = max(sp_all.max(), ml_all.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='Ideal 1:1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("SPICE Gate Leakage (A)")
    ax.set_ylabel("ML Predicted Gate Leakage (A)")
    all_rel = np.abs(ml_all - sp_all) / np.maximum(sp_all, 1e-15) * 100.0
    ax.set_title(f"Gate-Level Leakage Parity Plot (Deliverable 7)\nOverall Median Error = {np.median(all_rel):.2f}%")
    ax.legend()
    plt.tight_layout()
    parity_plot_path = gate_res_dir / "gate_parity_plot.png"
    plt.savefig(parity_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Deliverable 7 Gate Parity Plot to {parity_plot_path}")


if __name__ == "__main__":
    main()
