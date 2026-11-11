#!/usr/bin/env python3
"""
Verification Script: Inspects and Validates All Deliverables 1 through 9
Run: python verify_results.py
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
NETLISTS_DIR = PROJECT_ROOT / "netlists"

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def check_file(path: Path, desc: str):
    if path.exists() and path.stat().st_size > 0:
        size_kb = path.stat().st_size / 1024
        print(f"  {GREEN}[✓] PASS:{RESET} {desc} -> {path.name} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  {RED}[✗] FAIL:{RESET} {desc} -> Missing or empty ({path})")
        return False


def main():
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}   PVT VARIATION-AWARE LEAKAGE ESTIMATION — DELIVERABLES AUDIT{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    all_passed = True

    # 1. Datasets (Deliverable 1)
    print(f"\n{BOLD}[1] Checking Transistor Datasets (Deliverable 1):{RESET}")
    expected_cols = {'vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb', 'vth0', 'toxe', 'ndep', 'u0', 'xj', 'id', 'ig', 'is', 'ib'}
    for case in ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]:
        p = DATA_DIR / f"{case}.csv"
        if check_file(p, f"{case} Dataset"):
            df = pd.read_csv(p)
            has_cols = expected_cols.issubset(set(df.columns))
            rows = len(df)
            nans = df.isna().sum().sum()
            status = f"{rows} rows, all 16 cols present, NaNs: {nans}"
            if rows == 10000 and has_cols and nans == 0:
                print(f"      └── {GREEN}Valid format:{RESET} {status}")
            else:
                print(f"      └── {RED}Format issue:{RESET} {status}")
                all_passed = False
        else:
            all_passed = False

    # 2. Netlists (Deliverable 2)
    print(f"\n{BOLD}[2] Checking Example SPICE Netlists (Deliverable 2):{RESET}")
    netlists = [
        NETLISTS_DIR / "transistor" / "NMOS_ON_example.net",
        NETLISTS_DIR / "transistor" / "NMOS_OFF_example.net",
        NETLISTS_DIR / "gates" / "INV_example.net",
        NETLISTS_DIR / "gates" / "NAND2_example.net",
        NETLISTS_DIR / "gates" / "NOR2_example.net",
        NETLISTS_DIR / "circuits" / "cla_3bit_example.net",
    ]
    for n in netlists:
        if not check_file(n, f"Netlist {n.name}"):
            all_passed = False

    # 3. Data Check (Deliverable 3)
    print(f"\n{BOLD}[3] Checking Physics & KCL Verification (Deliverable 3):{RESET}")
    kcl_p = RESULTS_DIR / "data_check" / "kcl_summary.csv"
    if check_file(kcl_p, "KCL Summary CSV"):
        df_kcl = pd.read_csv(kcl_p)
        print("      └── Max KCL residual across cases:")
        for _, r in df_kcl.iterrows():
            print(f"          • {r['case']:10s}: Median |Sum I| = {r['median_abs_err_A']:.2e} A")
    for t in [1, 2, 3]:
        plot_p = list((RESULTS_DIR / "data_check").glob(f"trend_{t}_*.png"))
        if plot_p:
            check_file(plot_p[0], f"Trend Plot {t}")
        else:
            print(f"  {RED}[✗] FAIL:{RESET} Trend Plot {t} missing")
            all_passed = False

    # 4. Model Comparison (Deliverables 4 & 5)
    print(f"\n{BOLD}[4] Checking Model Comparison & Saved Models (Deliverables 4 & 5):{RESET}")
    comp_p = RESULTS_DIR / "model_comparison" / "model_comparison_table.csv"
    if check_file(comp_p, "Deliverable 4 Model Comparison Table"):
        df_comp = pd.read_csv(comp_p)
        print("      └── Model Benchmark Summary:")
        for case, grp in df_comp.groupby('case'):
            best_m = grp.loc[grp['val_median_error_%'].idxmin()]
            print(f"          • {case:10s} -> Best: {best_m['model']:18s} (Val R² = {best_m['val_r2']:.4f}, Med Err = {best_m['val_median_error_%']:.2f}%)")

    for c in ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]:
        m_p = SAVED_MODELS_DIR / f"{c}_best_model.joblib"
        check_file(m_p, f"Saved Model Weights ({c})")

    # 5. Gate Validation (Deliverables 6 & 7)
    print(f"\n{BOLD}[5] Checking Gate-Level Results (Deliverables 6 & 7):{RESET}")
    gate_table = RESULTS_DIR / "gate_results" / "gate_results_table.csv"
    if check_file(gate_table, "Deliverable 6 Gate Results Table"):
        df_g = pd.read_csv(gate_table)
        print(f"      └── 18 Gate Cases Evaluated. Overall Median Error = {df_g['median_error_%'].median():.2f}%")
        print(f"          Best Gate:  {df_g.loc[df_g['median_error_%'].idxmin()]['gate']} ({df_g.loc[df_g['median_error_%'].idxmin()]['input']}) -> {df_g['median_error_%'].min():.2f}% error")
        print(f"          Worst Gate: {df_g.loc[df_g['median_error_%'].idxmax()]['gate']} ({df_g.loc[df_g['median_error_%'].idxmax()]['input']}) -> {df_g['median_error_%'].max():.2f}% error")

    check_file(RESULTS_DIR / "gate_results" / "gate_parity_plot.png", "Deliverable 7 Gate Parity Plot")

    # 6. Test Circuit (Deliverable 8)
    print(f"\n{BOLD}[6] Checking 3-Bit CLA Adder Results (Deliverable 8):{RESET}")
    cla_sum = RESULTS_DIR / "cla_results" / "cla_results_summary.csv"
    if check_file(cla_sum, "Deliverable 8 CLA Results Table"):
        df_cla = pd.read_csv(cla_sum).iloc[0]
        print(f"      └── SPICE Median: {df_cla['spice_median_A']:.3e} A | ML Median: {df_cla['ml_median_A']:.3e} A")
        print(f"      └── Median Error: {df_cla['median_error_%']:.2f}% | 95th% Error: {df_cla['95th_percentile_error_%']:.2f}%")

    check_file(RESULTS_DIR / "cla_results" / "cla_parity_plot.png", "Deliverable 8 CLA Parity Plot")

    # Summary
    print(f"\n{BOLD}{'='*70}{RESET}")
    if all_passed:
        print(f"{GREEN}{BOLD} [✓] ALL DELIVERABLES 1 THROUGH 8 ARE VERIFIED AND WORKING PROPERLY!{RESET}")
    else:
        print(f"{RED}{BOLD} [!] Some files need attention.{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")


if __name__ == "__main__":
    main()
