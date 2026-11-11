"""
Script 02: Data Validation Check (Deliverable 3)
1. Kirchhoff Current Law (KCL) Check: sum of terminal currents (Id + Ig + Is + Ib ~ 0).
2. Three Trend Plots:
   - OFF current vs Temperature
   - OFF current vs vth0
   - Gate current vs toxe
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, RESULTS_DIR, CASES

sns.set_theme(style="whitegrid", font_scale=1.1)


def perform_kcl_check():
    """Validates Kirchhoff's Current Law for each transistor dataset."""
    kcl_dir = RESULTS_DIR / "data_check"
    kcl_dir.mkdir(parents=True, exist_ok=True)
    kcl_results = []
    print("\n" + "="*50)
    print("      KIRCHHOFF'S CURRENT LAW (KCL) CHECK")
    print("="*50)

    for case in CASES:
        csv_path = DATA_DIR / f"{case}.csv"
        if not csv_path.exists():
            print(f"[!] Warning: {csv_path} not found. Please run 01_generate_transistor_data.py first.")
            return

        df = pd.read_csv(csv_path)
        # Sum of 4 terminal currents
        i_sum = df['id'] + df['ig'] + df['is'] + df['ib']
        abs_sum = np.abs(i_sum)

        mean_err = np.mean(abs_sum)
        median_err = np.median(abs_sum)
        max_err = np.max(abs_sum)

        print(f"\n--- {case} ---")
        print(f"  Mean |Id + Ig + Is + Ib|   : {mean_err:.3e} A")
        print(f"  Median |Id + Ig + Is + Ib| : {median_err:.3e} A")
        print(f"  Max |Id + Ig + Is + Ib|    : {max_err:.3e} A")

        kcl_results.append({
            'case': case,
            'mean_abs_err_A': mean_err,
            'median_abs_err_A': median_err,
            'max_abs_err_A': max_err
        })

    df_kcl = pd.DataFrame(kcl_results)
    out_csv = RESULTS_DIR / "data_check" / "kcl_summary.csv"
    df_kcl.to_csv(out_csv, index=False)
    print(f"\n[+] Saved KCL Summary to {out_csv}")


def generate_trend_plots():
    """Generates the three required physical trend plots."""
    out_dir = RESULTS_DIR / "data_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. OFF current vs Temperature (NMOS_OFF & PMOS_OFF)
    df_noff = pd.read_csv(DATA_DIR / "NMOS_OFF.csv")
    df_poff = pd.read_csv(DATA_DIR / "PMOS_OFF.csv")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df_noff, x='temp', y=np.abs(df_noff['id']), ax=ax[0], alpha=0.3, color='steelblue')
    ax[0].set_yscale('log')
    ax[0].set_title("NMOS OFF: |Id| vs Temperature")
    ax[0].set_xlabel("Temperature (°C)")
    ax[0].set_ylabel("Drain Current |Id| (A) [log scale]")

    sns.scatterplot(data=df_poff, x='temp', y=np.abs(df_poff['id']), ax=ax[1], alpha=0.3, color='crimson')
    ax[1].set_yscale('log')
    ax[1].set_title("PMOS OFF: |Id| vs Temperature")
    ax[1].set_xlabel("Temperature (°C)")
    ax[1].set_ylabel("Drain Current |Id| (A) [log scale]")

    plt.tight_layout()
    plt.savefig(out_dir / "trend_1_off_current_vs_temp.png", dpi=300)
    plt.close()
    print("[+] Saved trend_1_off_current_vs_temp.png")

    # 2. OFF current vs vth0 (NMOS_OFF & PMOS_OFF)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df_noff, x='vth0', y=np.abs(df_noff['id']), ax=ax[0], alpha=0.3, color='steelblue')
    ax[0].set_yscale('log')
    ax[0].set_title("NMOS OFF: |Id| vs vth0")
    ax[0].set_xlabel("Threshold Voltage vth0 (V)")
    ax[0].set_ylabel("Drain Current |Id| (A) [log scale]")

    sns.scatterplot(data=df_poff, x=np.abs(df_poff['vth0']), y=np.abs(df_poff['id']), ax=ax[1], alpha=0.3, color='crimson')
    ax[1].set_yscale('log')
    ax[1].set_title("PMOS OFF: |Id| vs |vth0|")
    ax[1].set_xlabel("|Threshold Voltage| |vth0| (V)")
    ax[1].set_ylabel("Drain Current |Id| (A) [log scale]")

    plt.tight_layout()
    plt.savefig(out_dir / "trend_2_off_current_vs_vth0.png", dpi=300)
    plt.close()
    print("[+] Saved trend_2_off_current_vs_vth0.png")

    # 3. Gate current vs toxe (NMOS_ON & PMOS_ON)
    df_non = pd.read_csv(DATA_DIR / "NMOS_ON.csv")
    df_pon = pd.read_csv(DATA_DIR / "PMOS_ON.csv")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df_non, x=df_non['toxe']*1e9, y=np.abs(df_non['ig']), ax=ax[0], alpha=0.3, color='forestgreen')
    ax[0].set_yscale('log')
    ax[0].set_title("NMOS ON: |Ig| vs toxe")
    ax[0].set_xlabel("Oxide Thickness toxe (nm)")
    ax[0].set_ylabel("Gate Current |Ig| (A) [log scale]")

    sns.scatterplot(data=df_pon, x=df_pon['toxe']*1e9, y=np.abs(df_pon['ig']), ax=ax[1], alpha=0.3, color='darkorange')
    ax[1].set_yscale('log')
    ax[1].set_title("PMOS ON: |Ig| vs toxe")
    ax[1].set_xlabel("Oxide Thickness toxe (nm)")
    ax[1].set_ylabel("Gate Current |Ig| (A) [log scale]")

    plt.tight_layout()
    plt.savefig(out_dir / "trend_3_gate_current_vs_toxe.png", dpi=300)
    plt.close()
    print("[+] Saved trend_3_gate_current_vs_toxe.png")


def main():
    perform_kcl_check()
    generate_trend_plots()


if __name__ == "__main__":
    main()
