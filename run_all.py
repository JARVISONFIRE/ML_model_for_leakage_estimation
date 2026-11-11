#!/usr/bin/env python3
"""
Master Pipeline Runner (Deliverable 9)
Executes all project steps end-to-end in order with fixed random seeds:
1. Generate Transistor Datasets & Netlists (Deliverables 1 & 2)
2. Data Checks: KCL & Trend Plots (Deliverable 3)
3. Train & Compare 4 ML Models (Deliverables 4 & 5)
4. SPICE Simulation of 18 Gate Cases (Part B)
5. Gate-Level ML Evaluation & Parity Plots (Deliverables 6 & 7)
6. 3-Bit CLA Simulation & Hierarchical ML Evaluation (Deliverable 8)
"""
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

SCRIPTS = [
    ("Step 1: Transistor Data Generation", SRC_DIR / "01_generate_transistor_data.py"),
    ("Step 2: Data Validation & Trend Plots", SRC_DIR / "02_data_validation_check.py"),
    ("Step 3: ML Model Training & Selection", SRC_DIR / "03_train_transistor_models.py"),
    ("Step 4: SPICE Standard Gates Simulation", SRC_DIR / "04_simulate_gates.py"),
    ("Step 5: Gate ML Evaluation & Parity", SRC_DIR / "05_evaluate_gates_ml.py"),
    ("Step 6: 3-Bit CLA Simulation & Evaluation", SRC_DIR / "06_simulate_and_eval_cla.py"),
]


def main():
    print("=" * 70)
    print("   ML-DRIVEN PVT VARIATION AWARE LEAKAGE ESTIMATION PIPELINE")
    print("   Technology: 32nm HP | VDD = 1.0V | Test Circuit: 3-bit CLA")
    print("=" * 70)

    total_start = time.time()

    for desc, script_path in SCRIPTS:
        print(f"\n>>> Running {desc} ({script_path.name}) <<<")
        t0 = time.time()
        ret = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0

        if ret.returncode != 0:
            print(f"[!] ERROR: {script_path.name} failed with return code {ret.returncode}.")
            sys.exit(ret.returncode)
        else:
            print(f"[✓] {desc} completed in {elapsed:.1f} seconds.")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f" [✓] COMPLETE PIPELINE FINISHED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print(" All Deliverables 1 through 8 have been generated in data/ and results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
