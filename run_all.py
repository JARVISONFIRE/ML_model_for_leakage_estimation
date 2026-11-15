#!/usr/bin/env python3

"""

Master Pipeline Runner (Deliverable 9)

Runs all steps in order with fixed random seeds.



Usage:

    python run_all.py                 # run everything

    python run_all.py --from 3        # start at step 3 (e.g. after changing only the ML code)

    python run_all.py --fresh         # additionally force the CLA SPICE simulation to re-run

    CLA_NUM_CORNERS=20 python run_all.py --from 6     # quick CLA run on 20 corners



Steps 1, 4 and 6 call ngspice (slow); steps 2, 3 and 5 are ML/analysis only.

After every step the script prints which output files were (re)written, so a step that

silently reused old results is visible immediately.

"""

import argparse

import os

import subprocess

import sys

import time

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"



SCRIPTS = [

    ("Step 1: Transistor Data Generation (ngspice)", "01_generate_transistor_data.py", ["data"]),

    ("Step 2: Data Validation & Trend Plots", "02_data_validation_check.py", ["results/data_check"]),

    ("Step 3: ML Model Training & Selection", "03_train_transistor_models.py",

     ["saved_models", "results/model_comparison"]),

    ("Step 4: SPICE Standard Gates Simulation (ngspice)", "04_simulate_gates.py",

     ["results/gate_results/gate_spice_reference.csv"]),

    ("Step 5: Gate ML Evaluation & Parity", "05_evaluate_gates_ml.py",

     ["results/gate_results/gate_ml_vs_spice_full.csv", "results/gate_results/gate_results_table.csv"]),

    ("Step 6: 3-Bit CLA Simulation & Evaluation", "06_simulate_and_eval_cla.py", ["results/cla_results"]),

]





def newest_mtime(rel: str) -> float:

    p = PROJECT_ROOT / rel

    if p.is_dir():

        files = [f for f in p.rglob("*") if f.is_file()]

        return max((f.stat().st_mtime for f in files), default=0.0)

    return p.stat().st_mtime if p.exists() else 0.0





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--from", dest="start", type=int, default=1, help="first step to run (1-6)")

    ap.add_argument("--fresh", action="store_true", help="force CLA SPICE re-simulation")

    args = ap.parse_args()



    env = dict(os.environ)

    if args.fresh:

        env["LEAKAGE_FRESH"] = "1"



    print("=" * 70)

    print("   ML-DRIVEN PVT VARIATION AWARE LEAKAGE ESTIMATION PIPELINE")

    print("   Technology: 32nm HP | VDD = 1.0V | Test Circuit: 3-bit CLA")

    print(f"   Starting at step {args.start}" + ("  (fresh CLA SPICE)" if args.fresh else ""))

    print("=" * 70)



    total_start = time.time()

    for i, (desc, script, outputs) in enumerate(SCRIPTS, start=1):

        if i < args.start:

            print(f"\n--- Skipping {desc} (using existing outputs) ---")

            continue

        print(f"\n>>> Running {desc} ({script}) <<<")

        before = {o: newest_mtime(o) for o in outputs}

        t0 = time.time()

        ret = subprocess.run([sys.executable, str(SRC_DIR / script)], cwd=str(PROJECT_ROOT), env=env)

        elapsed = time.time() - t0

        if ret.returncode != 0:

            print(f"[!] ERROR: {script} failed with return code {ret.returncode}.")

            sys.exit(ret.returncode)

        print(f"[✓] {desc} completed in {elapsed:.1f} seconds.")

        for o in outputs:

            changed = newest_mtime(o) > before[o]

            print(f"    {'UPDATED  ' if changed else 'UNCHANGED'}  {o}")



    print("\n" + "=" * 70)

    print(f" [✓] PIPELINE FINISHED IN {(time.time() - total_start) / 60:.2f} MINUTES")

    print("=" * 70)





if __name__ == "__main__":

    main()
