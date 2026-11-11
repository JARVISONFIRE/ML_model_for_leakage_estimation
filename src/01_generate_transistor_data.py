"""
Script 01: Multi-Core SPICE Data Generation for Single Transistors (Part A)
Generates 10,000 samples each for NMOS_ON, NMOS_OFF, PMOS_ON, PMOS_OFF (Deliverable 1)
Also exports sample single-transistor netlists for Deliverable 2.
"""
import sys
import os
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR, NETLISTS_DIR, NOMINAL_PROCESS_PARAMS, PROCESS_VARIATION,
    VDD_NOM, VDD_MIN, VDD_MAX, TEMP_MIN, TEMP_MAX, W_MIN, W_MAX,
    NUM_TRANSISTOR_SAMPLES, RANDOM_SEED, CASES, SPICE_OPTIONS, L_MIN
)
from src.spice_utils import run_single_transistor_sim, build_custom_model_card


def draw_samples(case: str, n_samples: int, seed: int = RANDOM_SEED) -> list:
    """Draws uniform random samples for PVT parameters with independent seed per case."""
    np.random.seed(seed)
    is_nmos = "NMOS" in case
    fet_key = 'nmos' if is_nmos else 'pmos'
    nom = NOMINAL_PROCESS_PARAMS[fet_key]

    # Draw VDD first, then Vd and Vs depend on VDD
    vdd = np.random.uniform(VDD_MIN, VDD_MAX, n_samples)
    vd  = np.random.uniform(0.0, vdd)
    vs  = np.random.uniform(0.0, vdd)
    w   = np.random.uniform(W_MIN, W_MAX, n_samples)
    temp = np.random.uniform(TEMP_MIN, TEMP_MAX, n_samples)

    # Process parameters +/- 10%
    proc = {}
    for p in ['vth0', 'toxe', 'ndep', 'u0', 'xj']:
        val = nom[p]
        low = min(val * (1 - PROCESS_VARIATION), val * (1 + PROCESS_VARIATION))
        high = max(val * (1 - PROCESS_VARIATION), val * (1 + PROCESS_VARIATION))
        proc[p] = np.random.uniform(low, high, n_samples)

    rows = []
    for i in range(n_samples):
        row = {
            'vdd': vdd[i],
            'vd': vd[i],
            'vs': vs[i],
            'w': w[i],
            'temp': temp[i],
            'vth0': proc['vth0'][i],
            'toxe': proc['toxe'][i],
            'ndep': proc['ndep'][i],
            'u0': proc['u0'][i],
            'xj': proc['xj'][i],
        }
        rows.append(row)
    return rows


def _worker(args):
    case, row = args
    return run_single_transistor_sim(case, row)


def export_example_netlist(case: str):
    """Exports an example single-transistor netlist for Deliverable 2."""
    trans_dir = NETLISTS_DIR / "transistor"
    trans_dir.mkdir(parents=True, exist_ok=True)

    is_nmos = "NMOS" in case
    is_on = "ON" in case
    fet_type = "nmos" if is_nmos else "pmos"
    nom = NOMINAL_PROCESS_PARAMS[fet_type]

    model_card = build_custom_model_card(
        fet_type, nom['vth0'], nom['toxe'], nom['ndep'], nom['u0'], nom['xj'], model_name=f"{fet_type}_nom"
    )

    vg = "1.0" if ((is_nmos and is_on) or (not is_nmos and not is_on)) else "0.0"
    vb = "0.0" if is_nmos else "1.0"

    netlist = f"""* Example Single Transistor Netlist - {case} (Deliverable 2)
.options {SPICE_OPTIONS} temp=27.0

{model_card}

* Terminal Voltage Sources
Vd d 0 DC 1.0
Vg g 0 DC {vg}
Vs s 0 DC 0.0
Vb b 0 DC {vb}

* Transistor instance
M1 d g s b {fet_type}_nom W={2*L_MIN} L={L_MIN}

.control
op
print i(Vd) i(Vg) i(Vs) i(Vb)
.endc
.end
"""
    out_path = trans_dir / f"{case}_example.net"
    with open(out_path, "w") as f:
        f.write(netlist)
    print(f"  [+] Saved example netlist: {out_path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n_cpus = max(1, mp.cpu_count() - 1)
    print(f"=== Starting Data Generation using {n_cpus} CPU cores ===")
    print(f"Target: {NUM_TRANSISTOR_SAMPLES} samples per case across 4 cases")

    for idx, case in enumerate(CASES):
        print(f"\n[{idx+1}/{len(CASES)}] Generating dataset for {case}...")
        export_example_netlist(case)

        sample_rows = draw_samples(case, NUM_TRANSISTOR_SAMPLES, seed=RANDOM_SEED + idx)
        tasks = [(case, row) for row in sample_rows]

        results = []
        with mp.Pool(processes=n_cpus) as pool:
            for res in tqdm(pool.imap(_worker, tasks, chunksize=100), total=len(tasks), desc=case):
                results.append(res)

        df = pd.DataFrame(results)
        # Reorder columns explicitly per Deliverable 1
        cols = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb', 'vth0', 'toxe', 'ndep', 'u0', 'xj', 'id', 'ig', 'is', 'ib']
        df = df[cols]

        out_csv = DATA_DIR / f"{case}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  [+] Saved {out_csv} ({len(df)} records).")


if __name__ == "__main__":
    main()
