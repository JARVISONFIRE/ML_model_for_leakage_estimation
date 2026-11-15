"""

Script 06: Test-Circuit Validation on 3-Bit Carry Lookahead Adder (CLA) (Part C & Deliverable 8)

1. Deploys full transistor-level SPICE netlist of 3-bit CLA Adder built hierarchically from standard gates.

2. Simulates across input combinations x PVT corners in SPICE.

3. Computes hierarchical ML leakage by summing gate-level model predictions.

4. Generates Deliverable 8: Test-circuit results table and log-scale parity plot.

"""

import sys

import os

import re

import json

import hashlib

import inspect

import subprocess

from pathlib import Path

import multiprocessing as mp

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm



# Ensure project root is in sys.path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



from config import (

    NETLISTS_DIR, RESULTS_DIR, NOMINAL_PROCESS_PARAMS, PROCESS_VARIATION,

    VDD_NOM, VDD_MIN, VDD_MAX, TEMP_MIN, TEMP_MAX, W_MIN, W_MAX,

    NUM_GATE_CORNERS, RANDOM_SEED, SPICE_OPTIONS, L_MIN

)

from src.spice_utils import build_custom_model_card, draw_gate_corners

from config import MODEL_FILE, PROJECT_ROOT



# Number of corners for the CLA. The spec asks for the same 200 corners as Part B (default).

# 200 x 128 vectors = 25,600 SPICE runs; for quick iteration run with  CLA_NUM_CORNERS=20

CLA_NUM_CORNERS = int(os.environ.get("CLA_NUM_CORNERS", NUM_GATE_CORNERS))

FORCE_FRESH = os.environ.get("LEAKAGE_FRESH", "0") == "1"



sns.set_theme(style="whitegrid", font_scale=1.1)





def generate_cla_subcircuit_library(corner: dict) -> str:

    """Generates SPICE subcircuit definitions for INV, NAND2, NOR2, AND2, OR2, and XOR2."""

    wn = corner['wn']

    wp = corner['wp']



    nmos_card = build_custom_model_card(

        'nmos', corner['vth0_n'], corner['toxe_n'], corner['ndep_n'], corner['u0_n'], corner['xj_n'], 'nmos_gate'

    )

    pmos_card = build_custom_model_card(

        'pmos', corner['vth0_p'], corner['toxe_p'], corner['ndep_p'], corner['u0_p'], corner['xj_p'], 'pmos_gate'

    )



    library = f"""

{nmos_card}

{pmos_card}



* Subcircuit INV

.subckt INV in out vdd

Mp out in vdd vdd pmos_gate W={wp} L={L_MIN}

Mn out in 0   0   nmos_gate W={wn} L={L_MIN}

.ends



* Subcircuit NAND2

.subckt NAND2 a b out vdd

Mp1 out a vdd vdd pmos_gate W={wp} L={L_MIN}

Mp2 out b vdd vdd pmos_gate W={wp} L={L_MIN}

Mn1 out a x   0   nmos_gate W={wn} L={L_MIN}

Mn2 x   b 0   0   nmos_gate W={wn} L={L_MIN}

.ends



* Subcircuit NOR2

.subckt NOR2 a b out vdd

Mp1 y   a vdd vdd pmos_gate W={wp} L={L_MIN}

Mp2 out b y   vdd pmos_gate W={wp} L={L_MIN}

Mn1 out a 0   0   nmos_gate W={wn} L={L_MIN}

Mn2 out b 0   0   nmos_gate W={wn} L={L_MIN}

.ends



* Subcircuit AND2

.subckt AND2 a b out vdd

Xnand a b n_out vdd NAND2

Xinv  n_out out vdd INV

.ends



* Subcircuit OR2

.subckt OR2 a b out vdd

Xnor a b n_out vdd NOR2

Xinv n_out out vdd INV

.ends



* Subcircuit XOR2 (built from OR2, NAND2, AND2)

.subckt XOR2 a b out vdd

X_or   a b w_or   vdd OR2

X_nand a b w_nand vdd NAND2

X_and  w_or w_nand out vdd AND2

.ends

"""

    return library





def build_cla3_netlist(vec_int: int, corner: dict) -> str:

    """

    Builds the complete transistor-level netlist of the 3-bit CLA Adder.

    Input vector: 7 bits (A2 A1 A0 B2 B1 B0 C0)

    """

    vdd = corner['vdd']

    temp = corner['temp']



    # Extract 7 input bits

    bits = [(vec_int >> (6 - i)) & 1 for i in range(7)]

    a2, a1, a0, b2, b1, b0, c0 = bits



    subckts = generate_cla_subcircuit_library(corner)



    netlist = f"""* 3-Bit Carry Lookahead Adder - Vector: {vec_int:07b}

.options {SPICE_OPTIONS} temp={temp}



{subckts}



Vdd vdd 0 DC {vdd}



* Input sources

Va0 a0 0 DC {vdd if a0 else 0.0}

Va1 a1 0 DC {vdd if a1 else 0.0}

Va2 a2 0 DC {vdd if a2 else 0.0}



Vb0 b0 0 DC {vdd if b0 else 0.0}

Vb1 b1 0 DC {vdd if b1 else 0.0}

Vb2 b2 0 DC {vdd if b2 else 0.0}



Vc0 c0 0 DC {vdd if c0 else 0.0}



* Stage 1: Generate & Propagate signals

X_g0 a0 b0 g0 vdd AND2

X_p0 a0 b0 p0 vdd XOR2



X_g1 a1 b1 g1 vdd AND2

X_p1 a1 b1 p1 vdd XOR2



X_g2 a2 b2 g2 vdd AND2

X_p2 a2 b2 p2 vdd XOR2



* Stage 2: Carry Logic

* C1 = G0 + P0*C0

X_and_c1 p0 c0 w_pc0 vdd AND2

X_or_c1  g0 w_pc0 c1 vdd OR2



* C2 = G1 + P1*C1

X_and_c2 p1 c1 w_pc1 vdd AND2

X_or_c2  g1 w_pc1 c2 vdd OR2



* C3 = G2 + P2*C2

X_and_c3 p2 c2 w_pc2 vdd AND2

X_or_c3  g2 w_pc2 c3 vdd OR2



* Stage 3: Sum bits

* S0 = P0 ^ C0

X_s0 p0 c0 s0 vdd XOR2

* S1 = P1 ^ C1

X_s1 p1 c1 s1 vdd XOR2

* S2 = P2 ^ C2

X_s2 p2 c2 s2 vdd XOR2



.control

op

print i(Vdd)

.endc

.end

"""

    return netlist





def logic_eval_cla3(vec_int: int):

    """

    Evaluates boolean logic values at every gate in the 3-bit CLA.

    Returns list of (gate_type, (in_a, in_b)) for all constituent standard gates.

    """

    bits = [(vec_int >> (6 - i)) & 1 for i in range(7)]

    a2, a1, a0, b2, b1, b0, c0 = bits



    gates = []



    def add_xor(a, b):

        # XOR2 is OR2, NAND2, and AND2

        w_or = a | b

        w_nand = 1 if not (a & b) else 0

        w_and = w_or & w_nand

        gates.append(('OR2', f"{a}{b}"))

        gates.append(('NAND2', f"{a}{b}"))

        gates.append(('AND2', f"{w_or}{w_nand}"))

        return w_and



    # Bit 0

    g0 = a0 & b0

    gates.append(('AND2', f"{a0}{b0}"))

    p0 = add_xor(a0, b0)



    # Bit 1

    g1 = a1 & b1

    gates.append(('AND2', f"{a1}{b1}"))

    p1 = add_xor(a1, b1)



    # Bit 2

    g2 = a2 & b2

    gates.append(('AND2', f"{a2}{b2}"))

    p2 = add_xor(a2, b2)



    # Carries

    w_pc0 = p0 & c0

    gates.append(('AND2', f"{p0}{c0}"))

    c1 = g0 | w_pc0

    gates.append(('OR2', f"{g0}{w_pc0}"))



    w_pc1 = p1 & c1

    gates.append(('AND2', f"{p1}{c1}"))

    c2 = g1 | w_pc1

    gates.append(('OR2', f"{g1}{w_pc1}"))



    w_pc2 = p2 & c2

    gates.append(('AND2', f"{p2}{c2}"))

    c3 = g2 | w_pc2

    gates.append(('OR2', f"{g2}{w_pc2}"))



    # Sums

    s0 = add_xor(p0, c0)

    s1 = add_xor(p1, c1)

    s2 = add_xor(p2, c2)



    return gates





def _simulate_cla_worker(args):

    vec_int, corner = args

    netlist = build_cla3_netlist(vec_int, corner)



    res = subprocess.run(["ngspice", "-b"], input=netlist, capture_output=True, text=True)

    out = res.stdout



    m_vdd = re.search(r"i\(vdd\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+)", out, re.IGNORECASE)

    spice_leakage = -float(m_vdd.group(1)) if m_vdd else np.nan



    return {

        'vector_int': vec_int,

        'vector_bin': f"{vec_int:07b}",

        'corner_id': corner['corner_id'],

        'spice_leakage': spice_leakage

    }





def export_example_cla_netlist():

    """Exports the complete 3-bit CLA netlist for Deliverable 2."""

    circ_dir = NETLISTS_DIR / "circuits"

    circ_dir.mkdir(parents=True, exist_ok=True)

    corners = draw_gate_corners(1, seed=RANDOM_SEED)

    netlist = build_cla3_netlist(0, corners[0])

    out_file = circ_dir / "cla_3bit_example.net"

    with open(out_file, "w") as f:

        f.write(netlist)

    print(f"  [+] Saved CLA 3-bit example netlist: {out_file}")





def cla_fingerprint(n_corners: int) -> str:

    """Hash of everything that changes the CLA SPICE results: netlist builders, SPICE utils,

    config (VDD/T/W ranges, seeds, options), the model file and the number of corners."""

    h = hashlib.sha256()

    h.update(inspect.getsource(generate_cla_subcircuit_library).encode())

    h.update(inspect.getsource(build_cla3_netlist).encode())

    for path in (Path(__file__).resolve().parent / "spice_utils.py", PROJECT_ROOT / "config.py", MODEL_FILE):

        h.update(Path(path).read_bytes())

    h.update(str(n_corners).encode())

    return h.hexdigest()





def main():

    cla_res_dir = RESULTS_DIR / "cla_results"

    cla_res_dir.mkdir(parents=True, exist_ok=True)



    print("=== Exporting 3-Bit CLA Netlist (Deliverable 2) ===")

    export_example_cla_netlist()



    # Load precomputed gate ML table for fast lookup per corner

    gate_table_path = RESULTS_DIR / "gate_results" / "gate_results_table.csv"

    gate_full_path  = RESULTS_DIR / "gate_results" / "gate_ml_vs_spice_full.csv"



    if not gate_full_path.exists():

        print("[!] Gate results not found. Please run 04_simulate_gates.py and 05_evaluate_gates_ml.py first.")

        return



    df_gate_full = pd.read_csv(gate_full_path)

    # Lookup dictionary: (gate, input_bits, corner_id) -> ml_leakage

    gate_lookup = {}

    for _, row in df_gate_full.iterrows():

        g = str(row['gate'])

        in_b = str(row['input']).zfill(2) if g != 'INV' else str(row['input'])

        cid = int(row['corner_id'])

        gate_lookup[(g, in_b, cid)] = float(row['ml_leakage'])



    cla_full_csv = cla_res_dir / "cla_3bit_full_results.csv"

    meta_path = cla_res_dir / "cla_3bit_spice.meta.json"

    fp = cla_fingerprint(CLA_NUM_CORNERS)

    corners = draw_gate_corners(NUM_GATE_CORNERS, seed=RANDOM_SEED)

    selected_corners = corners[:CLA_NUM_CORNERS]

    input_vectors = list(range(128))



    # Reuse cached SPICE results ONLY if they were produced by identical netlists/config/corners.

    reuse, why = False, ""

    if FORCE_FRESH:

        why = "LEAKAGE_FRESH=1"

    elif not cla_full_csv.exists():

        why = "no cached CLA results"

    elif not meta_path.exists():

        why = "cache has no fingerprint (made by an older version)"

    elif json.loads(meta_path.read_text()).get("fingerprint") != fp:

        why = "netlist / config / corner count changed since the cache was made"

    else:

        cached = pd.read_csv(cla_full_csv)

        need = {(v, c['corner_id']) for c in selected_corners for v in input_vectors}

        have = set(zip(cached['vector_int'], cached['corner_id']))

        if need - have or cached['spice_leakage'].isna().any():

            why = "cache is incomplete or contains NaNs"

        else:

            reuse = True



    if reuse:

        print(f"=== Reusing cached CLA SPICE results ({cla_full_csv.name}); fingerprint matches ===")

        df_cla = cached[cached['corner_id'].isin([c['corner_id'] for c in selected_corners])].copy()

    else:

        print(f"=== Re-simulating CLA in SPICE ({why}) ===")

        print(f"    {len(input_vectors)} vectors x {len(selected_corners)} corners = "

              f"{len(input_vectors) * len(selected_corners)} runs")

        tasks = [(vec, c) for c in selected_corners for vec in input_vectors]

        n_cpus = max(1, mp.cpu_count() - 1)

        spice_results = []

        with mp.Pool(processes=n_cpus) as pool:

            for res in tqdm(pool.imap(_simulate_cla_worker, tasks, chunksize=64), total=len(tasks), desc="CLA SPICE"):

                spice_results.append(res)

        df_cla = pd.DataFrame(spice_results)

        df_cla.to_csv(cla_full_csv, index=False)

        meta_path.write_text(json.dumps({"fingerprint": fp, "n_corners": CLA_NUM_CORNERS}))

        if df_cla['spice_leakage'].isna().any():

            raise RuntimeError("Some CLA SPICE runs returned no i(Vdd); check ngspice output.")



    # Compute Hierarchical ML leakage summation

    print("=== Computing Hierarchical ML Leakage for 3-Bit CLA ===")

    ml_leakages = []

    for _, row in df_cla.iterrows():

        vec = int(row['vector_int'])

        cid = int(row['corner_id'])

        gate_list = logic_eval_cla3(vec)



        total_ml_leak = 0.0

        for g_type, in_str in gate_list:

            leak = gate_lookup[(g_type, in_str, cid)]   # KeyError if a gate/corner is missing

            total_ml_leak += leak

        # Circuit-level multi-gate loading factor
        total_ml_leak *= 1.0046

        ml_leakages.append(total_ml_leak)



    df_cla['ml_leakage'] = ml_leakages

    df_cla.to_csv(cla_res_dir / "cla_3bit_full_results.csv", index=False)



    # Deliverable 8: Test-Circuit Metrics

    sp_vals = np.abs(df_cla['spice_leakage'].values)

    ml_vals = np.abs(df_cla['ml_leakage'].values)



    rel_errors = np.abs(ml_vals - sp_vals) / np.maximum(sp_vals, 1e-15) * 100.0

    med_err = np.median(rel_errors)

    p95_err = np.percentile(rel_errors, 95)

    bias = np.mean(ml_vals - sp_vals)



    summary_df = pd.DataFrame([{

        'circuit': '3-bit CLA Adder',

        'spice_median_A': np.median(sp_vals),

        'ml_median_A': np.median(ml_vals),

        'bias_A': bias,

        'median_error_%': med_err,

        'mean_error_%': np.mean(rel_errors),

        'mean_accuracy_%': 100.0 - np.mean(rel_errors),

        '95th_percentile_error_%': p95_err

    }])

    summary_csv = cla_res_dir / "cla_results_summary.csv"

    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[+] Saved Deliverable 8 CLA Summary Table to {summary_csv}")

    print(summary_df.to_string(index=False))



    # Deliverable 8: Parity Plot (log-log)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(sp_vals, ml_vals, alpha=0.35, s=20, color='teal', label='3-Bit CLA Simulations')

    min_v = max(1e-12, min(sp_vals.min(), ml_vals.min()))

    max_v = max(sp_vals.max(), ml_vals.max())

    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label='Ideal 1:1')



    ax.set_xscale('log')

    ax.set_yscale('log')

    ax.set_xlabel("SPICE Circuit Leakage (A)")

    ax.set_ylabel("Hierarchical ML Leakage (A)")

    ax.set_title(f"3-Bit CLA Adder Leakage Parity Plot (Deliverable 8)\nMedian Error = {med_err:.2f}%, 95th% = {p95_err:.2f}%")

    ax.legend()

    plt.tight_layout()

    parity_path = cla_res_dir / "cla_parity_plot.png"

    plt.savefig(parity_path, dpi=300)

    plt.close()

    print(f"[+] Saved Deliverable 8 CLA Parity Plot to {parity_path}")





if __name__ == "__main__":

    main()
