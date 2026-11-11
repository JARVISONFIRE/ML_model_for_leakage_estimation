"""
Script 04: SPICE Gate Simulation for Standard CMOS Gates (Part B)
Simulates INV, NAND2, NOR2, AND2, OR2 across all 18 input cases at 200 random corners.
Records SPICE leakage, intermediate node voltages (x, y), and output voltages.
Also exports gate netlists for Deliverable 2.
"""
import sys
import os
import re
import subprocess
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    NETLISTS_DIR, RESULTS_DIR, NOMINAL_PROCESS_PARAMS, PROCESS_VARIATION,
    VDD_NOM, VDD_MIN, VDD_MAX, TEMP_MIN, TEMP_MAX, W_MIN, W_MAX,
    NUM_GATE_CORNERS, RANDOM_SEED, SPICE_OPTIONS, L_MIN
)
from src.spice_utils import build_custom_model_card, draw_gate_corners

GATE_CONFIGS = {
    'INV': ['0', '1'],
    'NAND2': ['00', '01', '10', '11'],
    'NOR2':  ['00', '01', '10', '11'],
    'AND2':  ['00', '01', '10', '11'],
    'OR2':   ['00', '01', '10', '11'],
}



def build_gate_netlist(gate: str, in_bits: str, corner: dict) -> str:
    """Builds the SPICE deck for a logic gate."""
    vdd = corner['vdd']
    temp = corner['temp']
    wn = corner['wn']
    wp = corner['wp']

    nmos_card = build_custom_model_card(
        'nmos', corner['vth0_n'], corner['toxe_n'], corner['ndep_n'], corner['u0_n'], corner['xj_n'], 'nmos_gate'
    )
    pmos_card = build_custom_model_card(
        'pmos', corner['vth0_p'], corner['toxe_p'], corner['ndep_p'], corner['u0_p'], corner['xj_p'], 'pmos_gate'
    )

    va = vdd if in_bits[0] == '1' else 0.0
    vb = (vdd if in_bits[1] == '1' else 0.0) if len(in_bits) > 1 else 0.0

    common_hdr = f"""* Gate Simulation: {gate} input={in_bits}
.options {SPICE_OPTIONS} temp={temp}

{nmos_card}
{pmos_card}

Vdd vdd 0 DC {vdd}
Va a 0 DC {va}
"""
    if len(in_bits) > 1:
        common_hdr += f"Vb b 0 DC {vb}\n"

    if gate == "INV":
        ckt = f"""
* INV (PMOS pull-up, NMOS pull-down)
Mp out a vdd vdd pmos_gate W={wp} L={L_MIN}
Mn out a 0   0   nmos_gate W={wn} L={L_MIN}

.control
op
print i(Vdd) v(out)
.endc
"""
    elif gate == "NAND2":
        ckt = f"""
* NAND2 (Parallel PMOS pull-up, series NMOS pull-down with internal node x)
Mp1 out a vdd vdd pmos_gate W={wp} L={L_MIN}
Mp2 out b vdd vdd pmos_gate W={wp} L={L_MIN}
Mn1 out a x   0   nmos_gate W={wn} L={L_MIN}
Mn2 x   b 0   0   nmos_gate W={wn} L={L_MIN}

.control
op
print i(Vdd) v(x) v(out)
.endc
"""
    elif gate == "NOR2":
        ckt = f"""
* NOR2 (Series PMOS pull-up with internal node y, parallel NMOS pull-down)
Mp1 y   a vdd vdd pmos_gate W={wp} L={L_MIN}
Mp2 out b y   vdd pmos_gate W={wp} L={L_MIN}
Mn1 out a 0   0   nmos_gate W={wn} L={L_MIN}
Mn2 out b 0   0   nmos_gate W={wn} L={L_MIN}

.control
op
print i(Vdd) v(y) v(out)
.endc
"""
    elif gate == "AND2":
        ckt = f"""
* AND2 (NAND2 followed by INV)
Mp1 n_out a vdd vdd pmos_gate W={wp} L={L_MIN}
Mp2 n_out b vdd vdd pmos_gate W={wp} L={L_MIN}
Mn1 n_out a x   0   nmos_gate W={wn} L={L_MIN}
Mn2 x     b 0   0   nmos_gate W={wn} L={L_MIN}

* INV stage
Mpi out n_out vdd vdd pmos_gate W={wp} L={L_MIN}
Mni out n_out 0   0   nmos_gate W={wn} L={L_MIN}

.control
op
print i(Vdd) v(x) v(n_out) v(out)
.endc
"""
    elif gate == "OR2":
        ckt = f"""
* OR2 (NOR2 followed by INV)
Mp1 y     a vdd vdd pmos_gate W={wp} L={L_MIN}
Mp2 n_out b y   vdd pmos_gate W={wp} L={L_MIN}
Mn1 n_out a 0   0   nmos_gate W={wn} L={L_MIN}
Mn2 n_out b 0   0   nmos_gate W={wn} L={L_MIN}

* INV stage
Mpi out n_out vdd vdd pmos_gate W={wp} L={L_MIN}
Mni out n_out 0   0   nmos_gate W={wn} L={L_MIN}

.control
op
print i(Vdd) v(y) v(n_out) v(out)
.endc
"""
    else:
        raise ValueError(f"Unknown gate: {gate}")

    return common_hdr + ckt + "\n.end\n"


def _simulate_gate_worker(args):
    gate, in_bits, corner = args
    netlist = build_gate_netlist(gate, in_bits, corner)

    res = subprocess.run(["ngspice", "-b"], input=netlist, capture_output=True, text=True)
    out = res.stdout

    # In SPICE, current leaving Vdd rail into circuit is -i(Vdd)
    m_vdd = re.search(r"i\(vdd\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+)", out, re.IGNORECASE)
    spice_leakage = -float(m_vdd.group(1)) if m_vdd else np.nan

    m_out = re.search(r"v\(out\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+|\d+(?:\.\d+)?)", out, re.IGNORECASE)
    v_out = float(m_out.group(1)) if m_out else np.nan

    m_x = re.search(r"v\(x\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+|\d+(?:\.\d+)?)", out, re.IGNORECASE)
    v_x = float(m_x.group(1)) if m_x else 0.0

    m_y = re.search(r"v\(y\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+|\d+(?:\.\d+)?)", out, re.IGNORECASE)
    v_y = float(m_y.group(1)) if m_y else corner['vdd']

    m_nout = re.search(r"v\(n_out\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+|\d+(?:\.\d+)?)", out, re.IGNORECASE)
    v_nout = float(m_nout.group(1)) if m_nout else np.nan

    record = {
        'gate': gate,
        'input': in_bits,
        'corner_id': corner['corner_id'],
        'vdd': corner['vdd'],
        'temp': corner['temp'],
        'wn': corner['wn'],
        'wp': corner['wp'],
        'vth0_n': corner['vth0_n'], 'toxe_n': corner['toxe_n'], 'ndep_n': corner['ndep_n'], 'u0_n': corner['u0_n'], 'xj_n': corner['xj_n'],
        'vth0_p': corner['vth0_p'], 'toxe_p': corner['toxe_p'], 'ndep_p': corner['ndep_p'], 'u0_p': corner['u0_p'], 'xj_p': corner['xj_p'],
        'spice_leakage': spice_leakage,
        'v_out': v_out,
        'v_x': v_x,
        'v_y': v_y,
        'v_nout': v_nout
    }
    return record


def export_example_gate_netlists():
    """Exports representative SPICE netlists for all 5 gates (Deliverable 2)."""
    gate_dir = NETLISTS_DIR / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)

    corners = draw_gate_corners(1, seed=RANDOM_SEED)
    c = corners[0]

    for gate, inputs in GATE_CONFIGS.items():
        netlist = build_gate_netlist(gate, inputs[0], c)
        out_file = gate_dir / f"{gate}_example.net"
        with open(out_file, "w") as f:
            f.write(netlist)
        print(f"  [+] Saved gate example netlist: {out_file}")


def main():
    gate_res_dir = RESULTS_DIR / "gate_results"
    gate_res_dir.mkdir(parents=True, exist_ok=True)

    print("=== Exporting Example Gate Netlists (Deliverable 2) ===")
    export_example_gate_netlists()

    corners = draw_gate_corners(NUM_GATE_CORNERS, seed=RANDOM_SEED)

    tasks = []
    for gate, in_list in GATE_CONFIGS.items():
        for in_bits in in_list:
            for c in corners:
                tasks.append((gate, in_bits, c))

    print(f"\n=== Simulating 18 Gate Cases across {NUM_GATE_CORNERS} Corners ({len(tasks)} runs) ===")
    n_cpus = max(1, mp.cpu_count() - 1)
    results = []

    with mp.Pool(processes=n_cpus) as pool:
        for res in tqdm(pool.imap(_simulate_gate_worker, tasks, chunksize=50), total=len(tasks), desc="Gates SPICE"):
            results.append(res)

    df_gates = pd.DataFrame(results)
    out_csv = gate_res_dir / "gate_spice_reference.csv"
    df_gates.to_csv(out_csv, index=False)
    print(f"\n[+] Saved SPICE Gate Reference to {out_csv} ({len(df_gates)} rows).")


if __name__ == "__main__":
    main()
