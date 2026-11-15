"""

Script 05: Gate-Level ML Evaluation and Comparison (Deliverables 6 & 7)



Gate leakage is what SPICE measures: the total current drawn from the VDD rail, -i(Vdd).

By KCL that equals the sum of the (signed) currents entering every transistor terminal that

is connected to the VDD net (for these static gates: sum over PMOS of Is + Ib).



For every transistor we take the node voltages from SPICE (inputs exactly 0/VDD; out, x, y,

n_out recorded), decide ON/OFF from the logic level of its gate node, and ask the matching

trained model (NMOS_ON / NMOS_OFF / PMOS_ON / PMOS_OFF) for the signed id, ig, is, ib.



Why a KCL step is needed ("kcl" method, the default):

  An ON device carries a channel current of ~1e-5 A while the leakage flowing through it is

  ~1e-8 A, i.e. 0.1% of its channel current. No regression model on a 1e-4..1e-18 A range can

  resolve that, so the ON model's id/is are NOT trusted. Instead, for every ON device the

  unknown channel current c (id = c, is = -(c + ig + ib)) is solved from KCL at the internal

  nodes (out, x, y, n_out), using the accurate currents of the OFF devices and the ig/ib of

  the ON devices. The supply current is then summed over all terminals on the VDD net.

  The naive "direct" sum (trust every model output) is also computed as ml_leakage_direct

  for the ablation discussed in the report.



Outputs: gate_ml_vs_spice_full.csv, gate_results_table.csv, gate_parity_plot.png

(file names and the key columns are unchanged, so 06_simulate_and_eval_cla.py still works).

"""

import sys

from pathlib import Path



import joblib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



from config import SAVED_MODELS_DIR, RESULTS_DIR, CASES

from src.model_wrapper import FEATURE_COLS



PROC = ['vth0', 'toxe', 'ndep', 'u0', 'xj']

TERM_IDX = {'d': 0, 'g': 1, 's': 2, 'b': 3}          # model output columns: id, ig, is, ib



# (fet, drain, gate, source, body) -- identical to the netlists written by 04_simulate_gates.py

TOPO = {

    'INV':   [('P', 'out', 'a', 'vdd', 'vdd'), ('N', 'out', 'a', 'gnd', 'gnd')],

    'NAND2': [('P', 'out', 'a', 'vdd', 'vdd'), ('P', 'out', 'b', 'vdd', 'vdd'),

              ('N', 'out', 'a', 'x', 'gnd'),   ('N', 'x', 'b', 'gnd', 'gnd')],

    'NOR2':  [('P', 'y', 'a', 'vdd', 'vdd'),   ('P', 'out', 'b', 'y', 'vdd'),

              ('N', 'out', 'a', 'gnd', 'gnd'), ('N', 'out', 'b', 'gnd', 'gnd')],

    'AND2':  [('P', 'n_out', 'a', 'vdd', 'vdd'), ('P', 'n_out', 'b', 'vdd', 'vdd'),

              ('N', 'n_out', 'a', 'x', 'gnd'),   ('N', 'x', 'b', 'gnd', 'gnd'),

              ('P', 'out', 'n_out', 'vdd', 'vdd'), ('N', 'out', 'n_out', 'gnd', 'gnd')],

    'OR2':   [('P', 'y', 'a', 'vdd', 'vdd'),   ('P', 'n_out', 'b', 'y', 'vdd'),

              ('N', 'n_out', 'a', 'gnd', 'gnd'), ('N', 'n_out', 'b', 'gnd', 'gnd'),

              ('P', 'out', 'n_out', 'vdd', 'vdd'), ('N', 'out', 'n_out', 'gnd', 'gnd')],

}





def load_models():

    models = {}

    for c in CASES:

        p = SAVED_MODELS_DIR / f"{c}_best_model.joblib"

        if not p.exists():

            raise FileNotFoundError(f"Model file {p} not found. Run 03_train_transistor_models.py first.")

        models[c] = joblib.load(p)

    return models





def node_voltages(gate: str, bits: str, g: pd.DataFrame) -> dict:

    vdd = g['vdd'].values

    nodes = {'vdd': vdd, 'gnd': np.zeros(len(g)),

             'a': vdd * int(bits[0]),

             'out': g['v_out'].values, 'x': g['v_x'].values, 'y': g['v_y'].values,

             'n_out': g['v_nout'].values}

    if len(bits) > 1:

        nodes['b'] = vdd * int(bits[1])

    return nodes





def _predict_devices(gate: str, g: pd.DataFrame, nodes: dict, models: dict) -> list:

    """Signed (id, ig, is, ib) predictions for every transistor of the gate, per corner row."""

    vdd = g['vdd'].values

    devs = []

    for fet, d, gt, s, b in TOPO[gate]:

        high = nodes[gt] > vdd / 2.0

        on = high if fet == 'N' else ~high

        suffix = 'n' if fet == 'N' else 'p'

        width = g['wn'].values if fet == 'N' else g['wp'].values



        # terminal voltages as seen by the model: clipped to the trained range, exact rails on g and b

        feat = pd.DataFrame({'vdd': vdd, 'temp': g['temp'].values, 'w': width,

                             'vd': np.clip(nodes[d], 0.0, vdd), 'vs': np.clip(nodes[s], 0.0, vdd),

                             'vg': np.where(high, vdd, 0.0),

                             'vb': np.zeros(len(g)) if fet == 'N' else vdd})

        for p in PROC:

            feat[p] = g[f'{p}_{suffix}'].values

        feat = feat[FEATURE_COLS]



        cur = np.zeros((len(g), 4))

        for state_on in (True, False):

            mask = on == state_on

            if mask.any():

                case = f"{'NMOS' if fet == 'N' else 'PMOS'}_{'ON' if state_on else 'OFF'}"

                cur[mask] = models[case].predict(feat[mask])

        devs.append({'nodes': (d, gt, s, b), 'on': on, 'cur': cur})

    return devs





def gate_group_leakage(gate: str, bits: str, g: pd.DataFrame, models: dict):

    """Returns (kcl_leakage, direct_leakage) arrays for every corner row of one gate/input."""

    nodes = node_voltages(gate, bits, g)

    devs = _predict_devices(gate, g, nodes, models)

    n = len(g)



    def vdd_terms(dv):

        return [(k, TERM_IDX[k]) for k, nd in zip('dgsb', dv['nodes']) if nd == 'vdd']



    # --- direct: trust every model output

    direct = np.zeros(n)

    for dv in devs:

        for _, idx in vdd_terms(dv):

            direct += dv['cur'][:, idx]



    # --- kcl: solve ON-device channel currents from KCL at the internal nodes

    internal = [nd for nd in ('out', 'x', 'y', 'n_out') if any(nd in dv['nodes'] for dv in devs)]

    on_matrix = np.column_stack([dv['on'] for dv in devs])          # (n, n_dev)

    kcl = np.zeros(n)

    patterns = np.unique(on_matrix, axis=0)

    for pat in patterns:

        rows = np.where((on_matrix == pat).all(axis=1))[0]

        unknown = [k for k in range(len(devs)) if pat[k]]

        A = np.zeros((len(internal), len(unknown)))

        rhs = np.zeros((len(internal), len(rows)))

        for mi, node in enumerate(internal):

            for k, dv in enumerate(devs):

                cur = dv['cur'][rows]

                for term, nd in zip('dgsb', dv['nodes']):

                    if nd != node:

                        continue

                    if not pat[k] or term in 'gb':                  # OFF device, or gate/body of ON device

                        rhs[mi] -= cur[:, TERM_IDX[term]]

                    else:                                           # channel terminal of an ON device

                        j = unknown.index(k)

                        if term == 'd':

                            A[mi, j] += 1.0                         # id = c

                        else:

                            A[mi, j] -= 1.0                         # is = -(c + ig + ib)

                            rhs[mi] += cur[:, 1] + cur[:, 3]

        c = np.linalg.lstsq(A, rhs, rcond=None)[0] if unknown else np.zeros((0, len(rows)))



        total = np.zeros(len(rows))

        for k, dv in enumerate(devs):

            cur = dv['cur'][rows]

            for term, idx in vdd_terms(dv):

                if not pat[k] or term in 'gb':

                    total += cur[:, idx]

                else:

                    ck = c[unknown.index(k)]

                    total += ck if term == 'd' else -(ck + cur[:, 1] + cur[:, 3])

        kcl[rows] = total

    return kcl, direct





def main():

    gate_res_dir = RESULTS_DIR / "gate_results"

    ref_csv = gate_res_dir / "gate_spice_reference.csv"

    if not ref_csv.exists():

        print(f"[!] Error: {ref_csv} not found. Please run 04_simulate_gates.py first.")

        return



    df = pd.read_csv(ref_csv, dtype={'input': str})

    df['input'] = [b if gt == 'INV' else b.zfill(2) for gt, b in zip(df['gate'], df['input'])]

    models = load_models()



    print("=== Computing Gate ML Leakage Predictions (supply-current / KCL method) ===")

    df['ml_leakage'] = np.nan

    df['ml_leakage_direct'] = np.nan

    for (gate, bits), grp in df.groupby(['gate', 'input']):

        kcl, direct = gate_group_leakage(gate, bits, grp, models)

        df.loc[grp.index, 'ml_leakage'] = kcl

        df.loc[grp.index, 'ml_leakage_direct'] = direct

    df.to_csv(gate_res_dir / "gate_ml_vs_spice_full.csv", index=False)



    # ---- Deliverable 6: results table ----

    records = []

    for (gate, bits), grp in df.groupby(['gate', 'input']):

        sp = np.abs(grp['spice_leakage'].values)

        ml = grp['ml_leakage'].values

        signed_rel = (ml - sp) / sp * 100.0

        abs_rel = np.abs(signed_rel)

        records.append({

            'gate': gate, 'input': bits,

            'spice_median_A': np.median(sp), 'ml_median_A': np.median(ml),

            'bias_A': np.mean(ml - sp),                       # mean(ML - SPICE)

            'median_error_%': np.median(abs_rel),

            '95th_percentile_error_%': np.percentile(abs_rel, 95),

            'median_signed_error_%': np.median(signed_rel),   # sign shows over/under-prediction

        })

    table = pd.DataFrame(records)

    table.to_csv(gate_res_dir / "gate_results_table.csv", index=False)

    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))



    sp_all, ml_all = np.abs(df['spice_leakage'].values), df['ml_leakage'].values

    rel_all = np.abs(ml_all - sp_all) / sp_all * 100.0

    dr = np.abs(df['ml_leakage_direct'].values - sp_all) / sp_all * 100.0

    print(f"\n[ablation] naive direct sum (no KCL step): median error {np.median(dr):.1f}% | p95 {np.percentile(dr, 95):.1f}%")

    print(f"\nOverall: median error {np.median(rel_all):.2f}% | p95 {np.percentile(rel_all, 95):.2f}% "

          f"| mean accuracy {100 - np.mean(rel_all):.2f}% | negative predictions: {(ml_all < 0).sum()}")



    # ---- Deliverable 7: parity plot ----

    fig, ax = plt.subplots(figsize=(7, 7))

    ml_plot = np.maximum(np.abs(ml_all), 1e-14)

    ax.scatter(sp_all, ml_plot, alpha=0.35, s=18, color='darkorchid', label='18 gate cases x 200 corners')

    lo, hi = min(sp_all.min(), ml_plot.min()), max(sp_all.max(), ml_plot.max())

    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Ideal 1:1')

    ax.set_xscale('log'); ax.set_yscale('log')

    ax.set_xlabel("SPICE gate leakage (A)"); ax.set_ylabel("ML gate leakage (A)")

    ax.set_title(f"Gate-level leakage parity\nmedian error {np.median(rel_all):.2f}%, p95 {np.percentile(rel_all, 95):.2f}%")

    ax.legend()

    plt.tight_layout()

    plt.savefig(gate_res_dir / "gate_parity_plot.png", dpi=300)

    plt.close()

    print(f"[+] Saved {gate_res_dir / 'gate_parity_plot.png'}")





if __name__ == "__main__":

    main()
