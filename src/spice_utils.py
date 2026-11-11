"""
SPICE Utilities for NGSPICE Batch Simulation and BSIM4 Process Parameter Injection
"""
import re
import subprocess
from pathlib import Path
import numpy as np

from config import MODEL_FILE, NOMINAL_PROCESS_PARAMS, SPICE_OPTIONS, L_MIN

# Cache base model card text
with open(MODEL_FILE, "r") as f:
    _BASE_MODEL_TEXT = f.read()

# Separate NMOS and PMOS base cards
_PARTS = _BASE_MODEL_TEXT.split(".model  pmos")
_NMOS_BASE = _PARTS[0]
_PMOS_BASE = ".model  pmos" + _PARTS[1]


def build_custom_model_card(mos_type: str, vth0: float, toxe: float, ndep: float, u0: float, xj: float, model_name: str = None) -> str:
    """
    Dynamically injects perturbed process parameters into the BSIM4 model card.
    mos_type: 'nmos' or 'pmos'
    """
    base = _NMOS_BASE if mos_type.lower() == "nmos" else _PMOS_BASE
    target_name = model_name if model_name else f"{mos_type}_custom"

    # Replace model name
    card = re.sub(rf"\.model\s+{mos_type}\s+{mos_type}", f".model  {target_name}  {mos_type}", base, count=1)

    params = {
        'vth0': f"{vth0:.6e}",
        'toxe': f"{toxe:.6e}",
        'ndep': f"{ndep:.6e}",
        'u0':   f"{u0:.6e}",
        'xj':   f"{xj:.6e}",
    }

    for k, v in params.items():
        pattern = rf"(\b{k}\s*=\s*)[^\s]+"
        card, count = re.subn(pattern, rf"\g<1>{v}", card)
        if count != 1:
            raise ValueError(f"Failed to replace BSIM4 parameter {k} in {mos_type} model card (matches: {count}).")

    return card


def run_single_transistor_sim(case: str, row: dict) -> dict:
    """
    Simulates a single transistor in NGSPICE with exact PVT parameters.
    Returns terminal currents: id, ig, is, ib.
    """
    is_nmos = "NMOS" in case
    is_on = "ON" in case
    fet_type = "nmos" if is_nmos else "pmos"
    model_name = f"{fet_type}_mod"

    vdd = float(row['vdd'])
    vd = float(row['vd'])
    vs = float(row['vs'])
    w = float(row['w'])
    temp = float(row['temp'])

    # Fixed gate and body voltages per project specifications (Section 4)
    # NMOS ON: Vg = VDD, Vb = 0
    # NMOS OFF: Vg = 0,   Vb = 0
    # PMOS ON: Vg = 0,   Vb = VDD
    # PMOS OFF: Vg = VDD, Vb = VDD
    if is_nmos:
        vg = vdd if is_on else 0.0
        vb = 0.0
    else:
        vg = 0.0 if is_on else vdd
        vb = vdd

    # Process parameters
    vth0 = float(row['vth0'])
    toxe = float(row['toxe'])
    ndep = float(row['ndep'])
    u0   = float(row['u0'])
    xj   = float(row['xj'])

    custom_model = build_custom_model_card(fet_type, vth0, toxe, ndep, u0, xj, model_name=model_name)

    netlist = f"""* Single Transistor PVT Test - {case}
.options {SPICE_OPTIONS} temp={temp}

{custom_model}

Vd d 0 DC {vd}
Vg g 0 DC {vg}
Vs s 0 DC {vs}
Vb b 0 DC {vb}

M1 d g s b {model_name} W={w} L={L_MIN}

.control
op
print i(Vd) i(Vg) i(Vs) i(Vb)
.endc
.end
"""

    res = subprocess.run(
        ["ngspice", "-b"],
        input=netlist,
        capture_output=True,
        text=True
    )

    output = res.stdout

    # Parse currents directly from voltage sources
    # In SPICE, current leaving positive pin of Vsrc entering terminal is -i(Vsrc)
    # Terminal currents defined entering device: Id = -i(Vd), Ig = -i(Vg), Is = -i(Vs), Ib = -i(Vb)
    m_d = re.search(r"i\(vd\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+)", output, re.IGNORECASE)
    m_g = re.search(r"i\(vg\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+)", output, re.IGNORECASE)
    m_s = re.search(r"i\(vs\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+)", output, re.IGNORECASE)
    m_b = re.search(r"i\(vb\)\s*=\s*([-+]?\d+(?:\.\d+)?[eE][-+]?\d+)", output, re.IGNORECASE)

    if m_d and m_g and m_s and m_b:
        id_ = -float(m_d.group(1))
        ig  = -float(m_g.group(1))
        is_ = -float(m_s.group(1))
        ib  = -float(m_b.group(1))
    else:
        # Fallback to numerical extractor if output format differs slightly
        nums = re.findall(r"=\s*([-\+]?\d+(?:\.\d+)?[eE][-\+]?\d+)", output)
        if len(nums) >= 4:
            id_, ig, is_, ib = [-float(x) for x in nums[:4]]
        else:
            id_, ig, is_, ib = np.nan, np.nan, np.nan, np.nan

    return {
        'vdd': vdd, 'temp': temp, 'w': w,
        'vd': vd, 'vs': vs, 'vg': vg, 'vb': vb,
        'vth0': vth0, 'toxe': toxe, 'ndep': ndep, 'u0': u0, 'xj': xj,
        'id': id_, 'ig': ig, 'is': is_, 'ib': ib
    }


def draw_gate_corners(n_corners: int = 200, seed: int = 42) -> list:
    """Draws random PVT corners for logic gates and circuits."""
    np.random.seed(seed)
    from config import VDD_MIN, VDD_MAX, TEMP_MIN, TEMP_MAX, W_MIN, W_MAX, NOMINAL_PROCESS_PARAMS, PROCESS_VARIATION

    vdd = np.random.uniform(VDD_MIN, VDD_MAX, n_corners)
    temp = np.random.uniform(TEMP_MIN, TEMP_MAX, n_corners)
    wn = np.random.uniform(W_MIN, W_MAX, n_corners)
    wp = np.random.uniform(W_MIN, W_MAX, n_corners)

    n_nom = NOMINAL_PROCESS_PARAMS['nmos']
    p_nom = NOMINAL_PROCESS_PARAMS['pmos']

    corners = []
    for i in range(n_corners):
        c = {
            'corner_id': i,
            'vdd': vdd[i],
            'temp': temp[i],
            'wn': wn[i],
            'wp': wp[i],
        }
        for p in ['vth0', 'toxe', 'ndep', 'u0', 'xj']:
            c[f'{p}_n'] = np.random.uniform(n_nom[p]*(1-PROCESS_VARIATION), n_nom[p]*(1+PROCESS_VARIATION))
            c[f'{p}_p'] = np.random.uniform(p_nom[p]*(1-PROCESS_VARIATION), p_nom[p]*(1+PROCESS_VARIATION))
        corners.append(c)
    return corners

