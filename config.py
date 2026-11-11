"""
Central Configuration for ML-Driven PVT Variation Aware Leakage Estimation
Assignment: 32nm_HP, VDD = 1.0V, Test Circuit: 3-bit Carry Lookahead Adder (CLA)
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
NETLISTS_DIR = PROJECT_ROOT / "netlists"
DATA_DIR = PROJECT_ROOT / "data"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_FILE = MODELS_DIR / "32nm_HP.pm"

# Technology and Geometry Specifications
TECH_NAME = "32nm_HP"
L_MIN = 32e-9                 # 32nm feature size
W_MIN = 1.0 * L_MIN           # 32nm
W_MAX = 8.0 * L_MIN           # 256nm

# Nominal Voltage and Range
VDD_NOM = 1.0                 # Assigned nominal VDD
VDD_VARIATION = 0.10          # +/- 10%
VDD_MIN = VDD_NOM * (1 - VDD_VARIATION)  # 0.90 V
VDD_MAX = VDD_NOM * (1 + VDD_VARIATION)  # 1.10 V

# Temperature Range
TEMP_MIN = -40.0              # -40 deg C
TEMP_MAX = 125.0              # 125 deg C

# Sample Counts & Random Seed
NUM_TRANSISTOR_SAMPLES = 10000
NUM_GATE_CORNERS = 200
RANDOM_SEED = 42

# BSIM4 Process Parameters Nominal Values (Extracted from 32nm_HP.pm)
NOMINAL_PROCESS_PARAMS = {
    'nmos': {
        'vth0': 0.49396,
        'toxe': 1.15e-09,
        'ndep': 4.12e+18,
        'u0':   0.05,
        'xj':   1.0e-08,
    },
    'pmos': {
        'vth0': -0.49155,
        'toxe': 1.20e-09,
        'ndep': 3.07e+18,
        'u0':   0.014,
        'xj':   1.0e-08,
    }
}

PROCESS_VARIATION = 0.10  # +/- 10% uniform variation

# SPICE Simulation Options required by the project specification
SPICE_OPTIONS = "gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"

# Standard Cases
CASES = ["NMOS_ON", "NMOS_OFF", "PMOS_ON", "PMOS_OFF"]

# Standard Gates for Part B
GATES = ["INV", "NAND2", "NOR2", "AND2", "OR2"]
