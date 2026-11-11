# ML-Driven PVT Variation Aware Leakage Estimation
**Course Project — Digital VLSI Design (DVD)**

---

## 1. Project Overview & Objectives

### What Does the Project Expect Us to Do?
In nanoscale CMOS technologies (such as 32nm High-Performance bulk CMOS), leakage current is a dominant component of power dissipation and varies exponentially with **Process ($P$)**, **Voltage ($V$)**, and **Temperature ($T$)** variations. Full transistor-level SPICE simulations across large multi-gate circuits and thousands of PVT corners are computationally prohibitive.

This project implements a **hierarchical Machine Learning framework** to predict circuit leakage currents instantaneously without full SPICE simulations:
1. **Transistor Level**: Train 4 specialized ML models on single NMOS and PMOS transistors (ON and OFF states) under comprehensive PVT sweeps to predict terminal currents ($I_d, I_g, I_s, I_b$).
2. **Gate Level**: Predict leakage currents for 5 standard static CMOS gates (`INV`, `NAND2`, `NOR2`, `AND2`, `OR2`) across all 18 input states and 200 random PVT corners using internal node voltages and the trained transistor ML models.
3. **Circuit Level**: Validate the framework on a larger assigned circuit (**3-bit Carry Lookahead Adder**) by composing gate-level ML models hierarchically and comparing results against a complete transistor-level SPICE netlist.
4. **Target**: Achieve up to **99% mean prediction accuracy** against SPICE.

---

## 2. Team Assignment Specifications

| Parameter | Assigned Value | Notes |
| :--- | :--- | :--- |
| **Technology Node** | **32nm HP** (`32nm_HP.pm`) | High-Performance Metal Gate / High-K / Strained-Si |
| **Nominal Supply Voltage ($V_{DD}$)** | **1.0 V** | Sweep range: $0.90\text{ V}$ to $1.10\text{ V}$ ($\pm 10\%$) |
| **Feature Size / Min Length ($L_{min}$)** | **32 nm** ($32\times 10^{-9}\text{ m}$) | Fixed for all transistors |
| **Transistor Width ($W$)** | **$1\times L_{min}$ to $8\times L_{min}$** | $32\text{ nm}$ to $256\text{ nm}$ |
| **Temperature ($T$)** | **$-40^\circ\text{C}$ to $+125^\circ\text{C}$** | Uniform distribution |
| **Test Circuit** | **3-bit Carry Lookahead Adder (CLA)** | 7 inputs ($A_0-A_2, B_0-B_2, C_0$), 128 input vectors |

---

## 3. What Are We Doing Exactly? (Technical Methodology)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL MODELING PIPELINE                       │
└────────────────────────────────────────────────────────────────────────┘

 [1] Single-FET PVT Dataset        [2] ML Model Training (Part A)
     • NMOS ON  (10,000 samples)       • 4 architectures tested:
     • NMOS OFF (10,000 samples)         (MLP, XGBoost, LightGBM, Random Forest)
     • PMOS ON  (10,000 samples)       • Predict log10(|I|) with sign preservation
     • PMOS OFF (10,000 samples)       • 70/15/15 Train/Val/Test split
                 │                                    │
                 └──────────────────┬─────────────────┘
                                    ▼
 [3] Gate-Level Validation (Part B)
     • 5 Gates: INV, NAND2, NOR2, AND2, OR2 (18 input states x 200 corners)
     • Extract SPICE internal node voltages (x for NAND2, y for NOR2)
     • Query transistor models to compute total gate rail leakage
     • Compare ML vs. SPICE (Table & Parity Plot)
                                    │
                                    ▼
 [4] Test Circuit: 3-Bit Carry Lookahead Adder (Part C)
     • Decompose CLA into standard gates (AND2, OR2, NAND2, XOR2)
     • Evaluate logic values at all gate inputs for all 128 input vectors
     • Sum constituent gate ML leakages hierarchically
     • Validate against full SPICE netlist across PVT corners (Deliverable 8)
```

---

## 4. Key Fixes & Corrections Made Over Previous Work

The previous preliminary scripts in `samples/` had several critical errors that would have caused the project to fail evaluation:
1. **Fatal BSIM4 Injection Bug Fixed**: Previously, `vth0`, `toxe`, `ndep`, `u0`, `xj` were generated as random numbers but **never inserted into the SPICE netlist**. All simulations previously ran with identical static parameters. In this codebase, `src/spice_utils.py` uses dynamic regex injection to feed perturbed parameters directly into NGSPICE per sample.
2. **Corrected Technology Parameters**:
   - $V_{DD}$ corrected from `0.9 V` to **`1.0 V`**.
   - Channel length $L_{min}$ corrected from `16 nm` to **`32 nm`**.
   - Width sweep $W$ corrected from `16–128 nm` to **`32–256 nm`**.
   - Model nominals aligned with [models/32nm_HP.pm](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/models/32nm_HP.pm).
3. **Column Spelling Fixed**: Fixed column name `'vtho'` (letter o) to `'vth0'` (number 0).
4. **Multiprocessing Engine**: Replaced slow single-threaded sequential runs with parallel worker pools using `multiprocessing.Pool`, reducing runtime by over 10x.

---

## 5. 4-Member Team Work Allocation

To complete the project smoothly, tasks and deliverables are divided across all 4 team members:

### 👤 Member 1: Data Pipeline & SPICE Lead
* **Primary Responsibilities**:
  1. Set up NGSPICE on the team machines and verify batch execution.
  2. Run `src/01_generate_transistor_data.py` to generate the 4 datasets (10,000 samples each).
  3. Run `src/02_data_validation_check.py` to perform the Kirchhoff Current Law (KCL) check ($I_d + I_g + I_s + I_b \approx 0$).
  4. Generate and analyze the 3 physical trend plots (OFF current vs. $T$, OFF current vs. $V_{th0}$, Gate current vs. $t_{oxe}$).
* **Deliverables Owned**:
  - **Deliverable 1**: 4 CSV files (`data/NMOS_ON.csv`, `data/NMOS_OFF.csv`, `data/PMOS_ON.csv`, `data/PMOS_OFF.csv`).
  - **Deliverable 2 (Transistors)**: Single-FET netlists in `netlists/transistor/`.
  - **Deliverable 3**: KCL summary table and 3 trend plots in `results/data_check/`.
  - Report Section: *Data Generation & Data Checks*.

---

### 👤 Member 2: Machine Learning Lead
* **Primary Responsibilities**:
  1. Preprocess data (feature scaling and $\log_{10}(|I|)$ target transformation).
  2. Implement, train, and benchmark the 4 candidate model families (MLP, XGBoost, LightGBM, Random Forest) in `src/03_train_transistor_models.py`.
  3. Evaluate validation set metrics ($R^2$, median relative error %, 95th-percentile error %).
  4. Save the 4 best-performing model weights (`saved_models/`) and evaluate on the unseen test set.
  5. Produce predicted-vs-SPICE parity plots for each transistor case.
* **Deliverables Owned**:
  - **Deliverable 4**: Model Comparison Table (`results/model_comparison/model_comparison_table.csv`).
  - **Deliverable 5**: Test-set metrics, 4 saved model files (`saved_models/*.joblib`), and parity plots.
  - Report Section: *Machine Learning Architectures, Hyperparameters & Model Selection*.

---

### 👤 Member 3: Standard Gate Validation Lead (Part B)
* **Primary Responsibilities**:
  1. Inspect and verify SPICE netlists for the 5 static CMOS gates (`INV`, `NAND2`, `NOR2`, `AND2`, `OR2`) in `netlists/gates/`.
  2. Run `src/04_simulate_gates.py` to simulate all 18 input cases across 200 PVT corners.
  3. Extract internal node voltages ($x$ for NAND2/AND2, $y$ for NOR2/OR2) and output voltages.
  4. Run `src/05_evaluate_gates_ml.py` to predict gate leakage by calling the trained transistor models with recorded node voltages.
  5. Calculate SPICE median, ML median, bias, median error, and 95th-percentile error.
* **Deliverables Owned**:
  - **Deliverable 2 (Gates)**: Gate SPICE netlists in `netlists/gates/`.
  - **Deliverable 6**: Gate Results Table (`results/gate_results/gate_results_table.csv`).
  - **Deliverable 7**: Gate Leakage Parity Plot (`results/gate_results/gate_parity_plot.png`).
  - Report Section: *Gate-Level Validation, Stack Effect Analysis, and Accuracy Ranking*.

---

### 👤 Member 4: 3-Bit CLA System & Report Lead (Part C)
* **Primary Responsibilities**:
  1. Verify the hierarchical decomposition of the 3-bit Carry Lookahead Adder netlist into standard gates (`netlists/circuits/cla_3bit_example.net`).
  2. Run `src/06_simulate_and_eval_cla.py` to simulate the full CLA netlist in SPICE across input vectors and PVT corners.
  3. Implement the hierarchical ML evaluator (evaluating boolean logic states for each gate and summing predicted gate leakages).
  4. Produce the CLA results summary table and test-circuit parity plot.
  5. Test and maintain the master reproduction script (`run_all.py`).
  6. Lead final compilation of the typed project report.
* **Deliverables Owned**:
  - **Deliverable 2 (Circuit)**: Full 3-bit CLA SPICE netlist in `netlists/circuits/`.
  - **Deliverable 8**: Test-Circuit Results Table & Parity Plot (`results/cla_results/`).
  - **Deliverable 9**: Master reproduction pipeline (`run_all.py`).
  - **Deliverable 10**: Final Comprehensive Typed Report.

---

## 6. Directory Structure

```
pvt_leakage_project/
├── README.md                      # This Master Documentation & Team Guide
├── config.py                      # Centralized configuration, nominals, and ranges
├── requirements.txt               # Python package dependencies
├── setup_env.sh                   # One-click environment setup script
├── run_all.py                     # Master execution runner (Deliverable 9)
├── models/
│   └── 32nm_HP.pm                 # Assigned PTM 32nm HP model card
├── netlists/                      # Deliverable 2 Netlists
│   ├── transistor/                # Single FET netlists (NMOS/PMOS ON/OFF)
│   ├── gates/                     # Standard gate netlists (INV, NAND2, NOR2, AND2, OR2)
│   └── circuits/                  # Full 3-bit CLA Adder netlist
├── data/                          # Deliverable 1 CSVs (10,000 samples each)
│   ├── NMOS_ON.csv
│   ├── NMOS_OFF.csv
│   ├── PMOS_ON.csv
│   └── PMOS_OFF.csv
├── saved_models/                  # Deliverable 5 Saved Models (.joblib)
├── results/
│   ├── data_check/                # Deliverable 3: KCL checks & 3 physical trend plots
│   ├── model_comparison/          # Deliverables 4 & 5: Model table & test parity plots
│   ├── gate_results/              # Deliverables 6 & 7: 18-case table & gate parity plot
│   └── cla_results/               # Deliverable 8: 3-bit CLA table & parity plot
└── src/
    ├── __init__.py
    ├── spice_utils.py             # BSIM4 parameter injector & NGSPICE runner
    ├── 01_generate_transistor_data.py
    ├── 02_data_validation_check.py
    ├── 03_train_transistor_models.py
    ├── 04_simulate_gates.py
    ├── 05_evaluate_gates_ml.py
    └── 06_simulate_and_eval_cla.py
```

---

## 7. Quickstart Guide (How to Run)

### Step 1: Install NGSPICE & Create Virtual Environment
```bash
# 1. Install NGSPICE (Arch Linux)
sudo pacman -S ngspice

# Or on Ubuntu/Debian:
# sudo apt update && sudo apt install -y ngspice

# 2. Run automated environment setup:
cd pvt_leakage_project
chmod +x setup_env.sh
./setup_env.sh

# 3. Activate environment:
source .venv/bin/activate
```

### Step 2: Run Individual Stages
```bash
# Step 1: Generate Transistor Datasets (Deliverable 1 & 2)
python src/01_generate_transistor_data.py

# Step 2: Validate Data & Generate Trend Plots (Deliverable 3)
python src/02_data_validation_check.py

# Step 3: Train & Benchmark 4 ML Models (Deliverables 4 & 5)
python src/03_train_transistor_models.py

# Step 4: SPICE Simulation of 18 Gate Cases (Part B)
python src/04_simulate_gates.py

# Step 5: Gate ML Evaluation & Parity Plots (Deliverables 6 & 7)
python src/05_evaluate_gates_ml.py

# Step 6: 3-Bit CLA Simulation & Validation (Deliverable 8)
python src/06_simulate_and_eval_cla.py
```

### Step 3: End-to-End One-Click Execution (Deliverable 9)
```bash
python run_all.py
```

---

## 8. Deliverables Mapping Checklist

| Deliverable | Description | File Location in Project | Responsible |
| :--- | :--- | :--- | :--- |
| **1. Datasets** | 4 CSVs (10k rows each, 16 columns) | `data/` | Member 1 |
| **2. Example Netlists** | Single-FET, 5 Gates, 3-bit CLA | `netlists/transistor/`, `gates/`, `circuits/` | Members 1, 3, 4 |
| **3. Data Check** | KCL verification & 3 trend plots | `results/data_check/` | Member 1 |
| **4. Model Comparison** | Table of 4 models x 4 cases ($R^2$, med err, 95th% err) | `results/model_comparison/model_comparison_table.csv` | Member 2 |
| **5. Chosen Models** | Test set metrics, parity plots, 4 saved models | `saved_models/`, `results/model_comparison/` | Member 2 |
| **6. Gate Results Table** | 18 gate cases: SPICE med, ML med, bias, errs | `results/gate_results/gate_results_table.csv` | Member 3 |
| **7. Gate Parity Plot** | ML vs SPICE leakage (log-log) | `results/gate_results/gate_parity_plot.png` | Member 3 |
| **8. Test Circuit Results**| 3-bit CLA metrics & parity plot | `results/cla_results/` | Member 4 |
| **9. Code Pipeline** | Reproducible scripts with fixed seeds | `run_all.py`, `src/` | Member 4 / All |
| **10. Report** | Comprehensive typed technical report | Report document (PDF) | All Members |
