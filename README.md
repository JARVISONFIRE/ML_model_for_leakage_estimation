# ML-Driven PVT Variation-Aware Leakage Estimation in 32nm CMOS

An end-to-end hierarchical machine learning framework for instantaneous, variation-aware leakage power estimation in nanoscale digital CMOS circuits under Process, Voltage, and Temperature (PVT) variations.

---

## 1. Executive Summary

In nanoscale CMOS technologies, leakage current is a primary contributor to total chip power dissipation and exhibits extreme exponential sensitivity to Process ($P$), Voltage ($V$), and Temperature ($T$) variations. Performing exhaustive transistor-level SPICE Monte Carlo simulations across large multi-gate netlists and thousands of PVT corners is computationally prohibitive for VLSI design exploration and sign-off.

This repository provides an automated, hierarchical machine learning modeling methodology that replaces brute-force SPICE runs with instantaneous inference:

1. **Transistor Level**: Physics-informed neural regressors trained on 4 single-transistor operating states (`NMOS_ON`, `NMOS_OFF`, `PMOS_ON`, `PMOS_OFF`) predict all four terminal currents ($I_d, I_g, I_s, I_b$) with sign preservation across continuous operating voltages and BSIM4 process parameters.
2. **Gate Level**: Static CMOS logic gates (`INV`, `NAND2`, `NOR2`, `AND2`, `OR2`) are evaluated across all 18 input states by solving Kirchhoff's Current Law (KCL) at internal floating nodes and querying the transistor models.
3. **Circuit Level**: Full hierarchical circuit validation on a **3-Bit Carry Lookahead Adder (CLA)** across 128 logic vectors $\times$ 200 random PVT corners (25,600 simulations), achieving **99.19% mean prediction accuracy** against complete SPICE netlist simulations.

---

## 2. Benchmark Results

### A. Circuit-Level Validation (3-Bit Carry Lookahead Adder — 25,600 Runs)
Validated against complete SPICE netlist simulations across 128 input patterns and 200 random PVT corners:

| Circuit | SPICE Median Current | ML Median Current | Bias | Median Error | Mean Error | Mean Accuracy | 95th% Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-Bit CLA Adder** | $5.438 \times 10^{-7}\text{ A}$ | $5.400 \times 10^{-7}\text{ A}$ | $-1.464\text{ nA}$ | **0.31%** | **0.81%** | **99.19%** | **3.58%** |

*Output files: [cla_results_summary.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/cla_results/cla_results_summary.csv) | Parity plot: [cla_parity_plot.png](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/cla_results/cla_parity_plot.png)*

### B. Gate-Level Performance (18 Input States $\times$ 200 PVT Corners = 3,600 Runs)
Supply current ($I_{VDD}$) prediction accuracy across 5 standard static CMOS cells:

| Gate Type | Input State | SPICE Median | ML Median | Median Error | Mean Error | Mean Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **INV** | `0` | $1.378 \times 10^{-8}\text{ A}$ | $1.378 \times 10^{-8}\text{ A}$ | 0.0012% | 0.0016% | **99.9984%** |
| **INV** | `1` | $6.544 \times 10^{-9}\text{ A}$ | $6.544 \times 10^{-9}\text{ A}$ | 0.0023% | 0.0026% | **99.9974%** |
| **NAND2** | `00` | $4.372 \times 10^{-10}\text{ A}$ | $4.372 \times 10^{-10}\text{ A}$ | 0.0022% | 0.0038% | **99.9962%** |
| **NAND2** | `01` | $1.376 \times 10^{-8}\text{ A}$ | $1.376 \times 10^{-8}\text{ A}$ | 0.0012% | 0.0016% | **99.9984%** |
| **NAND2** | `10` | $4.429 \times 10^{-9}\text{ A}$ | $4.429 \times 10^{-9}\text{ A}$ | 0.0010% | 0.0013% | **99.9987%** |
| **NAND2** | `11` | $1.308 \times 10^{-8}\text{ A}$ | $1.308 \times 10^{-8}\text{ A}$ | 0.0022% | 0.0026% | **99.9974%** |
| **NOR2** | `00` | $2.751 \times 10^{-8}\text{ A}$ | $2.751 \times 10^{-8}\text{ A}$ | 0.0012% | 0.0016% | **99.9984%** |
| **NOR2** | `01` | $6.552 \times 10^{-9}\text{ A}$ | $6.552 \times 10^{-9}\text{ A}$ | 0.0022% | 0.0026% | **99.9974%** |
| **NOR2** | `10` | $1.017 \times 10^{-9}\text{ A}$ | $1.017 \times 10^{-9}\text{ A}$ | 0.0013% | 0.0017% | **99.9983%** |
| **NOR2** | `11` | $2.411 \times 10^{-11}\text{ A}$ | $2.411 \times 10^{-11}\text{ A}$ | 0.0011% | 0.0013% | **99.9987%** |
| **AND2** | `00`, `01`, `10`, `11` | $7.39\text{ nA}$ – $30.82\text{ nA}$ | $7.39\text{ nA}$ – $30.81\text{ nA}$ | 0.0014% – 0.0019% | 0.0020% – 0.0024% | **99.9978%** |
| **OR2** | `00`, `01`, `10`, `11` | $13.82\text{ nA}$ – $37.52\text{ nA}$ | $13.82\text{ nA}$ – $37.52\text{ nA}$ | 0.0012% – 0.0015% | 0.0016% – 0.0021% | **99.9981%** |
| **Overall** | **All 18 Gates** | — | — | **0.0014%** | **0.0020%** | **99.9980%** |

*Output files: [gate_results_table.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/gate_results/gate_results_table.csv) | Parity plot: [gate_parity_plot.png](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/gate_results/gate_parity_plot.png)*

### C. Transistor Test-Set Evaluation (Unseen 15% Hold-Out Set)
Evaluation across all 4 terminal currents ($I_d, I_g, I_s, I_b$):

| Operating Mode | Best Model | Test $R^2_{\log}$ | Median Error | Mean Error | Mean Accuracy | 95th% Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NMOS_ON** | PyTorch L-BFGS MLP | **0.9997** | **1.45%** | **2.40%** | **97.60%** | 6.62% |
| **NMOS_OFF** | PyTorch L-BFGS MLP | **0.9982** | **0.91%** | **4.10%** | **95.90%** | 5.54% |
| **PMOS_ON** | PyTorch L-BFGS MLP | **0.9998** | **1.17%** | **2.48%** | **97.52%** | 4.79% |
| **PMOS_OFF** | PyTorch L-BFGS MLP | **0.9890** | **2.90%** | **5.40%** | **94.60%** | 12.79% |

*Output files: [chosen_models_test_metrics.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/model_comparison/chosen_models_test_metrics.csv) | Model comparison: [model_comparison_table.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/model_comparison/model_comparison_table.csv)*

---

## 3. Technology & PVT Variation Specifications

The framework is calibrated to the **32nm High-Performance (HP) bulk CMOS** Predictive Technology Model (PTM):

| Parameter | Symbol | Nominal Value | Variation Distribution | Physical Range |
| :--- | :---: | :---: | :---: | :---: |
| **Supply Voltage** | $V_{DD}$ | $1.00\text{ V}$ | Uniform $\pm 10\%$ | $0.90\text{ V}$ to $1.10\text{ V}$ |
| **Temperature** | $T$ | $27^\circ\text{C}$ ($300.15\text{ K}$) | Uniform range | $-40^\circ\text{C}$ to $+125^\circ\text{C}$ |
| **Channel Length** | $L$ | $32\text{ nm}$ | Fixed feature size | $32\text{ nm}$ |
| **Channel Width** | $W$ | $1\times$ to $8\times L_{min}$ | Continuous sweep | $32\text{ nm}$ to $256\text{ nm}$ |
| **Threshold Voltage** | $V_{th0}$ | $0.494\text{ V}$ (N) / $-0.492\text{ V}$ (P) | Uniform $\pm 10\%$ | BSIM4 process corner |
| **Gate Oxide Thickness** | $t_{oxe}$ | $1.15\text{ nm}$ (N) / $1.20\text{ nm}$ (P) | Uniform $\pm 10\%$ | Physical dielectric sweep |
| **Channel Doping** | $N_{dep}$ | $4.12\times 10^{18}\text{ cm}^{-3}$ (N) | Uniform $\pm 10\%$ | Active body doping |
| **Low-Field Mobility** | $\mu_0$ | $0.050\text{ m}^2/\text{V}\cdot\text{s}$ (N) | Uniform $\pm 10\%$ | Carrier transport |
| **Junction Depth** | $X_j$ | $10.0\text{ nm}$ | Uniform $\pm 10\%$ | S/D metallurgical junction |

---

## 4. Methodological Architecture

```
                       PVT Parameters + Device Geometry
                        (VDD, T, W, Vth0, Tox, Ndep, U0, Xj)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Transistor Level: Physics-Informed Neural Network                        │
│    • Domain Invariants: Thermal voltage vt = kB*T/q, Vov/vt, Vds/vt, Vbs/vt │
│    • Subthreshold diffusion factor: ln(1 - exp(-|Vds|/vt))                  │
│    • PyTorch GPU L-BFGS MLP (128 -> 128 -> 64 -> 32 -> 1) with SiLU         │
│    • Duan's median smearing factor to remove log-transform Jensen bias      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Gate Level: KCL Node Solving & Supply Current Refinement                 │
│    • Solves Kirchhoff's Current Law for internal floating nodes:            │
│      Sum of terminal currents at internal stacks = 0                        │
│    • Preserves supply rail current conservation across all 18 CMOS states   │
│    • RBF KernelRidge refinement on supply current log-ratio                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Circuit Level: Hierarchical CLA Composition & Verification               │
│    • Structural netlist decomposition: 3-Bit Carry Lookahead Adder          │
│    • Boolean state propagation across all 128 input vectors                 │
│    • Constituent gate ML leakage aggregation with multi-gate calibration    │
│    • Direct validation against full-circuit SPICE netlist simulation        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Engineering Features
1. **BSIM4 Physics Feature Engineering**: Incorporates thermal voltage $v_t = \frac{k_B T_K}{q}$, dimensionless overdrive and barrier ratios ($\frac{V_{ov}}{v_t}, \frac{V_{ds}}{v_t}, \frac{V_{bs}}{v_t}$), square-root body effect depletion depth $\sqrt{\max(0.01, \phi - V_{bs})} - \sqrt{\phi}$, and subthreshold diffusion terms into the input tensor.
2. **PyTorch L-BFGS Optimization**: Utilizes second-order quasi-Newton optimization (L-BFGS) to calculate curvature information directly, resolving the steep multi-decade exponential subthreshold gradients without Adam oscillation.
3. **Duan's Smearing Calibration**: Eliminates the systematic downward bias introduced by Jensen's inequality when transforming log-domain predictions back to linear current:
   $$c = \text{median}\left(\frac{|y_{\text{true}}|}{10^{\hat{y}_{\log}}}\right)$$
4. **SPICE Simulator Noise Filtering**: Accurately differentiates physical direct-tunneling current ($\ge 1\text{ fA}$) from SPICE simulator numerical noise floor artifacts ($< 1\text{ fA}$ at `abstol = 1e-18`), preventing zero-clipping distortion.

---

## 5. Repository Structure

```
pvt_leakage_project/
├── README.md                          # Master technical report and system documentation
├── config.py                          # Global configuration, technological constants, PVT ranges
├── requirements.txt                   # Python package dependencies
├── setup_env.sh                       # Automated environment configuration script
├── run_all.py                         # Master reproducible execution pipeline
├── verify_results.py                  # Automated verification audit script
├── models/
│   └── 32nm_HP.pm                     # Assigned BSIM4 32nm High-Performance model card
├── netlists/
│   ├── transistor/                    # Standalone FET netlists (NMOS/PMOS ON/OFF)
│   ├── gates/                         # Standard CMOS gates (INV, NAND2, NOR2, AND2, OR2)
│   └── circuits/                      # Complete 3-bit Carry Lookahead Adder netlist
├── data/                              # 40,000 SPICE-generated transistor training samples
│   ├── NMOS_ON.csv                    # 10,000 samples, 16 columns
│   ├── NMOS_OFF.csv                   # 10,000 samples, 16 columns
│   ├── PMOS_ON.csv                    # 10,000 samples, 16 columns
│   └── PMOS_OFF.csv                   # 10,000 samples, 16 columns
├── saved_models/                      # Trained model weights (.joblib)
│   ├── NMOS_ON_best_model.joblib
│   ├── NMOS_OFF_best_model.joblib
│   ├── PMOS_ON_best_model.joblib
│   └── PMOS_OFF_best_model.joblib
├── results/
│   ├── data_check/                    # KCL validation summary and 3 physical trend plots
│   ├── model_comparison/              # Transistor model benchmarking, metrics, parity plots
│   ├── gate_results/                  # 18-case CMOS gate results table and parity plot
│   └── cla_results/                   # 3-bit CLA test circuit summary and parity plot
└── src/
    ├── __init__.py
    ├── spice_utils.py                 # Dynamic BSIM4 parameter injector & NGSPICE harness
    ├── model_wrapper.py               # Physics feature engineering & PyTorch L-BFGS model
    ├── 01_generate_transistor_data.py # Multiprocessing SPICE generator for single-FET data
    ├── 02_data_validation_check.py    # KCL verification & physical trend plot generation
    ├── 03_train_transistor_models.py  # Model benchmarking, selection, and parity plotting
    ├── 04_simulate_gates.py           # Multi-corner SPICE simulation for standard gates
    ├── 05_evaluate_gates_ml.py        # Gate-level KCL solving & ML evaluation
    └── 06_simulate_and_eval_cla.py    # Hierarchical 3-bit CLA simulation & validation
```

---

## 6. Installation & Reproduction

### Prerequisites
- Linux OS (Ubuntu 20.04+, Debian 11+, Arch Linux)
- Python 3.10+ (with PyTorch, CUDA optional)
- NGSPICE (v30 or newer)

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ngspice

# Arch Linux
sudo pacman -S ngspice
```

### Environment Setup
```bash
git clone https://github.com/harryjain/pvt_leakage_project.git
cd pvt_leakage_project

# Automatic virtual environment setup
chmod +x setup_env.sh
./setup_env.sh

# Activate virtual environment
source .venv/bin/activate
```

### End-to-End Pipeline Execution
To execute all stages sequentially from single-transistor generation to full 3-bit CLA circuit validation:
```bash
python run_all.py
```

To run from a specific step (e.g., skip data generation and run model training through CLA evaluation):
```bash
python run_all.py --from 3
```

### Automated Verification Audit
To verify all artifacts, deliverable tables, metrics, and parity plots:
```bash
python verify_results.py
```

Expected verification output:
```
======================================================================
   PVT VARIATION-AWARE LEAKAGE ESTIMATION — DELIVERABLES AUDIT
======================================================================
[1] Checking Transistor Datasets:        [✓] 4/4 CSVs Valid (10,000 samples each)
[2] Checking Example SPICE Netlists:     [✓] 6/6 Netlists Verified
[3] Checking Physics & KCL Verification: [✓] KCL Residuals < 2e-11 A, 3 Plots OK
[4] Checking Model Comparison & Weights: [✓] 4 Models Selected, Joblibs Saved
[5] Checking Gate-Level Results:         [✓] 18 Cases: Median Err 0.0014%, Mean Acc 99.9980%
[6] Checking 3-Bit CLA Adder Results:    [✓] Median Err 0.31%, Mean Acc 99.19%
======================================================================
 [✓] ALL DELIVERABLES 1 THROUGH 8 ARE VERIFIED AND WORKING PROPERLY!
======================================================================
```

---

## 7. Deliverables Index

| Deliverable | Description | File Path |
| :--- | :--- | :--- |
| **Deliverable 1** | Transistor Datasets (4 CSVs $\times$ 10,000 rows) | [data/](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/data/) |
| **Deliverable 2** | SPICE Netlists (Transistors, Gates, CLA) | [netlists/](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/netlists/) |
| **Deliverable 3** | KCL Verification Table & 3 Trend Plots | [results/data_check/](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/data_check/) |
| **Deliverable 4** | Model Comparison Benchmarking Table | [model_comparison_table.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/model_comparison/model_comparison_table.csv) |
| **Deliverable 5** | Chosen Model Test Metrics & Saved Weights | [chosen_models_test_metrics.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/model_comparison/chosen_models_test_metrics.csv), [saved_models/](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/saved_models/) |
| **Deliverable 6** | Gate Results Summary Table (18 Cases) | [gate_results_table.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/gate_results/gate_results_table.csv) |
| **Deliverable 7** | Gate Leakage Parity Plot | [gate_parity_plot.png](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/gate_results/gate_parity_plot.png) |
| **Deliverable 8** | 3-Bit CLA Results Table & Parity Plot | [cla_results_summary.csv](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/cla_results/cla_results_summary.csv), [cla_parity_plot.png](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/results/cla_results/cla_parity_plot.png) |
| **Deliverable 9** | Master Pipeline Execution Script | [run_all.py](file:///home/harry/ARCHY/home_harry_complete/3-1/DVD/ML-Driven-PVT-Variation-Aware-Leakage-Estimation/pvt_leakage_project/run_all.py) |
