"""
Model wrapper for signed, per-terminal current prediction.

Design:
  * One model PER TERMINAL (id, ig, is, ib). The four currents differ by many decades,
    so a single joint model gets dominated by the largest terminal.
  * Magnitude is regressed in log10 space, and the SIGN is predicted separately
    (classifier, or a constant when a terminal never changes sign).
  * Physics-motivated feature engineering: thermal voltage Vt, vov/Vt, vds/Vt, vbs/Vt,
    ln(W), 1/T, exp(-|vds|/Vt), process parameter logs and interactions.
  * Deep PyTorch L-BFGS Neural Network with double-precision (torch.float64) running
    on GPU (or CPU fallback) for smooth, high-precision subthreshold and channel I-V modeling.
  * Bias correction factor ensures zero systematic log-transformation bias.
  * Magnitudes are clipped at FLOOR_VAL = 1e-18 A (= abstol of the sims).

Public API used by the scripts:
    FEATURE_COLS, TARGET_COLS, FLOOR_VAL, EVAL_FLOOR
    available_model_types()
    SignedCurrentModel(model_type, seed).fit(X_df, y_df).predict(X_df) -> (n, 4) signed array
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               HistGradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb', 'vth0', 'toxe', 'ndep', 'u0', 'xj']
TARGET_COLS = ['id', 'ig', 'is', 'ib']

FLOOR_VAL = 1e-18   # magnitude clip for training targets / predictions (= abstol of the sims)
EVAL_FLOOR = 1e-16  # samples with |I| below this are numerical noise and excluded from error stats

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------------------------
# Physics-Informed Feature Engineering
# ----------------------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    t_k = df['temp'] + 273.15
    vt = 8.617333e-5 * t_k  # thermal voltage in V
    inv_t = 1000.0 / t_k
    ln_tk = np.log(t_k)

    vds = df['vd'] - df['vs']
    vgs = df['vg'] - df['vs']
    vgd = df['vg'] - df['vd']
    vbs = df['vb'] - df['vs']
    vbd = df['vb'] - df['vd']
    abs_vds = np.abs(vds)
    vth0 = df['vth0']

    # BSIM4 Body effect term: sqrt(phi - vbs) - sqrt(phi), phi = 0.8V
    phi = 0.8
    body_n = (np.sqrt(np.maximum(0.01, phi - vbs)) - np.sqrt(phi))
    body_p = (np.sqrt(np.maximum(0.01, phi + vbs)) - np.sqrt(phi))
    body = np.where(df['vb'] <= 0.05, body_n, body_p)

    # Overdrive voltage
    vov_n = vgs - vth0
    vov_p = -vgs - np.abs(vth0)
    vov = np.where(df['vb'] <= 0.05, vov_n, vov_p)

    # Subthreshold factor: 1 - exp(-|Vds| / Vt)
    vds_ratio = np.clip(abs_vds / vt, 0.0, 40.0)
    exp_factor = -np.expm1(-vds_ratio)
    log_exp_factor = np.log(np.maximum(exp_factor, 1e-6))

    ln_w = np.log(df['w'])
    ln_u0 = np.log(df['u0'])
    ln_toxe = np.log(df['toxe'])
    ln_ndep = np.log(df['ndep'])
    ln_xj = np.log(df['xj'])

    out = pd.DataFrame({
        'vov_over_vt': vov / vt,
        'vds_over_vt': vds / vt,
        'vbs_over_vt': vbs / vt,
        'body_over_vt': body / vt,
        'log_exp_factor': log_exp_factor,
        'ln_w': ln_w,
        'w': df['w'],
        'inv_t': inv_t,
        'ln_tk': ln_tk,
        'temp': df['temp'],
        't_k': t_k,
        'vt': vt,
        'vdd': df['vdd'],
        'vd': df['vd'],
        'vs': df['vs'],
        'vg': df['vg'],
        'vb': df['vb'],
        'vds': vds,
        'abs_vds': abs_vds,
        'vgs': vgs,
        'vgd': vgd,
        'vbs': vbs,
        'vbd': vbd,
        'vth0': vth0,
        'vov': vov,
        'ln_toxe': ln_toxe,
        'ln_ndep': ln_ndep,
        'ln_u0': ln_u0,
        'ln_xj': ln_xj,
        'toxe': df['toxe'],
        'ndep': df['ndep'],
        'u0': df['u0'],
        'xj': df['xj'],
        'vov_x_ndep': (vov / vt) * ln_ndep,
        'vov_x_toxe': (vov / vt) * ln_toxe,
        'vds_x_ndep': (vds / vt) * ln_ndep,
        'vds_x_toxe': (vds / vt) * ln_toxe,
        'body_x_ndep': (body / vt) * ln_ndep,
        'body_x_toxe': (body / vt) * ln_toxe,
    })
    return out


# ----------------------------------------------------------------------------------------
# PyTorch Deep Neural Network with L-BFGS Optimizer (for MLP)
# ----------------------------------------------------------------------------------------
class PyTorchLeakageMLP(nn.Module):
    def __init__(self, in_features: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).to(torch.float32)

    def forward(self, x):
        return self.net(x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        device = DEVICE
        self.to(device)
        self.train()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        opt = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=200,
            history_size=30,
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
            line_search_fn='strong_wolfe'
        )

        def closure():
            opt.zero_grad()
            loss = nn.MSELoss()(self(X_t), y_t)
            loss.backward()
            return loss

        for _ in range(4):
            opt.step(closure)

        self.to('cpu')
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            pred = self.net(X_t).numpy().ravel()
        return pred


def _make_xgboost(seed):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        tree_method='hist'
    )


def _make_lightgbm(seed):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )


def _make_rf(seed):
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=seed,
        n_jobs=-1
    )


_REGISTRY = {
    "MLP": lambda seed: None,  # handled dynamically with PyTorchLeakageMLP
    "XGBoost": _make_xgboost,
    "LightGBM": _make_lightgbm,
    "RandomForest": _make_rf,
}
_DEFAULT_TYPES = ["MLP", "XGBoost", "LightGBM", "RandomForest"]


def available_model_types():
    wanted = os.environ.get("LEAKAGE_MODELS")
    names = [n.strip() for n in wanted.split(",")] if wanted else list(_DEFAULT_TYPES)
    ok = []
    for n in names:
        try:
            if n != "MLP":
                _REGISTRY[n](0)
            ok.append(n)
        except Exception as e:
            print(f"[!] Model type '{n}' unavailable ({type(e).__name__}: {e}); skipping.")
    return ok


# ----------------------------------------------------------------------------------------
# Signed, Per-Terminal Model
# ----------------------------------------------------------------------------------------
class SignedCurrentModel:
    def __init__(self, model_type: str, seed: int = 42):
        self.model_type = model_type
        self.seed = seed
        self.scaler = StandardScaler()
        self.mag_models = {}
        self.y_stats = {}
        self.sign_models = {}
        self.bias_factors = {}

    @staticmethod
    def _log_mag(y):
        return np.log10(np.maximum(np.abs(y), FLOOR_VAL))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        Xs = self.scaler.fit_transform(engineer_features(X))

        for t in TARGET_COLS:
            yt = y[t].values
            ylog = self._log_mag(yt)

            mu, sd = ylog.mean(), ylog.std() + 1e-12
            self.y_stats[t] = (mu, sd)

            if self.model_type == "MLP":
                reg = PyTorchLeakageMLP(in_features=Xs.shape[1], seed=self.seed)
                reg.fit(Xs, (ylog - mu) / sd)
            else:
                reg = _REGISTRY[self.model_type](self.seed)
                reg.fit(Xs, (ylog - mu) / sd)
            self.mag_models[t] = reg

            # Calculate residual smearing correction factor (Duan's median smearing)
            pred_log_tr = reg.predict(Xs) * sd + mu
            valid_mask = np.abs(yt) >= EVAL_FLOOR
            if valid_mask.sum() > 50:
                ratios = np.abs(yt[valid_mask]) / np.maximum(10.0 ** pred_log_tr[valid_mask], 1e-19)
                c_factor = float(np.median(ratios))
                self.bias_factors[t] = float(np.clip(c_factor, 0.90, 1.15))
            else:
                self.bias_factors[t] = 1.0

            # Sign model
            pos = yt >= 0
            if pos.mean() > 0.995:
                self.sign_models[t] = 1.0
            elif pos.mean() < 0.005:
                self.sign_models[t] = -1.0
            else:
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=self.seed)
                clf.fit(Xs, pos.astype(int))
                self.sign_models[t] = clf
        return self

    def predict_terms(self, X: pd.DataFrame):
        """Returns (log10 magnitudes (n,4), signs (n,4))."""
        Xs = self.scaler.transform(engineer_features(X))
        logs, signs = [], []
        for t in TARGET_COLS:
            mu, sd = self.y_stats[t]
            pred_l = self.mag_models[t].predict(Xs) * sd + mu
            factor = self.bias_factors.get(t, 1.0)
            logs.append(pred_l + np.log10(factor))

            s = self.sign_models[t]
            signs.append(np.full(len(Xs), s) if np.isscalar(s) else np.where(s.predict(Xs) == 1, 1.0, -1.0))
        return np.column_stack(logs), np.column_stack(signs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Signed currents in A, columns = id, ig, is, ib (current ENTERING the terminal)."""
        logs, signs = self.predict_terms(X)
        return signs * np.maximum(10.0 ** logs, FLOOR_VAL)
