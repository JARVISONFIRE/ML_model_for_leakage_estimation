"""

Model wrapper for signed, per-terminal current prediction.



Design (why it differs from the old LogTransformedModel):

  * One model PER TERMINAL (id, ig, is, ib). The four currents differ by many decades,

    so a single joint model gets dominated by the largest terminal.

  * Magnitude is regressed in log10 space (targets standardised), and the SIGN is

    predicted separately (classifier, or a constant when a terminal never changes sign).

    The old wrapper returned |I| only, which makes summing currents (KCL) impossible.

  * Physics-motivated feature engineering: vds/vgs/vbs, ln(W), 1/T, (vgs-vth0)/T ...

    Subthreshold log|I| is close to linear in these.

  * Magnitudes are clipped at FLOOR_VAL = 1e-18 A (the abstol/gmin noise floor of the

    simulations) instead of 1e-22, so exact zeros do not create huge fake targets.



Public API used by the scripts:

    FEATURE_COLS, TARGET_COLS, FLOOR_VAL, EVAL_FLOOR

    available_model_types()

    SignedCurrentModel(model_type, seed).fit(X_df, y_df).predict(X_df) -> (n, 4) signed array

"""

import os

import warnings



import numpy as np

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler



FEATURE_COLS = ['vdd', 'temp', 'w', 'vd', 'vs', 'vg', 'vb', 'vth0', 'toxe', 'ndep', 'u0', 'xj']

TARGET_COLS = ['id', 'ig', 'is', 'ib']



FLOOR_VAL = 1e-18   # magnitude clip for training targets / predictions (= abstol of the sims)

EVAL_FLOOR = 1e-17  # samples with |I| below this are numerical noise and excluded from error stats



MLP_SEEDS = int(os.environ.get("LEAKAGE_MLP_SEEDS", "3"))      # ensemble size of the MLP

MLP_MAX_ITER = int(os.environ.get("LEAKAGE_MLP_MAX_ITER", "1500"))





# ----------------------------------------------------------------------------------------

# Feature engineering

# ----------------------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    inv_t = 1000.0 / (df['temp'] + 273.15)

    vds = df['vd'] - df['vs']

    vgs = df['vg'] - df['vs']

    vov = vgs - df['vth0']

    out = pd.DataFrame({

        'vdd': df['vdd'],

        'inv_t': inv_t,

        'temp': df['temp'],

        'ln_w': np.log(df['w']),

        'vd': df['vd'],

        'vs': df['vs'],

        'vds': vds,

        'vgs': vgs,

        'vgd': df['vg'] - df['vd'],

        'vbs': df['vb'] - df['vs'],

        'vbd': df['vb'] - df['vd'],

        'vg': df['vg'],

        'vth0': df['vth0'],

        'vov': vov,

        'ln_toxe': np.log(df['toxe']),

        'ln_ndep': np.log(df['ndep']),

        'ln_u0': np.log(df['u0']),

        'ln_xj': np.log(df['xj']),

    })

    out['vov_over_t'] = vov * inv_t

    out['vds_over_t'] = vds * inv_t

    return out





# ----------------------------------------------------------------------------------------

# Magnitude regressors (all fit on standardised log10|I|)

# ----------------------------------------------------------------------------------------

class EnsembleMLP:

    """Average of a few lbfgs-trained tanh MLPs (lbfgs is very precise on small smooth problems)."""



    def __init__(self, seed=42, n_models=MLP_SEEDS):

        self.seed, self.n_models = seed, n_models

        self.nets = []



    def fit(self, X, y):

        self.nets = []

        for k in range(self.n_models):

            net = MLPRegressor(hidden_layer_sizes=(64, 64, 32), activation='tanh', solver='lbfgs',

                               alpha=1e-6, max_iter=MLP_MAX_ITER, random_state=self.seed + 101 * k)

            with warnings.catch_warnings():

                warnings.simplefilter("ignore")      # lbfgs "did not converge" notices

                net.fit(X, y)

            self.nets.append(net)

        return self



    def predict(self, X):

        return np.mean([n.predict(X) for n in self.nets], axis=0)





def _make_xgboost(seed):

    import xgboost as xgb

    return xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8,

                            colsample_bytree=0.9, random_state=seed, n_jobs=-1)





def _make_lightgbm(seed):

    import lightgbm as lgb

    return lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31, min_child_samples=10,

                             subsample=0.8, subsample_freq=1, colsample_bytree=0.9,

                             random_state=seed, n_jobs=-1, verbose=-1)





def _make_rf(seed):

    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.7,

                                 random_state=seed, n_jobs=-1)





def _make_histgb(seed):

    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,

                                         random_state=seed)





_REGISTRY = {

    "MLP": lambda seed: EnsembleMLP(seed),

    "XGBoost": _make_xgboost,

    "LightGBM": _make_lightgbm,

    "RandomForest": _make_rf,

    "HistGB": _make_histgb,          # sklearn fallback; only used if listed explicitly

}

_DEFAULT_TYPES = ["MLP", "XGBoost", "LightGBM", "RandomForest"]





def available_model_types():

    """Default 4 model types, dropping any whose library is not installed.

    Override with env var LEAKAGE_MODELS="MLP,RandomForest,HistGB"."""

    wanted = os.environ.get("LEAKAGE_MODELS")

    names = [n.strip() for n in wanted.split(",")] if wanted else list(_DEFAULT_TYPES)

    ok = []

    for n in names:

        try:

            _REGISTRY[n](0)

            ok.append(n)

        except Exception as e:                      # ImportError or unknown name

            print(f"[!] Model type '{n}' unavailable ({type(e).__name__}: {e}); skipping.")

    return ok





# ----------------------------------------------------------------------------------------

# Signed, per-terminal model

# ----------------------------------------------------------------------------------------

class SignedCurrentModel:

    def __init__(self, model_type: str, seed: int = 42):

        self.model_type, self.seed = model_type, seed

        self.scaler = StandardScaler()

        self.mag_models, self.y_stats = {}, {}

        self.sign_models = {}     # terminal -> classifier or constant (+1/-1)



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

            reg = _REGISTRY[self.model_type](self.seed)

            reg.fit(Xs, (ylog - mu) / sd)

            self.mag_models[t] = reg



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

            logs.append(self.mag_models[t].predict(Xs) * sd + mu)

            s = self.sign_models[t]

            signs.append(np.full(len(Xs), s) if np.isscalar(s) else np.where(s.predict(Xs) == 1, 1.0, -1.0))

        return np.column_stack(logs), np.column_stack(signs)



    def predict(self, X: pd.DataFrame) -> np.ndarray:

        """Signed currents in A, columns = id, ig, is, ib (current ENTERING the terminal)."""

        logs, signs = self.predict_terms(X)

        return signs * np.maximum(10.0 ** logs, FLOOR_VAL)
