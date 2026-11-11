"""
Model Wrapper for Log-Transformed Current Prediction
Ensures consistent serialization and deserialization across scripts.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

FLOOR_VAL = 1e-22


class LogTransformedModel:
    """Wrapper that fits models on log10(|I|) per project specifications."""
    def __init__(self, base_estimator):
        self.scaler = StandardScaler()
        self.model = base_estimator

    def fit(self, X, y):
        y_abs = np.abs(y.values)
        y_log = np.log10(np.maximum(y_abs, FLOOR_VAL))

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_log)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_log_pred = self.model.predict(X_scaled)
        if y_log_pred.ndim == 1:
            y_log_pred = y_log_pred[:, np.newaxis]
        # Reconstruct current magnitude in Amperes
        y_pred = 10.0 ** y_log_pred
        return y_pred
