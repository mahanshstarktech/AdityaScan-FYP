import numpy as np
import onnxruntime as ort
import joblib
import json
from pathlib import Path

class AdityScanONNXEngine:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        
        # Load the ONNX runtime session optimized for CPU inference
        self.session = ort.InferenceSession(
            str(self.model_dir / "adityscan_v4.onnx"),
            providers=['CPUExecutionProvider']
        )
        
        # Load calibration artifacts
        self.temp_scaler = joblib.load(self.model_dir / "temperature_scaler.pkl")
        self.conformal_predictor = joblib.load(self.model_dir / "conformal_predictor.pkl")
        self.feature_scaler = joblib.load(self.model_dir / "feature_scaler.pkl")
        
        # Extracted static calibration constant from training
        self.T = 0.863 

    def _apply_temperature_scaling(self, logits):
        """Applies T=0.863 post-hoc calibration to raw logits."""
        scaled_logits = logits / self.T
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict(self, raw_telemetry_window):
        """
        Executes a forward pass using the 1800-s rolling window.
        Input shape expected: (1, 1800, 22)
        """
        # 1. Scale the 22 physics features
        scaled_features = self.feature_scaler.transform(raw_telemetry_window.reshape(-1, 22))
        model_input = scaled_features.reshape(1, 1800, 22).astype(np.float32)

        # 2. ONNX Inference
        input_name = self.session.get_inputs()[0].name
        raw_outputs = self.session.run(None, {input_name: model_input})
        
        # ONNX export yields: [nowcast_logit, multi_horizon_logits, goes_class_logits, cme_risk_logit]
        goes_class_logits = raw_outputs[2]
        multi_horizon_logits = raw_outputs[1]

        # 3. Apply Calibration & Uncertainty Quantification
        goes_probs = self._apply_temperature_scaling(goes_class_logits)[0]
        
        # Generate 90% conformal intervals
        conformal_radius = self.conformal_predictor.get_q_hat(alpha=0.10)
        
        return {
            "goes_class_probabilities": {
                "B": float(goes_probs[0]),
                "C": float(goes_probs[1]),
                "M": float(goes_probs[2]),
                "X": float(goes_probs[3]),
                "X+": float(goes_probs[4])
            },
            "forecast_horizons": {
                "5_min": self._format_horizon(multi_horizon_logits[0][0], conformal_radius),
                "10_min": self._format_horizon(multi_horizon_logits[0][1], conformal_radius),
                "15_min": self._format_horizon(multi_horizon_logits[0][2], conformal_radius),
                "30_min": self._format_horizon(multi_horizon_logits[0][3], conformal_radius),
                "60_min": self._format_horizon(multi_horizon_logits[0][4], conformal_radius)
            },
            "status": "success",
            "inference_time_ms": 0 # to be populated by the API router
        }

    def _format_horizon(self, logit, radius):
        prob = 1 / (1 + np.exp(-logit / self.T))
        return {
            "probability": float(prob),
            "interval_90": [max(0.0, float(prob - radius)), min(1.0, float(prob + radius))]
        }