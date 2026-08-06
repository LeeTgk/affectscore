# game/affectscore/encoder.py
import os
import time


class _AffectEncoderImpl:
    """
    Runs the pretrained MLP that maps the combined two-layer signal to a 512-d embedding.

    Uses ONNX Runtime for fast CPU inference (~1-2 ms per call).
    The model file is at game/affectscore/weights/encoder_mlp.onnx.
    Input: 6 floats [scene_valence, scene_arousal, arc_position,
           choice_latency_norm, dwell_deviation_norm, interaction_rate_norm]
    Output: 512 floats (L2-normalized conditioning embedding)
    """

    def __init__(self):
        self._session = None
        self._input_name = None
        self._output_name = None

    def load(self):
        """Load ONNX model. Call once at game start."""
        try:
            import onnxruntime as ort
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "weights",
                "encoder_mlp.onnx",
            )
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            print("[AffectScore] Encoder loaded successfully.")
        except Exception as e:
            print(f"[AffectScore] Encoder load failed: {e}")
            self._session = None

    def encode(self, signal):
        """Encode combined signal into a 512-d conditioning embedding.

        Args:
            signal: List of 6 floats or dict with named keys.
        Returns:
            List of 512 floats, or zero vector on failure.
        """
        if self._session is None:
            return [0.0] * 512

        import numpy as np

        feature_order = [
            "scene_valence",
            "scene_arousal",
            "arc_position",
            "choice_latency_norm",
            "dwell_deviation_norm",
            "interaction_rate_norm",
        ]

        if isinstance(signal, dict):
            vec = np.array(
                [[signal.get(k, 0.0) for k in feature_order]],
                dtype=np.float32,
            )
        else:
            vec = np.array(
                [list(signal)],
                dtype=np.float32,
            )

        try:
            result = self._session.run(
                [self._output_name],
                {self._input_name: vec},
            )
            embedding = result[0][0].tolist()
            return embedding
        except Exception as e:
            print(f"[AffectScore] Encoding error: {e}")
            return [0.0] * 512
