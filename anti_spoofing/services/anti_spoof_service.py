# services/anti_spoof_service.py
"""Anti-spoofing inference service — exact mirror of fas_test_repo/main.py::predict()."""

import numpy as np
import torch

from ..models.model_loader import ModelLoader
from ..utils.image_preprocess import to_tensor


class AntiSpoofService:
    """Runs MiniFASNet inference on a pre-cropped 80×80 face patch.

    The face patch MUST have been produced by utils.face_cropper.get_face_crop()
    (which applies the scaled crop_face() pipeline) — NOT a raw tight crop.

    Label mapping (identical to fas_test_repo/main.py::predict()):
        index 0 → Fake / Spoof
        index 1 → Real / Live
        index 2 → (third class; argmax never selects it for live faces in practice)

    Production thresholds (applied on top of the reference label logic):
        live_score >= 0.95  → LIVE    (high confidence real)
        live_score >= 0.80  → SUSPECT (ambiguous)
        live_score <  0.80  → SPOOF
    """

    def __init__(self, weight_path: str, model_name: str = "v2") -> None:
        loader = ModelLoader(weight_path, model_name)
        self._model = loader.get_model()
        self._device = loader.get_device()

    def predict(self, face_patch: np.ndarray) -> dict:
        """Run anti-spoofing inference on an 80×80 BGR face patch.

        Args:
            face_patch: numpy array of shape (80, 80, 3), dtype uint8, BGR.
                        Must be the output of face_cropper.get_face_crop().

        Returns:
            dict with keys:
                label        – "LIVE" | "SUSPECT" | "SPOOF"
                confidence   – live-class probability (float, 0..1)
                is_live      – True only when label == "LIVE"
                probabilities – raw softmax output as list[float]
        """
        # --- Tensor preparation (exact mirror of fas_test_repo predict()) ---
        # to_tensor: HWC uint8 → CHW float (no normalisation)
        tensor = to_tensor(face_patch).unsqueeze(0).to(self._device)

        # --- Inference ---
        with torch.no_grad():
            output = self._model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()

        # --- Class interpretation (exact mirror of fas_test_repo predict()) ---
        # label_idx == 1  →  Real;  label_idx != 1  →  Fake
        label_idx = int(np.argmax(probs))
        raw_score = float(probs[0, label_idx])  # confidence of the winning class

        # live_score is always the probability of class-1 (Real), regardless of argmax.
        live_score = float(probs[0, 1])

        # Production tiered labels.
        if live_score >= 0.95:
            label = "LIVE"
            is_live = True
        elif live_score >= 0.80:
            label = "SUSPECT"
            is_live = False
        else:
            label = "SPOOF"
            is_live = False

        return {
            "label": label,
            "confidence": live_score,
            "is_live": is_live,
            "raw_label": "Real" if label_idx == 1 else "Fake",
            "raw_score": raw_score,
            "probabilities": probs[0].tolist(),
        }