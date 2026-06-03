# models/model_loader.py
"""Model loading — exact mirror of fas_test_repo/main.py::load_model()."""

import torch

from .fastnet import MiniFASNetV2, MiniFASNetV1SE


# Mirror of MODEL_CONFIGS in fas_test_repo/main.py.
MODEL_CONFIGS = {
    "v1se": {"class": MiniFASNetV1SE, "input_size": (80, 80), "scale": 4.0},
    "v2":   {"class": MiniFASNetV2,   "input_size": (80, 80), "scale": 2.7},
}


class ModelLoader:
    """Loads a MiniFASNet variant from a .pth checkpoint.

    Mirrors fas_test_repo/main.py::load_model() exactly:
        - Selects device (CUDA if available, else CPU).
        - Instantiates the correct model class.
        - Calls load_state_dict with weights_only=True.
        - Sets eval mode.
    """

    def __init__(self, weight_path: str, model_name: str = "v2") -> None:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model variant '{model_name}'. Choose from {list(MODEL_CONFIGS)}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = MODEL_CONFIGS[model_name]

        model = config["class"]()
        model.load_state_dict(
            torch.load(weight_path, map_location=self._device, weights_only=True)
        )
        model.to(self._device).eval()

        self._model = model
        self._config = config
        print(f"[ModelLoader] Loaded '{model_name}' on {self._device}")

    def get_model(self):
        return self._model

    def get_device(self) -> torch.device:
        return self._device

    def get_config(self) -> dict:
        """Returns the model config dict (input_size, scale)."""
        return self._config