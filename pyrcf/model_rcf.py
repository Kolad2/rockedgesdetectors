"""Official checkpoint loader and legacy BGR interface."""
from pathlib import Path
import torch
from .models import RCF

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "bsds500_pascal_model.pth"

class RCFBSDS(RCF):
    """Official RCF: mean-subtracted BGR NCHW input, six probability maps."""
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, trainable: bool = False):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RCF checkpoint not found: {checkpoint_path}")
        super().__init__()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state.get("state_dict", state), strict=True)
        self.requires_grad_(trainable)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

class ModelRCF:
    """Compatibility wrapper accepting BGR uint8 images, as the old API did."""
    def __init__(self, path_to_model=None, device="cuda"):
        from .adapter import NumpyRCFAdapter
        self.model = RCFBSDS(path_to_model or DEFAULT_CHECKPOINT).to(device).eval()
        self.adapter = NumpyRCFAdapter(self.model)

    def __call__(self, image):
        return self.get_model_edges(image)

    def get_model_edges(self, image):
        return self.adapter(image[..., ::-1])
