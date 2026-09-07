# RCF BSDS500+PASCAL

Inference port of https://github.com/yun-liu/RCF-PyTorch.
See NOTICE for the pinned source revision and LICENSE.md for the upstream license.
The original convolutions, bilinear upsampling, crop offsets and six sigmoid
outputs are preserved. Fixed upsampling kernels follow `.to()`, `.cpu()` and
`.cuda()` without changing checkpoint keys.

Download the official checkpoint to `models/bsds500_pascal_model.pth` in the
parent project:
https://github.com/yun-liu/RCF-PyTorch/releases/download/v1.0/bsds500_pascal_model.pth

SHA256: `9913d9ae1eaa4a71022e89e8c8f6e3eeab5f9bd1cb6a2cc91b1bba7bf36e898c`

```python
from rockedgesdetectors import RCFBSDS, NumpyRCFAdapter, Cropper

module = RCFBSDS("models/bsds500_pascal_model.pth").cuda().eval()
model = Cropper(NumpyRCFAdapter(module), crop=350, pad=50)
edges = model(rgb_image)
```

The adapter accepts RGB uint8 images or float images in [0,1]. It converts
to BGR and subtracts the original training mean. Output is a float32 HxW
edge probability map in [0,1], without inversion or per-tile normalization.
Direct inference requires both image dimensions >= 9; Cropper also handles
smaller source images through padding. The legacy `ModelRCF` interface still
accepts BGR images and now respects its `device` argument.

Run from the parent project root:

```powershell
.venv\Scripts\python.exe -m scripts_pipeline.task1_edge_detect_rcf
```

The script reads `preparation.folder_dataset` from `config.toml`, iterates
through Storage folders, and writes `<base_name>_edges_rcf.png`. It uses CUDA
when available, otherwise CPU. This is single-scale tiled inference; upstream
multi-scale testing is not included. Fine-tuning is available in `train/`;
see `train/README.md` for the official training algorithm and project adaptations.
