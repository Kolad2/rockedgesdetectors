# RCF training adaptation

Source: https://github.com/yun-liu/RCF-PyTorch, revision
`3e6b3a2772197da47ca37add4ba049e635897737`. Upstream license: ../LICENSE.md.

The implementation follows the authors' training algorithm:

- `data.py`: BGR pixels minus `[104.00698793,116.66876762,122.67891434]`,
  and upstream labels: zero = background, values >= 127.5 = edge,
  intermediate nonzero values = ignored (label 2).
- `loss.py`: `utils.py::Cross_entropy_loss`, balanced BCE summed over pixels
  and all six sigmoid outputs with equal weights. Counts span the batch.
- `optimization.py`: `train.py` SGD, momentum 0.9, base LR 1e-6, weight decay
  2e-4. Weight LR multipliers: conv1-4 = 1, conv5 = 100, down = 0.1,
  side = 0.01, fusion = 0.001. Bias LR is doubled with no weight decay.
- `trainer.py`: float32 forward, sum of six losses, division by iter_size,
  backward and SGD after gradient accumulation. No AMP or probabilistic loss.
- `runner.py`: 10 epochs, iter_size 10, StepLR every 3 epochs with gamma 0.1.

Project adaptations:

- Fine-tune the official BSDS500+PASCAL weights rather than initialize from
  MATLAB ImageNet VGG weights.
- Accept tab-separated `image label [mask]` manifests (whitespace also works
  when paths have no spaces). Relative paths resolve against the manifest.
  Mask pixels below threshold and crop padding receive ignore label 2.
- Optional random crops, flips and edge-containing crop selection; validation
  uses a fixed split and center crops. These are project data handling options,
  not the authors' BSDS benchmark evaluation or multi-scale test.
- Flush the final incomplete accumulation group, dividing by its actual size.
  Ignore-only and single-class crops have zero balanced BCE, without NaNs.
- Save atomically after scheduler.step(), using one-based completed epoch
  counts. Resume restores model, SGD momentum and scheduler; it does not
  restore random/worker state for bitwise reproduction of future crops.
  The local checkpoint version distinguishes this convention from upstream's
  zero-based epoch and pre-scheduler-step saves. Original weights remain valid
  for initial_checkpoint; resume_checkpoint expects a local training checkpoint.

Run from the parent project root:

```powershell
.venv\Scripts\python.exe -m scripts_train.train_rcf
```

All editable settings are in `scripts_train/train_rcf.py`. Set
`resume_checkpoint` to a saved checkpoint to continue; `epochs` is the total
target epoch count. Checkpoints in `save_models/rcf` can also be loaded by
`RCFBSDS` for inference. `ignore_ambiguous=False` switches to binary thresholded
labels when needed for project annotations. `device="cpu"` supports smoke tests.

Run the regression checks from the parent project root:

```powershell
.venv\Scripts\python.exe -m unittest rockedgesdetectors.pyrcf.train.test_training
```
