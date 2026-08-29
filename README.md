# rockedgesdetectors

## DiffusionEdge BSDS

The `diffusion_edge` package contains an inference-only PyTorch port of the
official DiffusionEdge BSDS model. It uses only PyTorch, torchvision and NumPy;
the training-time dependencies from the authors' repository are not required.
The fixed model input is 320x320, so large images should be processed with
`Cropper(crop=320, pad=40)`.

Original implementation and weights:
https://github.com/GuHuangAI/DiffusionEdge

## DDN BSDS500

The `ddn` package contains a pure-PyTorch inference port of DDN-M36. It loads
the authors' complete BSDS500 checkpoint directly and does not require timm or
a separate CAFormer checkpoint. Large images can be processed with
`Cropper(crop=320, pad=40)`; the smaller crop size keeps attention memory usage
reasonable on 4 GB GPUs. Normalize the final assembled edge map rather than
each crop separately to avoid visible tile boundaries.

Original implementation and weights:
https://github.com/Li-yachuan/DDN

Fine-tuning components are available in `ddn.training`. The repository-level
entry point is `scripts_train/train_ddn.py`; its paths and hyperparameters are
kept at the top of that script.

```bash
python scripts_train/train_ddn.py
```

## RCF model example
rcf model [download](https://drive.google.com/file/d/1ZY6W41xDJjG5jERd9aDHo6NJhu_H0EsW/view?usp=sharing)

```python
import cv2
from rockedgesdetectors import ModelRCF

big_frame = cv2.imread("image.png")
model = ModelRCF("../models/RCFcheckpoint_epoch12.pth")
edges = model.get_model_edges(big_frame)
#
cv2.namedWindow("wnd", cv2.WINDOW_NORMAL)
cv2.resizeWindow('wnd', 800, 600)
cv2.imshow("wnd", big_frame)
cv2.waitKey(0)
#
cv2.namedWindow("wnd", cv2.WINDOW_NORMAL)
cv2.resizeWindow('wnd', 800, 600)
cv2.imshow("wnd", edges)
cv2.waitKey(0)
```

## PiDiNet model example

```python
import cv2
import matplotlib.pyplot as plt
from rockedgesdetectors import ModelPiDiNet

checkpoint_path_7 = "models/pidinetmodels/table7_pidinet.pth"
checkpoint_path_5 = "models/pidinetmodels/table5_pidinet.pth"


model = ModelPiDiNet(checkpoint_path_7)
image = cv2.imread(f"..//images//test.png")
result_1 = model(image)

model = ModelPiDiNet(checkpoint_path_5)
image = cv2.imread(f"..//images//test.png")
result_2 = model(image)

fig = plt.figure(figsize=(7, 9))
axs = [fig.add_subplot(2, 2, 1),
       fig.add_subplot(2, 2, 3),
       fig.add_subplot(2, 2, 4)]
axs[0].imshow(image)
axs[1].imshow(result_1)
axs[2].imshow(result_2)
plt.show()
```
