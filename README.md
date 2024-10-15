# rockedgesdetectors
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