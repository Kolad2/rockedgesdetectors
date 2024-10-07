# rockedgesdetectors

rcf model [download](https://drive.google.com/file/d/1ZY6W41xDJjG5jERd9aDHo6NJhu_H0EsW/view?usp=sharing)


```python
import cv2
from rockedgesdetectors import ModelGPU

big_frame = cv2.imread("image.png")
model = ModelGPU("../models/RCFcheckpoint_epoch12.pth")
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
