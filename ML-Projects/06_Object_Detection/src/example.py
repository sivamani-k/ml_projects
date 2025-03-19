import cv2
import numpy as np
image = np.zeros((500, 500, 3), dtype='uint8')
cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), 2)
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()