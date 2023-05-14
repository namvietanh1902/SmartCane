import cv2
from yolo import YOLO_Process

print(cv2.__version__)
YOLO = YOLO_Process("best2.onnx","data2.yaml")
img  = cv2.imread("image/group2.jpg")

object_name,output= YOLO.process_img(img)
print(object_name)

cv2.imshow("Result",output)
cv2.waitKey(0)
cv2.destroyAllWindows()

