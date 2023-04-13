import cv2
from yolo import YOLO_Process


YOLO = YOLO_Process("best2.onnx","data2.yaml")
img  = cv2.imread("102200165.jpg")
#img = cv2.resize(img,(1000,600))
output= YOLO.process_img(img)
cv2.imshow("Result",output)
cv2.waitKey(0)
cv2.destroyAllWindows()


