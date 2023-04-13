import numpy as np
import cv2
import yaml
from yaml import SafeLoader
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
 
# Colors.

colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
class YOLO_Process:
    def __init__(self,weight,yml):
        self.weight = weight
        self.yml =yml
        self.classes = self.load_classes()
    def predict(self,img):
        net = cv2.dnn.readNetFromONNX(self.weight)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT),  swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds
    
    def load_classes(self):
        with open(self.yml) as f:
            data = list(yaml.load_all(f, Loader=SafeLoader))
            return data[0]['names']
    def process_img(self,frame):

        inputImage = format_yolov5(frame)
        outs = self.predict(inputImage)

        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, self.classes[classid]+": "+str(round(confidence,2)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
        return frame
    def process_video(self,video):
            video = cv2.VideoCapture(video)
            while True:
                _, frame = video.read()
                if frame is None:
                    print("End of stream")
                    break
                output = self.process_img(frame)
                cv2.imshow("output", output)
                if cv2.waitKey(1) > -1:
                    print("finished by user")
                    break

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


# step 2 - feed a 640x640 image to get predictions
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


