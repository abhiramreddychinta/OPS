import numpy as np
import cv2

net = cv2.dnn.readNet('yolov3_training_1000.weights', 'yolov3_testing.cfg')
classes = ['Mobile']
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    img = cv2.resize(img, (500, 500))
    h, w, q = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))
                top_left_x = int(center_x - (width / 2))
                top_left_y = int(center_y - (height / 2))
                boxes.append([top_left_x, top_left_y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.6)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            cv2.putText(img, "Mobile Detected", (x, y - 5), FONT,  1, (0, 255, 0), 1)

    cv2.imshow('Frame', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

cam.release()
cv2.destroyAllWindows()
