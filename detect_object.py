import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-cl", "--confidenceLabel", type=float, default=0.85, help="minimum probability to filter strong detections")
ap.add_argument("-cr", "--confidenceRect", type=float, default=0.05, help="minimum probability to filter weak detections")
ap.add_argument("-c1", "--extCamera", action='store_true')
ap.add_argument("-fw", "--width", action='store', default = 300, type=int)
ap.add_argument("-fh", "--height", action='store', default = 300, type=int)
args = ap.parse_args()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
conf=0.7; #Minimum Confidence value
confidenceThresholdLabel = args.confidenceLabel
confidenceThresholdRect = args.confidenceRect
targetFrameRate = args.maxFrameRate
thresholdTemp = args.watchTemperature
frameWidth = args.width
frameHeight = args.height

#Loading the pretrained model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

if(args.extCamera):
    cap=cv2.VideoCapture(1)
else:
    cap=cv2.VideoCapture(0)

frameRate = 0.0
while cap.isOpened():
        prev = time.perf_counter()
        ret,frame=cap.read()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (frameHeight, frameWidth)), 0.007843, (frameHeight, frameWidth), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # check for confidence threshold for making rectangle
            if confidence > confidenceThresholdRect:
                idx = int(detections[0, 0, i, 1])
            	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                # check for confidence threshold for making rectangle
                if confidence > confidenceThresholdLabel:
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print(label, end=",")
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        cv2.rectangle(frame, (1,1), (105,42),(0,0,0,0.4), -1)
        cv2.putText(frame, str(round((frameRate),2))+" FPS", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        delay = 1/targetFrameRate
        #print(rem, delay, now-prev)
        availableFrameRate = 1/(time.perf_counter() - prev)
        #print("FPS: %.2f" %(frameRate))
        print("FPS: %.2f | Avg: %.2f | Available: %.2f" %(frameRate,rollAvg(frameRate),float(availableFrameRate)))

        # wait before capturing next frame to maintain frame rate
        rem = delay - (time.perf_counter()-prev)
        if(rem > 0):
            time.sleep(round(rem,3))
        frameRate = 1 / (float(time.perf_counter()) - prev)

cap.release()
cv2.destroyAllWindows()
