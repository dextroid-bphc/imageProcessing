import cv2
import numpy as np
import argparse
import time
import subprocess
import psutil

BLUE =  '\033[0;38;2;32;128;192m'
RED =   '\033[0;38;2;255;32;32m'
GREEN = '\033[0;38;2;0;192;0m'
YELLOW ='\033[0;38;2;192;192;0m'
NC =    '\033[0m'

# for keeping frameRate's rolling avg
avg = np.ndarray(shape=(1,10), dtype=float)
for x in range(len(avg[0])):
    avg[0][x] = 0.0
def rollAvg(x):
    global avg
    avg = np.roll(avg,1)
    avg[0][0] = x
    #print(avg)
    return np.sum(avg) / 10.0

def cmdLine(cmd):
    process = subprocess.Popen(args = cmd, stdout = subprocess.PIPE, universal_newlines = True, shell = True)
    return process.communicate()[0]

def consolePrint(msg):
    print(">" + msg)

ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
ap.add_argument("-cl", "--confidenceLabel", type=float, default=0.85, help="minimum probability to filter strong detections")
ap.add_argument("-cr", "--confidenceRect", type=float, default=0.05, help="minimum probability to filter weak detections")
ap.add_argument("-c1", "--extCamera", action='store_true')
ap.add_argument("-fw", "--width", action='store', default = 300, type=int)
ap.add_argument("-fh", "--height", action='store', default = 300, type=int)
ap.add_argument("-mfr", "--maxFrameRate", action='store', default = 15.0, type=float, help = 'maximum frame rate that should be processed')
ap.add_argument("-wt", "--watchTemperature", action='store', default = 90.0, type=float, help = 'threshold temperature for throttling')
ap.add_argument("-ns", "--noshow", action='store_true', default = False)
args = ap.parse_args()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "table",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "screen"]

#COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLORS =[
    [ 48.45663575,  33.71783245, 160.0331461 ], # background
    [ 28.35609305,  43.2382419 , 126.26032403], # aeroplane
    [238.99214345, 174.70501852,  65.96561562], # bicycle
    [238.70420089, 135.95906804,  69.72652473], # bird
    [234.9233787 ,  49.24844281,  53.5632204 ], # boat
    [148.98987605, 196.91828719,  88.41692749], # bottle
    [187.1156841 ,  48.69416351,  34.98664509], # bus
    [157.40254339, 156.88181776, 182.2419243 ], # car 
    [166.63628602,  12.82611402, 184.30525981], # cat
    [  7.03089482, 199.87139056,  38.54555922], # chair
    [118.86066783, 123.88401146, 226.9210897 ], # cow
    [193.55343459, 214.08615103,  31.5229942 ], # table
    [ 57.62116528, 215.4512353 , 171.74540365], # dog
    [ 67.66099027, 210.18069248,  87.45066166], # horse
    [212.32059355, 147.10722589,   9.02163039], # motorbike 
    [164.67043415, 230.84978889,  93.57353107], # person
    [ 20.64037859, 224.68043457, 131.14701982], # pottedplant
    [ 61.06152017, 141.03391818,  58.98316371], # sheep
    [212.0190112 ,  83.79312278, 136.57798582], # sofa
    [240.68507159, 148.17770628,  28.56186161], # train
    [168.74494908, 174.50197005,  71.61536541]] # screen

confidenceThresholdLabel = args.confidenceLabel
confidenceThresholdRect = args.confidenceRect
targetFrameRate = args.maxFrameRate
thresholdTemp = args.watchTemperature
frameWidth = args.width
frameHeight = args.height

# precursory check for display
if not args.noshow:
    if(cmdLine("xrandr -q") == ""):
        # no display is available, switch to no-show
        args.noshow = True
        consolePrint(RED+"No video output available, switching to no-show mode"+NC)

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
# optionally use DNN_TARGET_OPENCL_FP16
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
frameRate = 0.0
lastTempCheckTime = time.perf_counter()
temp = float(cmdLine("vcgencmd measure_temp")[5:-3])
cpu = float(psutil.cpu_percent())
ram = float(psutil.virtual_memory()[2])

if(args.extCamera):
    cap=cv2.VideoCapture(1)
else:
    cap=cv2.VideoCapture(0)

try:
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
                # show frame only if xrandr available
                if not args.noshow:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                # check for confidence threshold for making rectangle
                if confidence > confidenceThresholdLabel:
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    print(label, end=",")
                    # show frame only if xrandr available
                    if not args.noshow:
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output frame if specified
        if not args.noshow:
            cv2.rectangle(frame, (1,1), (105,42),(0,0,0,0.4), -1)
            cv2.putText(frame, str(round((frameRate),2))+" FPS", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
            cv2.putText(frame, str(round((temp),1))+"'C", (64, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
            cv2.putText(frame, "CPU: "+str(round((cpu),1))+"%", (2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
            cv2.putText(frame, "Mem: "+str(ram)+"%", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        delay = 1/targetFrameRate
        #print(rem, delay, now-prev)
        availableFrameRate = 1/(time.perf_counter() - prev)
        #print("FPS: %.2f" %(frameRate))
        temp = float(cmdLine("vcgencmd measure_temp")[5:-3])
        #cpu = float(cmdLine("top -bn 2 -d 0.01 | grep '^%Cpu' | tail -n 1 | awk '{print ($2+$4)*100/($2+$4+$8)}'"))
        cpu = float(psutil.cpu_percent());
        ram = float(psutil.virtual_memory()[2])
        print("FPS: %.2f | Avg: %.2f | Available: %.2f" %(frameRate,rollAvg(frameRate),float(availableFrameRate)))

        # watch temperature
        if(temp > thresholdTemp and time.perf_counter() - lastTempCheckTime > 5):
            consolePrint(YELLOW+"Temperature high, reducing frame rate from %.2f to %.2f" %(frameRate,frameRate-0.1)+NC)
            if(targetFrameRate > 0.1):
                targetFrameRate -= 0.1
            lastTempCheckTime = time.perf_counter()
        if(args.maxFrameRate - targetFrameRate >= 0.1 and thresholdTemp - temp > 2 and time.perf_counter() - lastTempCheckTime > 10):
            consolePrint(GREEN+"Increasing frame rate from %.2f to %.2f" %(frameRate,frameRate+0.1)+NC)
            targetFrameRate += 0.1
            lastTempCheckTime = time.perf_counter()

        # wait before capturing next frame to maintain frame rate
        rem = delay - (time.perf_counter()-prev)
        if(rem > 0):
            time.sleep(round(rem,3))
        frameRate = 1 / (float(time.perf_counter()) - prev)
# for when the camera reads null
except AttributeError as e:
    # but keep the service live
    consolePrint(RED+"Corrupt Frame!"+NC)
except KeyboardInterrupt as e:
    print("\nSIGINT\n")

cap.release()
cv2.destroyAllWindows()
