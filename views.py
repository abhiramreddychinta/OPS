# web app plugins
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
# web app plugins

# ML Algorithms
from scipy.spatial import distance as dist
import threading
import concurrent.futures
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
# ML Algorithms

# utilities
from gaze_tracking.gaze_tracking import GazeTracking
from django.core.files.base import File
from .models import Cloud
from datetime import datetime as d
import os, shutil
import zipfile
# utilities


# FOLDER CREATION
path = r'C:\Users\SUNEEL VARMA\PycharmProjects\proctorProject'
folderName = '\Screenshots'
path += folderName
if os.path.isdir(path):
    shutil.rmtree(path)
if not os.path.isdir(path):
    os.mkdir(path)
# FOLDER CREATION


# GLOBAL VARIABLES
YAWNSPERMINUTE = [] #yawning
YAWNS_TRACKER = 0
DROWSINESSPMIN = [] #blinks
DROWSINESS_TRACKER = 0
MALPRACTICE = [] #object_detection
OBJECT_TRACKER = 0
MULTIPLE_FACES = []
multipleFacesTracker = 0
DISTRACTION = [] #gaze_tracking
GAZE_TRACKER = 0
NUMBER_OF_CALLS = 0
COUNT = 0
# EXTERNAL DATA
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
net = cv2.dnn.readNet('yolov3_training_1000.weights', 'yolov3_testing.cfg')
numberOfFacesDetector = dlib.get_frontal_face_detector()
# EXTERNAL DATA

# GLOBAL VARIABLES


#ZIPPING A FOLDER
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))
    return None
#ZIPPING A FOLDER


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        return self.frame

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR


def drawLipContour(frame, shape):
    lip = shape[48:60]
    cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)


def is_yawning(shape, threshold):
    """functions returns true if yawn detected else false"""
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance > threshold


def is_drowsy(IsBlinking):
    """functions returns true if yawn detected else false"""
    global COUNT
    if IsBlinking:
        COUNT += 1
        if COUNT > 5:
            return True
    else:
        COUNT = 0
    return False


def is_blinking(frame, shape):
    """functions returns true if yawn detected else false"""
    EARThresh = 0.3
    (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lstart:lend]
    rightEye = shape[rstart: rend]
    leftEAR = eyeAspectRatio(leftEye)
    rightEAR = eyeAspectRatio(rightEye)

    EAR = (leftEAR + rightEAR) / 2.0

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
    return EAR < EARThresh


def IsMobileDetected(frame, net, output_layers):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    results = net.forward(output_layers)

    for result in results:
        for detection in result:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                return {"IsMobilePresent": True}
    return {"IsMobilePresent": False}


def facialResult(frame, shape, YAWN_THRESH=20):
    IsBlinking = is_blinking(frame, shape)
    IsDrowsy = is_drowsy(IsBlinking)
    IsYawning = is_yawning(shape, YAWN_THRESH) and IsBlinking
    return {"IsBlinking": IsBlinking, "IsDrowsy": IsDrowsy, "IsYawning": IsYawning}

def gaze_tracker(frame):
    global numberOfFacesDetector, predictor
    gaze = GazeTracking(numberOfFacesDetector, predictor)
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    flag = False
    if gaze.is_right() or gaze.is_left():
        flag = True
    return {"LookingAway": flag}

def gen(camera):
    # GLOBALS
    global SECONDS, NUMBER_OF_CALLS, YAWNS_TRACKER, YAWNSPERMINUTE, DROWSINESS_TRACKER, \
        DROWSINESSPMIN, OBJECT_TRACKER, multipleFacesTracker, MULTIPLE_FACES, MALPRACTICE, GAZE_TRACKER, DISTRACTION, predictor, detector, net, numberOfFacesDetector
    # GLOBALS

    NUMBER_OF_CALLS += 1
    if NUMBER_OF_CALLS == 24: # signifies a minute
        # update Global metric lists
        # set metric trackers to zero
        YAWNSPERMINUTE.append(YAWNS_TRACKER)
        YAWNS_TRACKER = 0
        DROWSINESSPMIN.append(DROWSINESS_TRACKER)
        DROWSINESS_TRACKER = 0
        MALPRACTICE.append(OBJECT_TRACKER)
        OBJECT_TRACKER = 0
        DISTRACTION.append(GAZE_TRACKER)
        GAZE_TRACKER = 0
        MULTIPLE_FACES.append(multipleFacesTracker)
        multipleFacesTracker = 0

        # Number of calls set to zero
        NUMBER_OF_CALLS = 0


    layer_names = net.getLayerNames()

    classes = ['Mobile']
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Variables
    YawnTracker, DrowsinessTracker, multipleFacesTracker = 0, 0, 0
    # Variables

    while True:
        frame = camera.get_frame()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        numberOfFaces = numberOfFacesDetector(gray)
        threadPool = concurrent.futures.ThreadPoolExecutor()

        rects = detector.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        results = dict()
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            """ threading """
            activeThreads = [threadPool.submit(facialResult, frame, shape),
                             threadPool.submit(IsMobileDetected, frame, net, output_layers),
                             threadPool.submit(gaze_tracker,frame)]
            for thread in concurrent.futures.as_completed(activeThreads): results.update(thread.result())
            """ threading """

            multipleFaces = len(numberOfFaces) > 1
            drawLipContour(frame, shape)
            timeStamp = d.now().strftime('%X%f').replace(':', '')

            if results["IsDrowsy"]:
                DROWSINESS_TRACKER += 1
                fileName = '\Drowsy' + timeStamp + '.jpg'
                cv2.imwrite(path + fileName, frame)
            if results["IsYawning"]:
                YawnTracker += 1
                YAWNS_TRACKER += 1
                fileName = '\Yawn' + timeStamp + '.jpg'
                cv2.imwrite(path + fileName, frame)
            if multipleFaces:
                multipleFacesTracker += 1
                fileName = '\MultipleIndividuals' + timeStamp + '.jpg'
                cv2.imwrite(path + fileName, frame)
            if results["IsMobilePresent"]:
                OBJECT_TRACKER += 1
                fileName = '\MobileDetected' + timeStamp + '.jpg'
                cv2.imwrite(path + fileName, frame)

            if results["LookingAway"]:
                GAZE_TRACKER += 1
                fileName = '\LookingAway' + timeStamp + '.jpg'
                cv2.imwrite(path + fileName, frame)
                print("Looking Away")

        return {
                "IsDrowsy": bool(DROWSINESS_TRACKER),
                "IsYawning":  bool(YawnTracker),
                "IsMobilePresent": int(results.get("IsMobilePresent", 0)),
                "LookingAway": bool(GAZE_TRACKER),
                "MultipleFace": bool(multipleFacesTracker)
                }

def optimizeFolder(path):
    files = os.listdir(path)
    if not files:
        return False
    else:
        command = 'optimize-images ' + '"' + path + '"' + '\.'
        os.system(command)
        return True


def zipFolder(path):
    sourceLocation = path
    dropLocation = path[:path.rfind("\\")] + "\Images.zip"
    zipf = zipfile.ZipFile(dropLocation, 'w', zipfile.ZIP_DEFLATED)
    zipdir(sourceLocation, zipf)
    zipf.close()
    return True, dropLocation


def uploadToCloud(path):
    """uploads the folder present in path to cloud"""
    #file ready
    timeStamp = d.now().strftime('%X%f').replace(':', '')
    try:
        zippedFolder = File(open(path, 'rb'))
        cloud = Cloud()
        cloud.upload.save("Images_"+timeStamp, zippedFolder, save=True)
    except Exception as e:
        print(e)
        return False
    return True

def OptimizeAndUploadToCloud(path):
    optimized = optimizeFolder(path)
    if optimized: zipped,path = zipFolder(path)
    if optimized and zipped: uploaded = uploadToCloud(path)
    return True if optimized and zipped and uploaded else False

def UploadToCloudService(request):
    isUploaded = OptimizeAndUploadToCloud(path)
    print(isUploaded)
    return render(request, 'proctorservice/uploadToCloud.html')


def calcMetrics():
    global YAWNSPERMINUTE, DROWSINESSPMIN, MALPRACTICE, GAZE_TRACKER, MULTIPLE_FACES
    yawnSummary,drowSummary  = sum(YAWNSPERMINUTE)/len(YAWNSPERMINUTE), sum(DROWSINESSPMIN)/len(DROWSINESSPMIN)
    interestedness = round((yawnSummary+drowSummary)/2, 2)
    malprtc = sum(MALPRACTICE)+sum(MULTIPLE_FACES)
    distractedNess = (round(sum(DISTRACTION)/len(DISTRACTION), 2)*0.80)+(malprtc*0.20)
    return (100-interestedness), malprtc, distractedNess

cam = False
PREV_SESSION = -1
def index(request, start):
    # GLOBALS
    global SECONDS, NUMBER_OF_CALLS, YAWNS_TRACKER, YAWNSPERMINUTE, DROWSINESS_TRACKER, \
        DROWSINESSPMIN, OBJECT_TRACKER,multipleFacesTracker, MULTIPLE_FACES, MALPRACTICE, GAZE_TRACKER, DISTRACTION, PREV_SESSION, COUNT, cam, path
    # GLOBALS
    print(start, PREV_SESSION)
    context = {"start": start}
    if start in (2, -1):
        start = 2
        PREV_SESSION = "neither"
        return render(request, 'proctorservice/index.html', context)
    elif start == 1 and PREV_SESSION == "neither":
        PREV_SESSION = "start"
        cam = VideoCamera()
        YAWNSPERMINUTE = []  #yawning
        YAWNS_TRACKER = 0
        DROWSINESSPMIN = []  #blinks
        DROWSINESS_TRACKER = 0
        MALPRACTICE = []  #object_detection
        OBJECT_TRACKER = 0
        DISTRACTION = []  #gaze_tracking
        GAZE_TRACKER = 0
        MULTIPLE_FACES = [] #multipleIndividuals
        multipleFacesTracker = 0
        NUMBER_OF_CALLS = 0
        COUNT = 0
    elif start == 0 and PREV_SESSION == "start":
        cam.__del__()
        if len(YAWNSPERMINUTE) == 0:
            return render(request, 'proctorservice/sessionEndedQuickly.html')
        else:
            context["InterestedNessMetric"], context["MalpracticeMetric"], context["distractedNess"] = calcMetrics() # (x,x)
    return render(request, 'proctorservice/index.html', context)


def Home(request):
    return render(request, 'proctorservice/Home.html')

def livefe(request):
    global cam
    try:
        results = gen(cam)
        return JsonResponse(results)
    except:
        return JsonResponse({"NA": "NA"})