from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
frequency = 2500
duration = 1000
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    EAR = (A+B) / (2.0 * C)
    return EAR


blink =0
count = 0
EARThresh = 0.3
EARFrame = 48
shapePredictor = 'shape_predictor_68_face_landmarks.dat'

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)


(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart: rend]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        EAR = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull],-1,(0,0,255),1)
        cv2.drawContours(frame, [rightEyeHull],-1,(0,0,255),1)

        if EAR < EARThresh:
            count += 1
            if count >= EARFrame:
                cv2.putText(frame, "DROWSINESS DETECTED" , (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
                winsound.Beep(frequency, duration)
                blink = 0
            elif count >= 1 & count < EARFrame :
                blink +=1
        else:
            count = 0
        cv2.putText(frame, "Blink counter {}".format(int(blink/10)),(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX , 0.7 ,(255,0,0),2)


    cv2.imshow('frame ', frame )
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()