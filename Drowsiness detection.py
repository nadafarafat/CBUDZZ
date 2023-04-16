# Download the helper library from https://www.twilio.com/docs/python/install
# these are l
import os
from twilio.rest import Client
import urllib.request
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import numpy as np
import dlib
import cv2
import pyttsx3
import winsound
frequency = 2500
duration = 1000


# Define a function to calculate eye aspect ratio (EAR)
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

count = 0
earThresh = 0.3 #distance between vertical eye coordinate Threshold 
earFrames = 48 #consecutive frames for eye closure
shapePredictor = "shape_predictor_68_face_landmarks.dat"
cam = cv2.VideoCapture(0)
#while taking the pictures from the ip webcam 
#url = 'http://192.168.99.75:8080/video'
#cam = cv2.VideoCapture(url)
print(cam.read())
# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)    # Speed of speech (words per minute)
engine.setProperty('volume', 1)  # Volume of speech (0 to 1)
# Define the function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

#get the coord of left & right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    success, frame = cam.read()
    if not success:
        print("No Video Captured!")
        break
    frame = imutils.resize(frame, width=1000)
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        if ear < earThresh:
            count += 1

            if count >= earFrames:
                #if drowsiness is detected, alert the driver
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
                speak("Stay awake. Please open your eyes.")
                # Set environment variables for your credentials
                # Read more at http://twil.io/secure
                #this api make a phone call alert 

                account_sid = "ACadac79fe8c36c46eb015144045b47f4c"
                auth_token = "1de685162be3302a289ad70c1dd0d291"
                client = Client(account_sid, auth_token)

                call = client.calls.create(
                  url="http://demo.twilio.com/docs/voice.xml",
                  to="+919527728149",
                  from_="+16205429612"
                )
                print(call.sid)
                #for band vibration band is connected to the band app in mobile that enables the call alerts 
        else:
            cv2.putText(frame, "EYES ARE OPEN", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
             
            count = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break      
cam.release()
cv2.destroyAllWindows()
