"""
    Title: Eye Blink Detector
    Author:Ashish Kore
    Language: Python
    Requirements:
    Python version-> 3 or later
    Python Packages or Modules or libraries->	1. scipy 
                                    		2. imutils
						3. numpy
						4. argparse
						5. time
						6. dlib
						7. cv2
    Additional Files->      1. shape_predictor_68_face_landmarks.dat
"""

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
"""
This function computes and returns the eye aspect ratio
"""
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

detector = dlib.get_frontal_face_detector()	#creating detector object
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")	#creating predictor object	
vs = FileVideoStream("video.mp4").start()	#loading video named as video.mp4
fileStream = True

time.sleep(1.0)

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.0
EYE_AR_CONSEC_FRAMES = 3
 
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# grab the frame from the threaded video file stream, resize it, and convert it to grayscale channels)
frame = vs.read()
frame = imutils.resize(frame, width=450)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale frame
rects = detector(gray, 0)

# loop over the face detections
for rect in rects:
  # determine the facial landmarks for the face region, then
  # convert the facial landmark (x, y)-coordinates to a NumPy
  # array
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)
  # grab the indexes of the facial landmarks for the left and right eye, respectively
  (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
  (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	
  # extract the left and right eye coordinates, then use the
  # coordinates to compute the eye aspect ratio for both eyes
  leftEye = shape[lStart:lEnd]
  rightEye = shape[rStart:rEnd]
  
  leftEAR = eye_aspect_ratio(leftEye)
  rightEAR = eye_aspect_ratio(rightEye)
  
  # average the eye aspect ratio together for both eyes
  ear = (leftEAR + rightEAR) / 2.0

  # setting threshold value according to the EAR value
  if ear>0.3:
    EYE_AR_THRESH =0.3
  elif ear>=0.2 and ear<=0.3:
    EYE_AR_THRESH =0.2
    
# performing above task till the video gets ended
# loop over frames from the video stream
while True:
  # if this is a file video stream, then we need to check if
  # there any more frames left in the buffer to process
  if fileStream and not vs.more():
    break
  
  try:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
      shape = predictor(gray, rect)
      shape = face_utils.shape_to_np(shape)
      (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
      (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
      leftEye = shape[lStart:lEnd]
      rightEye = shape[rStart:rEnd]
      leftEAR = eye_aspect_ratio(leftEye)
      rightEAR = eye_aspect_ratio(rightEye)

      ear = (leftEAR + rightEAR) / 2.0

      leftEyeHull = cv2.convexHull(leftEye)
      rightEyeHull = cv2.convexHull(rightEye)
      cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
      cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

      if ear < EYE_AR_THRESH:
        COUNTER += 1
      else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
          TOTAL += 1

        COUNTER = 0
      cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      cv2_imshow(frame)
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break
  except:
    pass

print("Number of Eye blinks in the video:",TOTAL)
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
