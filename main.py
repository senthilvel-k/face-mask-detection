import numpy as np
import cv2
import random

face_cascade = cv2.CascadeClassifier('cascade\\haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('cascade\\haarcascade_mcs_mouth.xml')

threshold = 80  #light_threshold

font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
no_face_font_color = (255, 255, 255)
mask_font_color = (0, 255, 0)
no_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
no_face = "Face Not Detected"
weared_mask = "Mask Detected"
no_mask = "Mask Not Detected"

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img,1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, black_and_white) = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, no_face, org, font, font_scale, no_face_font_color, thickness, cv2.LINE_AA) #face not detected
    elif(len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(img, weared_mask, org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA) #for facemask with white color
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
        if(len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, mask_font_color, thickness, cv2.LINE_AA) #mask detected
        else:
            for (mx, my, mw, mh) in mouth_rects:
                if(y < my < y + h):
                    cv2.putText(img, no_mask, org, font, font_scale, no_mask_font_color, thickness, cv2.LINE_AA) #mask not detected
                    break

    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
