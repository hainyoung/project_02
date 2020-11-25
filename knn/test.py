import numpy as np
import cv2
import sys

from function import *


cap = cv2.VideoCapture('./suwon_1116_test.mp4')

c = 0

while(cap.isOpened()):

    ret, frame = cap.read()

    if not ret :
        break

    if c % 1 == 0 :

        rtime = date_time_recognizer(frame, 10, 1, 33, 308)
        print(rtime)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
    c += 1   

cap.release()
cv2.destroyAllWindows()