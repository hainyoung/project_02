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

        rtime = date_time_recognizer(frame, 10, 3, 31, 308)
        print(rtime)
        
        location = location_recognizer(frame, 1, 40, 36, 341)
        print(location) 

        rt_win = frame[1:34, 10:318]
        loc_win = frame[40:76, 1:342]       
        
        cv2.imshow("Frame", frame)

        cv2.imshow("rtime", rt_win)
        cv2.imshow("location", loc_win)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
    c += 1   

cap.release()
cv2.destroyAllWindows()