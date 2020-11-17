import cv2
import numpy as np

cap = cv2.VideoCapture('./1117/people_walking_1.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while(cap.isOpened()):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    try:
        cv2.imshow('Frame', frame)
        cv2.imshow('Background Substraction', fgmask)

    except:
        print('EOF')
        break

    k = cv2.waitKey(30) & 0xff

    if k == 27 :
        break

cap.release()
cv2.destroyAllWindows()