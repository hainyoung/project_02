import cv2
import numpy as np

cap = cv2.VideoCapture('./1117/people_walking_1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernelOp = np.ones((3,3), np.uint8)
kernelCl = np.ones((11,11), np.uint8)

while(cap.isOpened()):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    try:
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

    except:
        print('EOF')
        break

    contours0, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours0:
        cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3, 8)

    cv2.imshow('Frame', frame)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()