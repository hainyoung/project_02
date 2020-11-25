import sys

import numpy as np
import cv2

src = cv2.imread('./train_img_resize.png')

#im3 = im.copy()

gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(3,3),0)
kernel = np.ones((2,2), np.uint8)
erosion = cv2.erode(gray, kernel, iterations=1)

# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

# thresh, src_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh, src_bin = cv2.threshold(erosion, 123, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# print(contours)
print(thresh)
cv2.imshow('image', src_bin)
cv2.waitKey()


samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>25:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  28>h>25:
            cv2.rectangle(src,(x,y),(x+w,y+h),(0,0,255),2)
            roi = src_bin[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',src)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('./rtime_erode_generalsamples.data',samples)
np.savetxt('./rtime_erode_generalresponses.data',responses)
