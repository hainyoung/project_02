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

        # dt = date_time_recognizer(frame, 1, 10, 308, 33)

        samples_dt = np.loadtxt('./model/suwon_generalsamples.data', np.float32)
        responses_dt = np.loadtxt('./model/suwon_generalresponses.data', np.float32)
        responses_dt = responses_dt.reshape(responses_dt.size, 1)

        model_dt = cv2.ml.KNearest_create()
        model_dt.train(samples_dt, cv2.ml.ROW_SAMPLE, responses_dt)

        roi_dt = frame[1:34, 10:318]

        out_dt = np.zeros(roi_dt.shape, np.uint8)
        gray_dt = cv2.cvtColor(roi_dt, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2,2), np.uint8)
        erosion = cv2.erode(gray_dt, kernel, iterations = 1)

        _, src_dt = cv2.threshold(erosion, 123, 255, cv2.THRESH_BINARY)

        contours_dt, hierarchy_dt = cv2.findContours(src_dt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        dt = []
        for cnt_dt in contours_dt:
            if cv2.contourArea(cnt_dt)>25:           
                # x, y, w, h = 28, 51, 873-28, 131-51
                [x, y, w, h] = cv2.boundingRect(cnt_dt)
                if  h>22:
                    cv2.rectangle(roi_dt,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_dt_1 = src_dt[y:y+h,x:x+w]
                    roismall = cv2.resize(roi_dt_1,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    # for each in roismall:
                    retval, results, neigh_resp, dists = model_dt.findNearest(roismall, k = 1)
                    string_dt= str(int((results[0][0])))
                    # string_dt = str((results[0][0]))
                    cv2.putText(out_dt,string_dt,(x,y+h),0,1,(0,255,0))
                    # print(string_dt)
                    dt.append(string_dt)

        # print(dt)
        date_time = "".join(dt)
        date_time = date_time[::-1]
        date = date_time[:8]
        time = date_time[8:]

        print(date[:4]+"-"+date[4:6]+"-"+date[6:], end = ' ', sep = ' ')
        print(time[:2]+":"+time[2:4]+":"+time[4:])

        cv2.imshow("Frame", frame)
        cv2.imshow("E", erosion)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
    c += 1   

cap.release()
cv2.destroyAllWindows()