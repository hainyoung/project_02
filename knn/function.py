import numpy as np
import cv2
import sys

def date_time_recognizer(image,x,y,h,w):
    samples_dt = np.loadtxt('./model/suwon_generalsamples.data', np.float32)
    responses_dt = np.loadtxt('./model/suwon_generalresponses.data', np.float32)
    responses_dt = responses_dt.reshape((responses_dt.size, 1))
    model_dt = cv2.ml.KNearest_create()
    model_dt.train(samples_dt, cv2.ml.ROW_SAMPLE, responses_dt)
    
    roi_dt = image[y:y+h, x:x+w]
    out_dt = np.zeros(roi_dt.shape,np.uint8)
    gray_dt = cv2.cvtColor(roi_dt,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(gray_dt, kernel, iterations=1)

    _, thresh_dt = cv2.threshold(erosion, 123,255, cv2.THRESH_BINARY)
    contours_dt,hierarchy_dt = cv2.findContours(thresh_dt,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    dt = []

    for cnt_dt in contours_dt:
        if cv2.contourArea(cnt_dt)>25:
            [x, y, w, h] = cv2.boundingRect(cnt_dt)
            
            if  h>22:
                cv2.rectangle(roi_dt,(x,y),(x+w,y+h),(0,255,0),2)
                roi_date = thresh_dt[y:y+h,x:x+w]
                roismall = cv2.resize(roi_date,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                # for each in roismall:
                retval, results, neigh_resp, dists = model_dt.findNearest(roismall, k = 1)
                string_dt= str(int((results[0][0])))
                # string_dt = str((results[0][0]))
                cv2.putText(out_dt,string_dt,(x,y+h),0,1,(0,255,0))
                # print(string_dt)
                dt.append(string_dt)

                date_time = "".join(dt)
                date_time = date_time[::-1]

                date = date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8]
                time = date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:]

                rtime = date + " " + time

    return rtime           

def location_recognizer(image, x, y, h, w):
    samples_loc = np.loadtxt('./model/suwon_generalsamples.data', np.float32)
    responses_loc = np.loadtxt('./model/suwon_generalresponses.data', np.float32)
    responses_loc = responses_loc.reshape((responses_loc.size, 1))
    model_loc = cv2.ml.KNearest_create()
    model_loc.train(samples_loc, cv2.ml.ROW_SAMPLE, responses_loc)
    
    roi_loc = image[y:y+h, x:x+w]
    out_loc = np.zeros(roi_loc.shape,np.uint8)
    gray_loc = cv2.cvtColor(roi_loc,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(gray_loc, kernel, iterations=1)

    _, thresh_loc = cv2.threshold(erosion, 123,255, cv2.THRESH_BINARY)
    contours_loc, hierarchy_loc = cv2.findContours(thresh_loc, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    loc = []

    for cnt_loc in contours_loc:
        print("Area :", cv2.contourArea(cnt_loc))
        if cv2.contourArea(cnt_loc) > 25:
            [x, y, w, h] = cv2.boundingRect(cnt_loc)
            print("height :",h)
            if  h > 22 :
                cv2.rectangle(roi_loc,(x,y),(x+w,y+h),(0,255,0),2)
                roi_loc = thresh_loc[y:y+h,x:x+w]
                roismall = cv2.resize(roi_loc,(10,10))
                roismall = roismall.reshape((1,100))
                roismall = np.float32(roismall)
                # for each in roismall:
                retval, results, neigh_resp, dists = model_loc.findNearest(roismall, k = 1)
                string_loc= str(int((results[0][0])))
                # string_dt = str((results[0][0]))
                cv2.putText(out_loc,string_loc,(x,y+h),0,1,(0,255,0))
                # print(string_dt)
                loc.append(string_loc)

                location = "".join(loc)
                location = location[::-1]

    return location          