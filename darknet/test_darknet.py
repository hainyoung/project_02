import sys
sys.path.insert(1, 'C:/Users/user/anaconda3/Lib/site-packages/darknet/')

import darknet
print("complete")

import cv2, numpy as np
from ctypes import *
import time, datetime

# net = darknet.load_net(b"C:/darknet-master/darknet-master/build/darknet/x64/data/3classes_train_obj.cfg", 
#                        b"C:/darknet-master/darknet-master/build/darknet/x64/backup/backup1112_4600/yolov4-obj_best.weights", 0) 
# meta = darknet.load_meta(b"C:/darknet-master/darknet-master/build/darknet/x64/data/3_classes.data") 
# cap = cv2.VideoCapture("C:/darknet-master/darknet-master/build/darknet/x64/data/3classes/test_2.mp4") 


net = darknet.load_net(b"C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/yolov4-obj.cfg", 
                       b"C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/weights/yolov4-obj_best.weights", 0) 
meta = darknet.load_meta(b"C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/obj.data") 
cap = cv2.VideoCapture("C:/Users/user/anaconda3/Lib/site-packages/darknet/data/1117/suwon_test.mp4") 

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter('test.avi', fourcc, fps, (640, 480))

# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

delay = round(1000/fps)

lines = []
with open("./suwon_test.txt", "r") as file :
    for line in file :
        lines.append(line)
    loc = lines[3][:]
    loc = loc.strip()

    year = int(lines[2][1:5]) 
    month = int(lines[2][6:8])
    day = int(lines[2][9:11])
    hour = int(lines[2][12:14])
    minute = int(lines[2][15:17])
    second = int(lines[2][18:20])
    rtime = datetime.datetime(year, month, day, hour, minute, second)    


i = 0
while(cap.isOpened()):
    i += 1
    ret, image = cap.read()
    image = cv2.resize(image, dsize=(1088, 1088), interpolation=cv2.INTER_AREA)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if not ret: 
        break 
    frame = darknet.nparray_to_image(image)
    r = darknet.detect_image(net, meta, frame, thresh = .5, hier_thresh=.5, nms=.45, debug=False)

    # print(fps)
    # print(r)

    boxes = [] 

    for k in range(len(r)): 
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2) 
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height 
        mytexts = r[k][0]
        mythresh = r[k][1]
        boxes.append((x, y, w, h, mytexts, mythresh))
    # print(1)
        
    for k in range(len(boxes)): 
        x, y, w, h, texts, threshs = boxes[k] 

        top = max(0, np.floor(x + 0.5).astype(int)) 
        left = max(0, np.floor(y + 0.5).astype(int)) 
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) 
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int)) 

        obj_name = texts.decode('utf-8')

        print("Current Time : {}, Location : {}, Object found : {}, Confidence : {:.2f}".format(rtime, loc, obj_name, threshs))
        # print(x, y, w, h)

        if texts.decode('utf-8') == 'traffic cone':
            cv2.rectangle(image, (top, left), (right, bottom), (0, 0, 255), 2)
            # cv2.putText(image, texts.decode('utf-8') + '(' + str(threshs*100)[:5] + '%)', (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            cv2.putText(image, texts.decode('utf-8'), (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

        elif texts.decode('utf-8') == 'traffic bollard':
            cv2.rectangle(image, (top, left), (right, bottom), (0, 255, 0), 2)
            # cv2.putText(image, texts.decode('utf-8') + '(' + str(threshs*100)[:5] + '%)', (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            cv2.putText(image, texts.decode('utf-8'), (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

        elif texts.decode('utf-8') == 'traffic barrel':
            cv2.rectangle(image, (top, left), (right, bottom), (0, 0, 255), 2)
            # cv2.putText(image, texts.decode('utf-8') + '(' + str(threshs*100)[:5] + '%)', (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            cv2.putText(image, texts.decode('utf-8'), (top, left-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)


        # cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2) 
        # cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
        # cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
        # cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)

    cv2.imshow('frame', image) 
    # out.write(image)
    darknet.free_image(frame)

    rtime += datetime.timedelta(seconds=1.0)
    
    if cv2.waitKey(1) & 0xFF == 27: 
    # if cv2.waitKey(0) == 27: 
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
