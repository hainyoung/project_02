import cv2
 
# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
vidcap = cv2.VideoCapture('./test_2.mp4')
 
count = 0
 
while(vidcap.isOpened()):
    ret, image = vidcap.read()
    if ret :
        if(int(vidcap.get(1)) % 20 == 0):
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            cv2.imwrite("./capture/frame%d.jpg" % count, image)
            print('Saved frame%d.jpg' % count)
            count += 1
    else :
        break

 
vidcap.release()
cv2.destroyAllWindows()
