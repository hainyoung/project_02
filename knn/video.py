import cv2

cap = cv2.VideoCapture('./suwon_1116_test.mp4')


while True:
    ret, frame = cap.read()


    cv2.imshow('frame', frame)

    if cv2.waitKey(0) == 27:
        break

cap.release()
cv2.destroyAllWindows()
