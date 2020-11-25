import cv2

# img = cv2.imread('./cap_1.png')
# img = cv2.imread('./cap_2.png')
# img = cv2.imread('./cap_3.png')
img = cv2.imread('./cap_4.png')

x = 1
y = 1
w = 347
h = 73

crop = img[y:y+h, x:x+w]

cv2.imwrite('./crop_4.png', crop)
cv2.imshow('crop', crop)

cv2.waitKey()