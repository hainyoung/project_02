import numpy as np
import cv2

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)

# detecting range for real
# pt1 = np.array([[(7, 1088), (7,435), (944,435), (1088, 601), (1088, 1088)]], dtype=np.int32)

# detecting range for show
# pt1 = np.array([[(8, 1045), (8,718), (840,459), (940,459), (940, 1045), (8, 1045)]], dtype=np.int32)
pt2 = np.array([[(870, 459), (200, 1045)]])
pt3 = np.array([[(425, 588), (940, 588)]])
pt4 = np.array([[(8, 718), (940, 718)]])

image = cv2.imread('image.jpg', -1)

img = cv2.polylines(image, [pt2, pt3, pt4], True, red, 2)
img = cv2.line(image, (722, 588), (722, 588), blue, 10)
img = cv2.line(image, (573, 718), (573, 718), blue, 10)
img = cv2.line(image, (193, 717), (193, 717), white, 1)

tmp = np.array([[(145, 848)]])

# roi_corners = np.array([[(8, 1014), (8,718), (860,459), (1087,452), (1084, 1015)]], dtype=np.int32)
area_0 = np.array([[(8, 1045), (8, 718), (573, 718), (200, 1045), (8, 1045)]])
area_2 = np.array([[(8, 718), (425, 588), (722, 588), (573, 718), (8, 718)]])


# img = cv2.polylines(image, [area_0, area_2], True, green, 1)

# if tmp in area_0:
#     print("car")
# else :
#     print("nothing")

cv2.imshow('polylines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()