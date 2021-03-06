import numpy as np
import matplotlib.pyplot as plt

def get_crosspt(x11,y11, x12,y12, x21,y21, x22,y22):
    if x12==x11 or x22==x21:
        print('delta x=0')
        return None
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1==m2:
        print('parallel')
        return None
    print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return cx, cy

x11, y11, x12, y12 = (8, 718, 940, 718)
x21, y21, x22, y22 = (870, 459, 200, 1045)
# x11 =np.random.randint(1,50)
# y11 =np.random.randint(1,50)
# x12 =np.random.randint(50,100)
# y12 =np.random.randint(50,100)
# x21 =np.random.randint(1,50)
# y21 =np.random.randint(50,100)
# x22 =np.random.randint(50,100)
# y22 =np.random.randint(1,50)

plt.figure()
plt.plot([x11,x12], [y11,y12], c='r')
plt.plot([x21,x22], [y21,y22], c='b')

cx, cy = get_crosspt(x11, y11, x12, y12, x21, y21, x22, y22)
# plt.plot(cx, cy, 'ro')
# plt.show()
print(cx, cy)
