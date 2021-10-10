import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

#最大値を255，最小値を0にする
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    result = result * 255
    return result

raw = cv2.imread("images/figure2a.png")
raw = raw
raw = raw.astype(np.float_)
raw = raw / 255

B,G,R = cv2.split(raw)

#7*7でRGB平均化
R = cv2.blur(R, (7, 7))
G = cv2.blur(G, (7, 7))
B = cv2.blur(B, (7, 7))

np.where(R == 0, 0.0001, R)
np.where(G == 0, 0.0001, G)
np.where(B == 0, 0.0001, B)

I = R + G + B
#I = np.sqrt(I)

'''
C = np.array([[0, 0, 1, 0, 0],
              [0, 1, 2, 1, 0],
              [1, 2, -16, 2, 1],
              [0, 1, 2, 1, 0],
              [0, 0, 1, 0, 0]])
'''

ksize=5
C=np.empty([ksize,ksize],dtype='float')
sigma=0.3*(ksize/2-1)+0.8
for y in range(int(-ksize/2),int(ksize/2)):
    for x in range(int(-ksize/2),int(ksize/2)):
        C[x,y]=(x**2+y**2-2*(sigma**2))/(2*3.14*sigma**6)*math.exp(-(x**2+y**2)/(2*sigma**2))
print(C)

#mask low contrast pixels
delta_r = cv2.filter2D(R, -1, C)
delta_g = cv2.filter2D(G, -1, C)
delta_b = cv2.filter2D(B, -1, C)
np.where(R == delta_r, 255, R)
np.where(G == delta_g, 255, G)
np.where(B == delta_b, 255, B)


log_i = np.log(I)

log_r = np.log(R) - log_i
log_r = cv2.filter2D(log_r, -1, C)
#log_r = cv2.GaussianBlur(log_r, (5, 5), 3)
#log_r = cv2.Laplacian(log_r, cv2.CV_32F, ksize=5)

log_b = np.log(B) - log_i
log_b = cv2.filter2D(log_b, -1, C)
#log_b = cv2.GaussianBlur(log_b, (5, 5), 3)
#log_b = cv2.Laplacian(log_b, cv2.CV_32F, ksize=5)

preGI = np.sqrt(log_b**2 + log_r**2)

ranked = preGI.flatten(preGI)
print(ranked)

"""
R = min_max(R).astype(np.uint8)
G = min_max(G).astype(np.uint8)
B = min_max(B).astype(np.uint8)
I = min_max(I).astype(np.uint8)

log_b = (min_max(log_b)).astype(np.uint8)
log_r = (min_max(log_r)).astype(np.uint8)
preGI = (min_max(preGI)).astype(np.uint8)

raw = min_max(raw).astype(np.uint8)
cv2.imshow("RAW", raw)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.imshow("log_r", log_r)
cv2.imshow("log_b", log_b)
cv2.imshow("Intensity", preGI)
plt.imshow(preGI, cmap=plt.cm.brg)
plt.colorbar()
plt.show()

cv2.waitKey()
"""