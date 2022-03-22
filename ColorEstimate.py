import cv2
from matplotlib.colors import rgb_to_hsv
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

def LoGFilter(src, ksize, sigma):
    # C = np.array([[0, 0, 1, 0, 0],
    #               [0, 1, 2, 1, 0],
    #               [1, 2, -16, 2, 1],
    #               [0, 1, 2, 1, 0],
    #               [0, 0, 1, 0, 0]])
    dst = src
    dst = cv2.GaussianBlur(dst, ksize, sigma)
    #dst = cv2.Laplacian(dst, -1, ksize)


    # dst = cv2.filter2D(src, -1, C)
    return dst

raw = cv2.imread("images/figure2a.png")
#raw = cv2.resize(raw, (500, 500))
height, width, channel = raw.shape

#raw = raw
#raw_float = np.ones((height, width, channel), np.float32)
raw_float = raw.astype(np.float32) / 255
#raw = raw / 255

B,G,R = cv2.split(raw_float)
#R /= 255
#G /= 255
#B /= 255

#I = np.sqrt(I)


ksize = (5, 5)
sigma = 0.5
# #CはLoGフィルタ
# C = np.empty([ksize,ksize],dtype='float')
# sigma = 0.5
# for y in range(0, ksize):
#     for x in range(0, ksize):
#         C[x,y] = (x**2 + y**2 - 2 * sigma**2) * math.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * math.pi * sigma**6)

#デノイズ
''' R = cv2.blur(R, (7, 7))
G = cv2.blur(G, (7, 7))
B = cv2.blur(B, (7, 7))

#0の値を極小値に置き換え(0除算回避)
eps = 2.2204e-16
R = np.where(R <= 0, eps, R)
G = np.where(G <= 0, eps, G)
B = np.where(B <= 0, eps, B)

#式（11）
I = R + G + B
log_i = np.log(I)
log_r = np.log(R) - log_i
log_r = LoGFilter(log_r, ksize, sigma)
log_b = np.log(B) - log_i
log_b = LoGFilter(log_b, ksize, sigma)
preGI = np.sqrt((log_b + log_r  )**2 + 2)

#式(12)
#コントラストが低い画素をマスクする
E = 1e-4
mask = np.ones((height, width, 1), np.uint8)
CR = LoGFilter(R, ksize, sigma)
CG = LoGFilter(G, ksize, sigma)
CB = LoGFilter(B, ksize, sigma)
GI = preGI
GIMax = GI.max()
GI[np.where((CR <= E) | (CG <= E) | (CB <= E))] = GIMax

#(7, 7)で平均化
GI = cv2.blur(GI, (7, 7)) '''

#----------------------------------------------------------------------------------------------------------------
#MatLab準拠で書いてみる
#https://github.com/yanlinqian/Grayness-Index/blob/master/runme.m

#極端に明るいピクセルと暗いピクセルをマスクする
''' mask = np.zeros((height, width, 1))
mask[np.where(raw_float[:, :, 0] >= 0.95)] = 1
mask[np.where(raw_float[:, :, 1] >= 0.95)] = 1
mask[np.where(raw_float[:, :, 2] >= 0.95)] = 1
test = np.zeros((height, width, channel), np.float32)
test = raw_float[:, :, 0] + raw_float[:, :, 1] + raw_float[:, :, 2]

mask[np.where(raw_float[:, :, 0] + raw_float[:, :, 1] + raw_float[:, :, 2] <= 0.0315)] = 1

plt.imshow(mask)
plt.show()

ksize = (5, 5)
sigma = 5
Mr = LoGFilter(np.log(R), ksize, sigma).flatten()
Mg = LoGFilter(np.log(G), ksize, sigma).flatten()
Mb = LoGFilter(np.log(B), ksize, sigma).flatten()

data = np.array([Mr, Mg, Mb])
data[0][np.where(Mr == 0)] = eps
data[1][np.where(Mg == 0)] = eps
data[2][np.where(Mb == 0)] = eps
data = data.transpose(1, 0)

gt = np.ones(data.shape[1], np.float32) / np.linalg.norm(np.ones(data.shape[1], np.float32))

#行ごとに分割
data_split = np.split(data, data.shape[0])

dot = []
for i in range(data.shape[0]):
    data_normed = np.abs(data_split[i] / np.linalg.norm(data_split[i]))
    dot.append(np.inner(data_normed, gt))

acos = np.arccos(dot)
angular_error = np.real(acos)

#angular_error = angular_error / lumi_factor
Greyidx_angular = np.reshape(angular_error, (height, width))

ReshapeMr = np.reshape(Mr, (height, width))
ReshapeMg = np.reshape(Mg, (height, width))
ReshapeMb = np.reshape(Mb, (height, width))

del_idx = np.where((ReshapeMr < eps) & (ReshapeMg < eps) & (ReshapeMb < eps))
Greyidx_angular[del_idx] = np.nanmax(Greyidx_angular)

Greyidx_angular = cv2.blur(Greyidx_angular, (7, 7)) '''

#最もグレーである可能性が高い上位1%のピクセルを取得
#まずは1次元配列にしてソート
tt = np.sort(GI.flatten())
Gidx = np.ones((height, width))
numGPs = int(GI.size * 0.001)
Gidx[np.where(GI <= tt[numGPs])] = 0

#下位1%のインデックスを保存
RR = np.sum(R[np.where(Gidx == 0)]) / numGPs
GG = np.sum(G[np.where(Gidx == 0)]) / numGPs
BB = np.sum(B[np.where(Gidx == 0)]) / numGPs

colorImg = np.zeros((200, 200, 3), np.float32)
colorImg[:, :, 0].fill(RR)
colorImg[:, :, 1].fill(GG)
colorImg[:, :, 2].fill(BB)
#HSVImage = cv2.cvtColor(colorImg, cv2.COLOR_RGB2HSV)

#輝度補正
#print(HSVImage)
#HSVImage[:, :, 2] = 1

#colorImg = cv2.cvtColor(HSVImage, cv2.COLOR_HSV2RGB)

#画像出力
plt.figure()
plt.title("Color")
plt.imshow(colorImg)

plt.figure()
raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
plt.title("raw")
plt.imshow(raw)

plt.figure()
plt.title("mask")
plt.imshow(Gidx, cmap="Greys")
plt.colorbar()

plt.figure()
plt.title("Greyidx")
plt.imshow(GI, cmap = "rainbow")
plt.colorbar()

plt.show()
print(0)