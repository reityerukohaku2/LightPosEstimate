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

def LoGFilter(src, ksize, sigma):
    dst = cv2.GaussianBlur(src, ksize, sigma)
    #dst = cv2.Laplacian(dst, -1, ksize)
    return dst

raw = cv2.imread("images/15815.png")
height, width, channel = raw.shape

#raw = raw
raw_float = raw.astype(np.float32)
#raw = raw / 255

B,G,R = cv2.split(raw_float)

#0の値を極小値に置き換え(0除算回避)
eps = 2.2204e-16
R = np.where(R <= 0, eps, R)
G = np.where(G <= 0, eps, G)
B = np.where(B <= 0, eps, B)

#I = np.sqrt(I)

# C = np.array([[0, 0, 1, 0, 0],
#               [0, 1, 2, 1, 0],
#               [1, 2, -16, 2, 1],
#               [0, 1, 2, 1, 0],
#               [0, 0, 1, 0, 0]])


ksize=5
# #CはLoGフィルタ
# C = np.empty([ksize,ksize],dtype='float')
# sigma = 0.5
# for y in range(0, ksize):
#     for x in range(0, ksize):
#         C[x,y] = (x**2 + y**2 - 2 * sigma**2) * math.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * math.pi * sigma**6)

#式（11）
# log_i = np.log(I)
# log_r = np.log(R) - log_i
# log_r = LoGFilter(log_r, (5, 5), .5)
# log_b = np.log(B) - log_i
# log_b = LoGFilter(log_b, (5, 5), .5)
# preGI = np.sqrt((log_b + log_r)**2 + 2)

#MatLab準拠で書いてみる
#https://github.com/yanlinqian/Grayness-Index/blob/master/greypixel_kaifu/GPconstancy.m
ksize = (5, 5)
sigma = .5
Mr = LoGFilter(np.log(R), ksize, sigma)
Mg = LoGFilter(np.log(G), ksize, sigma)
Mb = LoGFilter(np.log(B), ksize, sigma)

data = [Mr, Mg, Mb]
data[0][np.where(Mr == 0.1)] = eps
data[1][np.where(Mg == 0.1)] = eps
data[2][np.where(Mb == 0.1)] = eps

data_normed = data / np.linalg.norm(data)
gt = np.linalg.norm(np.ones(np.shape(data)))
dot = np.dot(data_normed, gt)
acos = np.arccos(dot)
angular_error = np.real(acos)

lumi = [R, G, B]

mink_norm = 1
lumi_factor = np.power(sum(np.power(lumi, mink_norm), 2), 1 / mink_norm) + eps
sort_lumi = np.sort(lumi_factor)
sortid_lumi = np.argsort(lumi_factor)
threshold_lumi = np.round(0.05 * len(lumi_factor))

Greyidx_angular = angular_error
angular_error = angular_error / lumi_factor

GMax = np.nanmax(Greyidx_angular)
Greyidx = Greyidx_angular / GMax
del_idx = np.where((Mr < eps) & (Mg < eps) & (Mb < eps))
Greyidx[del_idx] = Greyidx.max()
Greyidx_angular[del_idx] = Greyidx_angular.max()

Greyidx = cv2.blur(Greyidx, (7, 7))
Greyidx_angular = cv2.blur(Greyidx_angular, (7, 7))

# Ds = np.std(data, 0)
# Ds = Ds / (np.mean(data, 0) + eps)

# data1 = [R, G, B]
# Ps = Ds / (np.mean(data1, 0) + eps)

# Greyidx = Ps / (Ps.max() + eps)
# E = 1e-4
# del_idx = np.where((Mr < E) & (Mg < E) & (Mb < E))
# Greyidx[del_idx] = Greyidx.max()
# Greyidx = cv2.blur(Greyidx, (7, 7))

"""
#式(12)

# delta_r = cv2.filter2D(R, -1, C)
# delta_g = cv2.filter2D(G, -1, C)
# delta_b = cv2.filter2D(B, -1, C)
delta_r = LoGFilter(R, (5, 5), .5)
delta_g = LoGFilter(G, (5, 5), .5)
delta_b = LoGFilter(B, (5, 5), .5)
del_index = np.where((delta_r <= E) & (delta_b <= E) & (delta_g <= E))
print(del_index)
GI = preGI

#式(12)で求めた除去画素を除去
max_GI = GI.max()
GI[del_index] = max_GI

#カメラノイズによる孤立したグレーピクセルを除去
GI = cv2.blur(preGI, (7, 7))
"""

#最もグレーである可能性が高い上位0.1%のピクセルを取得
#まずは1次元配列に
flat_GI = Greyidx.flatten(order="F")

#ソート
sorted_GI = np.argsort(flat_GI)

#下位0.1%のインデックスを保存
sorted_GI = np.delete(sorted_GI, np.s_[(int)(sorted_GI.size * 0.01):], 0)
print(sorted_GI.size)
#most_gray_indexのマスク画像作る(デバッグ用)
#平均色を求める
avg = np.zeros(3, np.float32)
raw_mask = np.zeros((height, width, channel), np.uint8)
for i in sorted_GI:
    raw_mask[i % height][i // height] = raw[i % height][i // height]
    avg += raw[i % height][i // height]

avg /= sorted_GI.size
color = np.zeros((200, 200, 3), np.float32)
color[:][:] = avg

#画像出力
plt.figure()
plt.imshow(color.astype(np.uint))

plt.figure()
raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
plt.imshow(raw)

plt.figure()
plt.title("raw_mask")
raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2RGB)
plt.imshow(raw_mask)

plt.figure()
plt.imshow(Greyidx, cmap = "rainbow")
plt.colorbar()

plt.figure()
plt.imshow(Greyidx_angular, cmap = "rainbow")
plt.colorbar()

plt.show()
