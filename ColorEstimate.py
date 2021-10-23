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
height, width, channel = raw.shape

#raw = raw
raw_float = raw.astype(np.float_)
#raw = raw / 255

B,G,R = cv2.split(raw_float)

#0の値を極小値に置き換え(0除算回避)
eps = 2.2204e-16
R = np.where(R <= 0, eps, R)
G = np.where(G <= 0, eps, G)
B = np.where(B <= 0, eps, B)

I = R + G + B
#I = np.sqrt(I)

C = np.array([[0, 0, 1, 0, 0],
              [0, 1, 2, 1, 0],
              [1, 2, -16, 2, 1],
              [0, 1, 2, 1, 0],
              [0, 0, 1, 0, 0]])


ksize=5
# #CはLoGフィルタ
# C = np.empty([ksize,ksize],dtype='float')
# sigma = 0.5
# for y in range(0, ksize):
#     for x in range(0, ksize):
#         C[x,y] = (x**2 + y**2 - 2 * sigma**2) * math.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * math.pi * sigma**6)

#式（11）
log_i = np.log(I)
log_r = np.log(R) - log_i
log_r = cv2.filter2D(log_r, -1, C)
log_b = np.log(B) - log_i
log_b = cv2.filter2D(log_b, -1, C)
preGI = np.sqrt((log_b + log_r)**2 + 2)

#式(12)
E = 1e-4
delta_r = cv2.filter2D(R, -1, C)
delta_g = cv2.filter2D(G, -1, C)
delta_b = cv2.filter2D(B, -1, C)
del_index = np.where((delta_r <= E) & (delta_b <= E) & (delta_g <= E))
print(del_index)
GI = preGI

#式(12)で求めた除去画素を除去
max_GI = GI.max()
GI[del_index] = max_GI

#カメラノイズによる孤立したグレーピクセルを除去
GI = cv2.blur(preGI, (7, 7))

#最もグレーである可能性が高い上位0.1%のピクセルを取得
#まずは1次元配列に
flat_GI = GI.flatten(order="F")

#ソート
sorted_GI = np.argsort(flat_GI)

#下位0.1%のインデックスを保存
sorted_GI = np.delete(sorted_GI, np.s_[(int)(sorted_GI.size * 0.001):], 0)
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
plt.imshow(GI, cmap = "rainbow")
plt.colorbar()

plt.show()
