import cv2
import numpy as np
import math
import sys
import copy
import matplotlib.pyplot as plt

def LoGFilter(src, sigma):
    dst = src.copy()
    dst = cv2.GaussianBlur(dst, (5, 5), sigma)
    dst = cv2.Laplacian(dst, -1, ksize=5)
    return dst

#RGBが同じピクセルほど光源の影響がでやすい ->　白またはグレーのピクセルが欲しい
#完全２５５の白は情報がなさすぎるから困る
raw = cv2.imread("images/1.png")
raw = cv2.resize(raw, (500, 500))
height, width, channels = raw.shape[:3]
sigma = 0.5

R = raw[:, :, 2].astype(np.float_) /255.
G = raw[:, :, 1].astype(np.float_) /255.
B = raw[:, :, 0].astype(np.float_) /255.

#0除算回避
EPS = sys.float_info.epsilon
R[R==0] = EPS
G[G==0] = EPS
B[B==0] = EPS
filter_size = (7, 7)

#denoise
R = cv2.GaussianBlur(R, filter_size, sigma)
G = cv2.GaussianBlur(G, filter_size, sigma)
B = cv2.GaussianBlur(B, filter_size, sigma)

#matlabコードベースのGI計算　たぶん論文のまま書いても大丈夫
#https://github.com/yanlinqian/Grayness-Index/blob/master/greypixel_kaifu/GetGreyidx.m

#多分グレーっぽさを計算
log_r = np.log(R)
#Mr = cv2.GaussianBlur(log_r, filter_size, sigma)
Mr = LoGFilter(log_r, sigma)

log_b = np.log(B)
#Mb = cv2.GaussianBlur(log_b, filter_size, sigma)
Mb = LoGFilter(log_b, sigma)

log_g = np.log(G)
#Mg = cv2.GaussianBlur(log_g, filter_size, sigma)
Mg = LoGFilter(log_g, sigma)

data = [Mb, Mg, Mr]
Ds = np.std(data, 0)
Ds = Ds / (np.average(Ds))

data1 = [B,G,R]
Ps = Ds / (np.average(data1))
Greyidx = Ps / np.max(Ps)

#GI
result = cv2.blur(Greyidx, filter_size)

#GIのピクセル強度上位５％を抜き出す
target_pixel = np.resize(result, height*width)
target_pixel.sort()
target_pixel_num = int(len(target_pixel) *0.01)
target_value = target_pixel[target_pixel_num]
print(target_value)
target_pixel_result = copy.copy(result)
target_pixel_result[target_pixel_result < target_value] = -1
target_pixel_result[target_pixel_result > target_value] = 0
target_pixel_result[target_pixel_result == -1] = 1

#-------------------------------------------
#マスク作成
#空間的な手がかりのないGIを破棄
#RGBの各チャネルの周辺の値変動がないとそこら一体は同一色である
#閾値処理をかけて色制限（基本的に白背景に光源色のあったったグレーピクセルが欲しい)
#式12の実装？ 各チャネルのガウシアン後の閾値処理
sigma = 0.5
delt_r = cv2.GaussianBlur(R, filter_size, sigma)
delt_g = cv2.GaussianBlur(G, filter_size, sigma)
delt_b = cv2.GaussianBlur(B, filter_size, sigma)
# delt_r = LoGFilter(R, sigma)
# delt_g = LoGFilter(G, sigma)
# delt_b = LoGFilter(B, sigma)

#要調整
delta_threshold = 0.6
delt_r = cv2.threshold(delt_r, delta_threshold, 1, cv2.THRESH_BINARY)[1]
delt_g = cv2.threshold(delt_g, delta_threshold, 1, cv2.THRESH_BINARY)[1]
delt_b = cv2.threshold(delt_b, delta_threshold, 1, cv2.THRESH_BINARY)[1]
mask = delt_r + delt_g + delt_b
mask[mask < 3] = 0
mask[mask == 3] = 1
#mask 白に近い色でのっぺりとした空間

#-------------------------------------------
#GI強度の上位ピクセルの位置と白っぽくてのっぺりとした空間を照らし合わす
mask = cv2.bitwise_and(target_pixel_result, mask)
target = copy.copy(raw)
target[mask==0] = 0

color = np.zeros((200, 200, 3), np.float32)
color[:, :, 0] = R[mask != 0].sum() / mask.sum()
color[:, :, 1] = G[mask != 0].sum() / mask.sum()
color[:, :, 2] = B[mask != 0].sum() / mask.sum()

hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
hsv_color[:, :, 2] = 1
result_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)

result = (result*255).astype(np.uint8)
target_pixel_result = (target_pixel_result*255).astype(np.uint8)

plt.figure()
plt.title("RAW")
plt.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))

plt.figure()
plt.title("Intensity")
plt.imshow(result)

plt.figure()
plt.title("target_pixel")
plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))

plt.figure()
plt.title("mask")
plt.imshow(mask*255)

plt.figure()
plt.title("color")
plt.imshow(color)

plt.figure()
plt.title("result_color")
plt.imshow(cv2.cvtColor(result_color, cv2.COLOR_BGR2RGB))
plt.show()

# cv2.imshow("RAW", raw)
# cv2.imshow("Intensity", result)
# cv2.imshow("target_pixel", target)
# cv2.imshow("mask", mask*255)
# cv2.imshow("color", color)
# cv2.imshow("result_color", result_color)
# cv2.waitKey()
# cv2.destroyAllWindows()