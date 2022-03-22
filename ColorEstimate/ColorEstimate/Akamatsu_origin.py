import cv2
import numpy as np
import math
import sys
import copy
import matplotlib.pyplot as plt


#RGBが同じピクセルほど光源の影響がでやすい ->　白またはグレーのピクセルが欲しい
#完全２５５の白は情報がなさすぎるから困る
raw = cv2.imread("images/figure2a.png")
height, width, channels = raw.shape[:3]

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
R = cv2.GaussianBlur(R, filter_size, 3)
G = cv2.GaussianBlur(G, filter_size, 3)
B = cv2.GaussianBlur(B, filter_size, 3)

#matlabコードベースのGI計算　たぶん論文のまま書いても大丈夫
#https://github.com/yanlinqian/Grayness-Index/blob/master/greypixel_kaifu/GetGreyidx.m

#多分グレーっぽさを計算
log_r = np.log(R)
Mr = cv2.GaussianBlur(log_r, filter_size, 3)

log_b = np.log(B)
Mb = cv2.GaussianBlur(log_b, filter_size, 3)

log_g = np.log(G)
Mg = cv2.GaussianBlur(log_g, filter_size, 3)

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
target_pixel_num = int(len(target_pixel) *0.05)
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
delt_r = cv2.GaussianBlur(R, filter_size, 3)
delt_g = cv2.GaussianBlur(G, filter_size, 3)
delt_b = cv2.GaussianBlur(B, filter_size, 3)

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

result = (result*255).astype(np.uint8)
target_pixel_result = (target_pixel_result*255).astype(np.uint8)
cv2.imshow("RAW", raw)
cv2.imshow("Intensity", result)
cv2.imshow("target_pixel", target)
cv2.imshow("mask", mask*255)
cv2.waitKey()
cv2.destroyAllWindows()