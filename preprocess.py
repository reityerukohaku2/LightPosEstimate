import os
import torch
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.io as io
from pathlib import Path
import cv2
from pathlib import Path
import re

class PreProcessDataSet(torch.utils.data.Dataset):
    def __init__(self, imageSize, train = True):

        self.trainFlag = train
        #外れ値ID
        error_ids = [14313]

        #正解ラベル
        x = []
        y = []
        z = []

        #前処理の宣言
        self.FirstTransform = nn.Sequential(
            transforms.Resize((imageSize, imageSize)),
            #transforms.ConvertImageDtype(torch.float32),
            #transforms.Normalize(0.5, 0.5)
        ).cuda()

        # ここに入力データとラベルを入れる
        self.RGBimagePaths = [str(p) for p in Path("C:/Users/moko0/OneDrive/ドキュメント/repos/SyuraiRinjin/TrainData2/Photo/RGB/").glob("*.png")]

        #ファイル名ソート用
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        #ファイル名ソート(1,2,3,...23112)
        self.RGBimagePaths = sorted(self.RGBimagePaths, key=natural_keys)

        #外れ値の削
        if len(error_ids) != 0:
            for id in error_ids:
                print("before:" + str(len(self.RGBimagePaths)))
                print("del_path:" + self.RGBimagePaths[id])
                del self.RGBimagePaths[id]
                print("del:" + str(id))
                print("after:" + str(len(self.RGBimagePaths)))

        self.dataNum = len(self.RGBimagePaths)# - len(error_ids)

    def __len__(self):
        return self.dataNum

    #DataLoader使うときに呼ばれる
    def __getitem__(self, idx):
        RGBimage = io.read_image(path=self.RGBimagePaths[idx], mode=io.image.ImageReadMode.GRAY)

        #リサイズ
        GrayImage = self.FirstTransform(RGBimage)

        #numpy配列に変換
        GrayImage = np.transpose(GrayImage.numpy(), (1, 2, 0))

        #バイラテラルフィルタ
        GrayImage = cv2.cv2.ximgproc.dtFilter(GrayImage, GrayImage, 0, 32)

        #適応的ヒストグラム平坦化
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
        GrayImage = clahe.apply(GrayImage.astype(np.uint8))
        #after = cv2.equalizeHist(GrayImage.astype(np.uint8))
        #ret, after = cv2.threshold(after, 0, 255, cv2.THRESH_OTSU)
        #kernel = np.ones((3,3), np.uint8)
        #after = cv2.morphologyEx(after, cv2.MORPH_CLOSE, kernel)

        #tensorに変換
        #GrayImage = np.expand_dims(GrayImage, 0)
        out_image = torch.from_numpy(GrayImage.astype(np.float32))
        out_image = out_image / 256.0
        #out_image = self.SecondTransform(GrayImage)

        return out_image

#データセットの保存
def loop():
    dataSet = PreProcessDataSet(imageSize=200)
    dataLoader = DataLoader(dataSet, batch_size=1, shuffle=False, num_workers = os.cpu_count() , pin_memory = True)

    i = 0
    for image in dataLoader:
        torch.save(image, "dataSet/tensor_data/Gray/" + str(i) + ".pt")
        print(i)

        """
        npimg = image.numpy()
        print(npimg.shape)
        npimg = np.squeeze(npimg, 0)

        #npimg = (npimg + 1.0) / 2.0 * 256
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
        plt.show()
        plt.hist(npimg.ravel(), bins=256)
        plt.show()
        """

        i += 1

if __name__ == "__main__":
    loop()