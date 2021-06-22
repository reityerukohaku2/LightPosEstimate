import torch.utils.data
from pathlib import Path
#from PIL import Image
import pandas as pd
import torch
import re

#PyTorchのDetasetクラスを継承
#DataLoaderにこのクラスを渡すため
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, imageSize, train = True):

        self.trainFlag = train
        #外れ値ID
        error_ids = [14313]

        #正解ラベル
        x = []
        y = []
        z = []

        path = ""   #csvファイルのパスを指定する
        label = pd.read_csv(filepath_or_buffer=path, encoding="UTF-8", sep=",", index_col=0)

        #前処理の宣言
        """
        self.RGBtransform = nn.Sequential(
            #transforms.ToPILImage(),
            transforms.Resize((imageSize, imageSize)),
            #transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ).cuda()
        self.DepthTransform = nn.Sequential(
            #transforms.ToPILImage(),
            transforms.Resize((imageSize, imageSize)),
            #transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(0.5, 0.5),
        ).cuda()
        """


        # ここに入力データとラベルを入れる

        self.RGBimagePaths = [str(p) for p in Path("dataSet/tensor_data/RGB/").glob("*.pt")]
        self.DepthImagePaths = [str(p) for p in Path("dataSet/tensor_data/Depth/").glob("*.pt")]

        #ファイル名ソート用
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        #ファイル名ソート(1,2,3,...23112)
        self.RGBimagePaths = sorted(self.RGBimagePaths, key=natural_keys)
        self.DepthImagePaths = sorted(self.DepthImagePaths, key=natural_keys)

        #外れ値の削
        """
        if len(error_ids) != 0:
            for id in error_ids:
                print("before:" + str(len(self.RGBimagePaths)))
                print("del_path:" + self.RGBimagePaths[id])
                print("del_path:" + self.DepthImagePaths[id])
                del self.RGBimagePaths[id]
                del self.DepthImagePaths[id]
                print("del:" + str(id))
                print("after:" + str(len(self.RGBimagePaths)))
        """

        self.dataNum = len(self.RGBimagePaths)# - len(error_ids)

        for i in range(1, self.dataNum + len(error_ids) + 1):
            for id in error_ids:
                if (i - 1) != id:
                    x.append(float(label.iloc[i, 0]))
                    y.append(float(label.iloc[i, 1]))
                    z.append(float(label.iloc[i, 2]))

        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.dataNum

    #DataLoader使うときに呼ばれる
    def __getitem__(self, idx):

        #RGBimage = torch.from_numpy(np.load(file=self.RGBimagePaths[idx]))
        #DepthImage = torch.from_numpy(np.load(file=self.DepthImagePaths[idx]))
        out_RGBimage = torch.load(self.RGBimagePaths[idx])
        out_DepthImage = torch.load(self.DepthImagePaths[idx])
        #RGBimage = io.read_image(path=self.RGBimagePaths[idx], mode=io.image.ImageReadMode.RGB)
        #DepthImage = io.read_image(path=self.DepthImagePaths[idx], mode=io.image.ImageReadMode.GRAY)
        #RGBimage = Image.open(self.RGBimagePaths[idx]).convert("RGB")
        #DepthImage = Image.open(self.DepthImagePaths[idx]).convert("L")

        """
        if self.RGBtransform:
            out_RGBimage = self.RGBtransform(RGBimage)

        if self.DepthTransform:
            out_DepthImage = self.DepthTransform(DepthImage)
        """

        out_x = self.x[idx]
        out_y = self.y[idx]
        out_z = self.z[idx]

        return out_RGBimage, out_DepthImage, out_x, out_y, out_z
