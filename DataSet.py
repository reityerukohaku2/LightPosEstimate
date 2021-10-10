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

        #正解ラベル
        x = []
        y = []
        z = []

        path = "Photo/Train.csv"   #csvファイルのパスを指定する
        label = pd.read_csv(filepath_or_buffer=path, encoding="UTF-8", sep=",", index_col=0, header=None, skiprows=2)

        #外れ値ID(0スタート)
        error_ids = list(label[label.isnull().any(axis=1)].index)
        print(error_ids)

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
        #self.RGBimagePaths = [str(p) for p in Path("C:/Users/moko0/OneDrive/ドキュメント/repos/SyuraiRinjin/TrainData2/Photo/RGB/").glob("*.png")]
        #self.DepthImagePaths = [str(p) for p in Path("C:/Users/moko0/OneDrive/ドキュメント/repos/SyuraiRinjin/TrainData2/Photo/Depth/").glob("*.png")]
        self.RGBimagePaths = [str(p) for p in Path("Photo/tensor_data/Gray/").glob("*.pt")]
        self.DepthImagePaths = [str(p) for p in Path("Photo/tensor_data/Depth/").glob("*.pt")]

        #ファイル名ソート用
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]

        #ファイル名ソート(1,2,3,...23112)
        self.RGBimagePaths = sorted(self.RGBimagePaths, key=natural_keys)
        self.DepthImagePaths = sorted(self.DepthImagePaths, key=natural_keys)

        #外れ値の削
        del_num = 0
        for i in range(0, len(label)):
            if i not in error_ids:
                x.append(float(label.iloc[i, 0]))
                y.append(float(label.iloc[i, 1]))
                z.append(float(label.iloc[i, 2]))
            else:
                del self.RGBimagePaths[i - del_num]
                del_num += 1

        self.x = x
        self.y = y
        self.z = z
        self.dataNum = len(x)

    def __len__(self):
        return self.dataNum

    #DataLoader使うときに呼ばれる
    def __getitem__(self, idx):

        #RGBimage = torch.from_numpy(np.load(file=self.RGBimagePaths[idx]))
        #DepthImage = torch.from_numpy(np.load(file=self.DepthImagePaths[idx]))
        out_RGBimage = torch.load(self.RGBimagePaths[idx])
        out_DepthImage = torch.load(self.DepthImagePaths[idx])
        out_DepthImage = torch.squeeze(out_DepthImage, 0)
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
