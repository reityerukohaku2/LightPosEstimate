import torch
from torch.utils.data import DataLoader, dataset
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from OutOfPlaceArtifactDataset import OOPArtDataset
from OutOfPlaceArtifactNet import OOPArtNet


##############
###~~訓練~~###
##############

def loop():
    #データセットの定義
    dataSet = OOPArtDataset()

    #データローダーの定義
    batchSize = 32
    print("cpu_count:" + str(os.cpu_count()))
    trainLoader = DataLoader(dataSet, batch_size = batchSize, shuffle=True, num_workers = os.cpu_count() - 1, pin_memory = True)

    #学習の準備
    model = OOPArtNet()      #モデルの読み込み
    model = model.cuda()
    criterion = nn.MSELoss()            #損失関数は平均2乗誤差
    torch.backends.cudnn.benchmark = False

    epoch = 200
    lr = 4e-5
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for i in range(epoch):
        model.train()
        test_loss = 0
        torch.set_grad_enabled(True)
        step = 0

        for(RGBimage, DepthImage, Rotation) in trainLoader:
            #訓練データの初期化
            RGBimage = RGBimage.cuda()
            DepthImage = DepthImage.cuda()
            Rotation = Rotation.cuda()

            #勾配の初期化
            optimizer.zero_grad()
            #print("initialized")

            #混合精度計算部
            with torch.cuda.amp.autocast():
                #学習
                out = model(RGBimage, DepthImage)
                out = out.squeeze()
                loss = criterion(out, Rotation)

            #print("AMP Clear")

            test_loss += loss

            print(step)
            step += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        del test_loss
        print("epoch: " + str(i + 1) + " / " + str(epoch))

    #訓練済みモデル保存
    torch.save(model.to("cpu").state_dict(), os.getcwd() + "/NewOOPArt200_2022.pth")

    #もうここでONNX作る
    RGBDummyImage = torch.randn(1, 3, 160, 120)
    DepthDummyImage = torch.randn(1, 1, 160, 120)
    torch.onnx.export(model, args=(RGBDummyImage, DepthDummyImage), f="C:/Users/yuma/Desktop/Fujimoto/LightPosEstimate/NewOOPArt200_2022.onnx", verbose=True)

if __name__ == "__main__":
    loop()