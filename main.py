import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from DataSet import MyDataSet
import GrayNet as Net

#学習率スケジューラ
class LearningRateScheduler:
    def __init__(self, base_lr: float, max_epoch: int, power=0.9):
        self.max_epoch = max_epoch
        self.base_lr = base_lr
        self.power = power

    def __call__(self, epoch: int):
        if epoch < 5:
            return self.base_lr * (epoch + 1) / 5
        return (1 - max(epoch - 5, 1) / max(self.max_epoch - 5, 1)) ** self.power * self.base_lr

##############
###~~訓練~~###
##############
#model.train()
#scaler = torch.cuda.amp.GradScaler()

def loop():
    #データセットの定義
    dataSet = MyDataSet(imageSize=200)

    data_size = len(dataSet)
    train_size = int(data_size * 0.8)
    val_size = data_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataSet, (train_size, val_size))

    #データローダーの定義
    batchSize = 64
    print("cpu_count:" + str(os.cpu_count()))
    trainLoader = DataLoader(train_dataset, batch_size = batchSize, shuffle=True, num_workers = os.cpu_count() - 1, pin_memory = True)
    valLoader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = os.cpu_count() - 1, pin_memory = True)

    #学習の準備
    model = Net.MyResNet50(1)           #モデルの読み込み
    model = model.cuda()
    criterion = nn.MSELoss()            #損失関数は平均2乗誤差
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

    epoch = 100
    lr = 1e-2
    lr_scheduler = LearningRateScheduler(lr, epoch)
    optimizer = optim.AdamW(model.parameters(), lr=lr) #重み減衰追加した
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir="./logs/test")

    for i in range(epoch):
        model.train()
        sum_loss = 0
        torch.set_grad_enabled(True)
        step = 0

        for(RGBimage, DepthImage, pos) in trainLoader:
            #訓練データの初期化
            RGBimage = RGBimage.cuda()
            DepthImage = DepthImage.cuda()
            pos = pos.cuda()

            #勾配の初期化
            optimizer.zero_grad()
            print("initialized")

            #混合精度計算部
            with torch.cuda.amp.autocast():
                #学習
                out = model(RGBimage, DepthImage)
                out = F.normalize(out, 2, 1)    #単位ベクトル化
                out = out.squeeze()
                loss = criterion(out, pos)

            print("AMP Clear")

            sum_loss += loss

            print(step)
            step += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        #lossをtensorBoardに書き込み
        sum_loss /= train_size
        writer.add_scalars("losses", { "loss_direction_repaired2": sum_loss, } , i)
        #writer.add_histogram("fc_3.weights", model.fc_3.weight, i)
        #writer.add_histogram("fc_3.bias", model.fc_3.bias, i)

        del sum_loss

        #バリデーションロス計算
        model.eval()
        sum_loss = 0
        val_step = 1
        torch.set_grad_enabled(False)
        for (RGBimage, DepthImage, pos) in valLoader:
            print("ValStep:" + str(val_step) + " / " + str(val_size))
            val_step += 1
            RGBimage = RGBimage.cuda()
            DepthImage = DepthImage.cuda()
            pos = pos.cuda()

            out= model(RGBimage, DepthImage)
            out = F.normalize(out, 2, 1)    #単位ベクトル化
            out = out.squeeze()

            loss = criterion(out, pos)
            sum_loss += loss
            del RGBimage, DepthImage, pos

        #val_loss書き込み
        sum_loss /= val_size
        writer.add_scalars("losses", {"valLoss_direction_repaired2": sum_loss,} ,
                                        i)

        #学習率書き込み
        #writer.add_scalar("Lerning Rate_lr1e-2", np.array(scheduler.get_lr()), i)
        print("lr=" + str(scheduler.get_lr()))

        print("epoch: " + str(i + 1) + " / " + str(epoch))

        """
            学習率スケジューラ
        """
        scheduler.step()

    #訓練済みモデル保存
    torch.save(model.to("cpu").state_dict(), os.getcwd() + "/direction_repaired2.pth")

    #もうここでONNX作る
    RGBDummyImage = torch.randn(1, 1, 200, 200)
    DepthDummyImage = torch.randn(1, 1, 200, 200)
    torch.onnx.export(model, args=(RGBDummyImage, DepthDummyImage), f="C:/Users/yuma/Desktop/Fujimoto/LightPosEstimate/direction_repaired.onnx2", verbose=True)

if __name__ == "__main__":
    loop()