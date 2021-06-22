import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from DataSet import MyDataSet
import Net

#学習率スケジューラ
class LearningRateScheduler:
    def __init__(self, base_lr: float, max_epoch: int, power=0.9):
        self.max_epoch = max_epoch
        self.base_lr = base_lr
        self.power = power

    def __call__(self, epoch: int):
        if epoch <= 5:
            return self.base_lr * (epoch + 1) / 5
        return (1 - max(epoch - 6, 1) / self.max_epoch) ** self.power * self.base_lr

#############
###~~訓練~~###
##############
#model.train()
#scaler = torch.cuda.amp.GradScaler()

def loop():
    #データセットの定義
    dataSet = MyDataSet(imageSize=200)

    data_size = len(dataSet)
    train_size = int(data_size * 0.8)
    val_size = int(data_size * 0.1)
    test_size = data_size - train_size - val_size

    """
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, data_size + val_size))
    test_indices = list(range(train_size + val_size, data_size))

    train_dataset = Subset(dataSet, train_indices)
    val_dataset = Subset(dataSet, val_indices)
    test_dataset = Subset(dataSet, test_indices)
    """

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataSet, (train_size, val_size, test_size))

    #データローダーの定義
    batchSize = 64
    print("cpu_count:" + str(os.cpu_count()))
    #sampler = MySampler.MySampler(batch_size=batchSize, shuffle=True, sampler=train_dataset)
    trainLoader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers = os.cpu_count(), pin_memory = True)
    valLoader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = os.cpu_count(), pin_memory = True)
    testLoader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = os.cpu_count(), pin_memory = True)

    #学習の準備
    model = Net.MyResNet50(1)           #モデルの読み込み
    model = model.cuda()
    criterion = nn.MSELoss()            #損失関数は平均2乗誤差
    torch.backends.cudnn.benchmark = True

    epoch = 60
    lr = 1e-3
    lr_scheduler = LearningRateScheduler(lr, epoch)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir="./logs/epoch60")

    for i in range(epoch):
        model.train()
        test_loss = 0
        torch.set_grad_enabled(True)
        step = 0

        for(RGBimage, DepthImage, x, y, z) in trainLoader:
            #訓練データの初期化
            #print(RGBimage.shape)
            #print(DepthImage.shape)
            RGBimage = RGBimage.cuda()
            DepthImage = DepthImage.cuda()
            #RGBimage = torch.squeeze(RGBimage, 1)
            #DepthImage = torch.squeeze(DepthImage, 1)

            x = x.cuda()
            y = y.cuda()
            z = z.cuda()

            #勾配の初期化
            optimizer.zero_grad()
            print("initialized")

            #混合精度計算部
            with torch.cuda.amp.autocast():
                #学習
                #with torch.cuda.amp.autocast():
                out_x, out_y, out_z = model(RGBimage, DepthImage)
                x = x.unsqueeze(1)
                y = y.unsqueeze(1)
                z = z.unsqueeze(1)

                loss_x = criterion(out_x.float(), x.float())
                loss_y = criterion(out_y.float(), y.float())
                loss_z = criterion(out_z.float(), z.float())

            print("AMP Clear")

            #3つの損失を適当な重みをかけて束ねる
            loss = (loss_x + loss_y + loss_z) / 3.0
            test_loss += loss
            #loss = loss.float()

            print(step)
            step += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #loss.backward()
            #optimizer.step()
            scaler.update()
            #print("update Clear")

            #del loss_x, loss_y, loss_z, RGBimage, DepthImage, x, y, z, loss

        test_loss /= test_size
        writer.add_scalars("losses", { "loss_size200_lr1e-3_batch64_pow09": test_loss, } , i)
        #writer.add_histogram("fc_x.weights", model.fc_x.weight, i)
        #writer.add_histogram("fc_y.weights", model.fc_y.weight, i)
        #writer.add_histogram("fc_z.weights", model.fc_z.weight, i)
        del test_loss

        #バリデーションロス計算
        model.eval()
        loss = 0
        val_step = 1
        torch.set_grad_enabled(False)
        for (RGBimage, DepthImage, x, y, z) in valLoader:
            print("ValStep:" + str(val_step) + " / " + str(val_size))
            val_step += 1
            RGBimage = RGBimage.cuda()
            DepthImage = DepthImage.cuda()
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()

            out_x, out_y, out_z = model(RGBimage, DepthImage)
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            z = z.unsqueeze(1)

            loss_x = criterion(out_x.float(), x.float())
            loss_y = criterion(out_y.float(), y.float())
            loss_z = criterion(out_z.float(), z.float())

            #3つの損失を適当な重みをかけて束ねる
            loss += (loss_x + loss_y + loss_z) / 3.0

            del loss_x, loss_y, loss_z, RGBimage, DepthImage, x, y, z

        loss /= val_size

        writer.add_scalars("losses", {"val-loss_size200_lr1e-3_batch64_pow09": loss,} ,
                                        i)
        writer.add_scalar("Lerning Rate", np.array(scheduler.get_lr()), i)
        print("lr=" + str(scheduler.get_lr()))

        print("epoch: " + str(i + 1) + " / " + str(epoch))

        """
            学習率スケジューラ
        """
        scheduler.step()

    torch.save(model.to("cpu").state_dict(), os.getcwd() + "/epoch60_size200_lr1e-3__batch64_pow09.pth")

if __name__ == "__main__":
    loop()