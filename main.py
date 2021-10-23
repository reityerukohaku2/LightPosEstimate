import torch
from torch.utils.data import DataLoader
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
    trainLoader = DataLoader(train_dataset, batch_size = batchSize, shuffle=True, num_workers = os.cpu_count() - 1, pin_memory = True)
    valLoader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = os.cpu_count() - 1, pin_memory = True)
    testLoader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = os.cpu_count() - 1, pin_memory = True)

    #学習の準備
    model = Net.MyResNet50(1)           #モデルの読み込み
    model = model.cuda()
    criterion = nn.MSELoss()            #損失関数は平均2乗誤差
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

    epoch = 200
    lr = 1e-3
    lr_scheduler = LearningRateScheduler(lr, epoch)
    optimizer = optim.AdamW(model.parameters(), lr=lr) #重み減衰追加した
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir="./logs/test")

    for i in range(epoch):
        model.train()
        test_loss = 0
        torch.set_grad_enabled(True)
        step = 0

        for(RGBimage, DepthImage, pos) in trainLoader:
            #訓練データの初期化
            #print(RGBimage.shape)
            #print(DepthImage.shape)
            RGBimage = RGBimage.cuda()
            DepthImage = DepthImage.cuda()
            #RGBimage = torch.squeeze(RGBimage, 1)
            #DepthImage = torch.squeeze(DepthImage, 1)

            pos = pos.cuda()
            # x = x.cuda()
            # y = y.cuda()
            # z = z.cuda()

            #勾配の初期化
            optimizer.zero_grad()
            print("initialized")

            #混合精度計算部
            with torch.cuda.amp.autocast():
                #学習
                #with torch.cuda.amp.autocast():
                #print(RGBimage.shape)
                #print(DepthImage.shape)
                out = model(RGBimage, DepthImage)

                #要素の分離
                out = out.squeeze()
                # out_x = out[0].item()
                # out_y = out[1].item()
                # out_z = out[2].item()

                # x = x.unsqueeze(1)
                # y = y.unsqueeze(1)
                # z = z.unsqueeze(1)

                loss = criterion(out, pos)
                #loss_y = criterion(torch.Size(out_y), y.float())
                #loss_z = criterion(torch.Size(out_z), z.float())

            print("AMP Clear")

            #3つの損失を適当な重みをかけて束ねる
            # loss = (loss_x + loss_y + loss_z) / 3.0
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

        #lossをtensorBoardに書き込み
        test_loss /= test_size
        writer.add_scalars("losses", { "loss_dropLess_epoch200_size200_lr1e-3_batch64_pow09": test_loss, } , i)
        writer.add_histogram("fc_3.weights", model.fc_3.weight, i)
        writer.add_histogram("fc_3.bias", model.fc_3.bias, i)

        del test_loss

        #バリデーションロス計算
        model.eval()
        loss = 0
        val_step = 1
        torch.set_grad_enabled(False)
        for (RGBimage, DepthImage, pos) in valLoader:
            print("ValStep:" + str(val_step) + " / " + str(val_size))
            val_step += 1
            RGBimage = RGBimage.cuda()
            DepthImage = DepthImage.cuda()
            # x = x.cuda()
            # y = y.cuda()
            # z = z.cuda()
            pos = pos.cuda()

            out= model(RGBimage, DepthImage)
            out = out.squeeze()
            # out_x = out[0].item()
            # out_y = out[1].item()
            # out_z = out[2].item()
            # x = x.unsqueeze(1)
            # y = y.unsqueeze(1)
            # z = z.unsqueeze(1)
            # print(x)
            # print(y)
            # print(z)

            loss = criterion(out, pos)
            # loss_x = criterion(out_x, x.float())
            # loss_y = criterion(out_y, y.float())
            # loss_z = criterion(out_z, z.float())

            #3つの損失を適当な重みをかけて束ねる
            # loss += (loss_x + loss_y + loss_z) / 3.0

            #del loss_x, loss_y, loss_z, RGBimage, DepthImage, x, y, z
            del RGBimage, DepthImage, pos

        #val_loss書き込み
        writer.add_scalars("losses", {"valLoss_dropLess_epoch200_size200_lr1e-3_batch64_pow09": loss,} ,
                                        i)

        #学習率書き込み
        writer.add_scalar("Lerning Rate_lr1e-2", np.array(scheduler.get_lr()), i)
        print("lr=" + str(scheduler.get_lr()))

        print("epoch: " + str(i + 1) + " / " + str(epoch))

        """
            学習率スケジューラ
        """
        scheduler.step()

    #訓練済みモデル保存
    #torch.save(model.to("cpu").state_dict(), os.getcwd() + "/drop3-2_epoch200_size200_lr1e-3_batch64_pow09.pth")
    torch.save(model.to("cpu").state_dict(), os.getcwd() + "/dropLess_epoch200_lr1e-3.pth")

    #############
    ###~~推論~~###
    ##############
    model.eval()
    loss = 0
    step = 0
    torch.set_grad_enabled(False)
    ave_dist = 0
    for (RGBimage, DepthImage, pos) in testLoader:
        out = model(RGBimage, DepthImage)
        out = out.squeeze()
        # x = out[0].item()
        # y = out[1].item()
        # y = out[2].item()

        print("out = " + str(out))
        #dist = torch.sqrt((x - out_x) ** 2 + (y - out_y) ** 2 + (z - out_z) ** 2)
        #ave_dist += dist
        #print("dist:" + str(dist))
        step += 1

    ave_dist /= step
    print("ave_dist:" + str(ave_dist))

    #もうここでONNX作る
    RGBDummyImage = torch.randn(1, 1, 200, 200)
    DepthDummyImage = torch.randn(1, 1, 200, 200)
    torch.onnx.export(model, args=(RGBDummyImage, DepthDummyImage), f="C:/Users/yuma/Desktop/Fujimoto/LightPosEstimate/dropLess_epoch200_lr1e-3.onnx", verbose=True)

if __name__ == "__main__":
    loop()