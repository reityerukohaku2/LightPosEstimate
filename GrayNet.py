import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, channel_in = None, channel_out = int, first_block = False):
        super().__init__()
        if channel_in == None:
            channel_in = int(channel_out)

        if first_block or channel_in == channel_out:
            self.stride = (1, 1)
        else:
            self.stride = (2, 2)

        channel = int(channel_out / 4)

        #1*1の畳み込み
        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=(1, 1))
        self.relu1 = nn.ReLU()

        #3*3の畳み込み
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1, stride=self.stride)
        self.relu2 = nn.ReLU()

        # 1x1 の畳み込み
        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=(1, 1))

        # skip connection用のチャネル数調整
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu3 = nn.ReLU()

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.relu2(h)

        h = self.conv3(h)
        h = self.bn2(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)  # skip connection
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)    #チャネル数の調整
        else:
            return lambda x: x

    #チャネル数の調整
    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), stride=self.stride)

class MyResNet50(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        #最初の2層
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 1(Green)
        self.block0 = Block(channel_in=64, channel_out= 256, first_block=True)
        self.block1 = Block(channel_out = 256)
        self.block2 = Block(channel_out = 256)

        # Block 2(Red)
        self.block3 = Block(channel_in = 256, channel_out = 512)
        self.block4 = Block(channel_out = 512)
        self.block5 = Block(channel_out = 512)
        self.block6 = Block(channel_out = 512)

        # Block 3(blue)
        self.block7 = Block(channel_in = 512, channel_out = 1024)
        self.block8 = Block(channel_out = 1024)
        self.block9 = Block(channel_out = 1024)
        self.block10 = Block(channel_out = 1024)
        self.block11 = Block(channel_out = 1024)
        self.block12 = Block(channel_out = 1024)

        # Block 4(yellow)
        self.block13 = Block(channel_in = 1024, channel_out = 2048)
        self.block14 = Block(channel_out = 2048)
        self.block15 = Block(channel_out = 2048)

        #average pool以降
        self.avg_pool = GlobalAvgPool2d()  # TODO: GlobalAvgPool2d
        self.fc_256 = nn.Linear(2048, 256)
        self.fc_64_ReLu = nn.Linear(256, 64)
        self.fc_64_linear = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 3)
        self.fc_x = nn.Linear(3, output_dim)
        self.fc_y = nn.Linear(3, output_dim)
        self.fc_z = nn.Linear(3, output_dim)

    def forward(self, RGBimage, DepthImage):
        input = torch.cat((RGBimage, DepthImage), 1)
        #print(input.shape)

        h = self.conv1(input)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.block0(h)
        h = self.block1(h)
        h = self.block2(h)

        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)

        h = self.block7(h)
        h = self.block8(h)
        h = self.block9(h)
        h = self.block10(h)
        h = self.block11(h)
        h = self.block12(h)

        h = self.block13(h)
        h = self.block14(h)
        h = self.block15(h)

        h = self.avg_pool(h)
        h = self.fc_256(h)
        h = torch.relu(h)

        h = self.fc_64_ReLu(h)
        h = torch.relu(h)

        h = self.fc_64_linear(h)
        #h = nn.Identity(h)

        h = self.fc_3(h)
        #h = nn.Identity(h)

        x = self.fc_x(h)
        y = self.fc_y(h)
        z = self.fc_z(h)

        return x, y, z
class GlobalAvgPool2d(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))