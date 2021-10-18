import torch.onnx
from torch.utils import data
import GrayNet as Net

model = Net.MyResNet50(1)
model.load_state_dict(torch.load("dropLess_epoch200_lr1e-2.pth", map_location=torch.device("cpu")))
#model = torch.load("drop3-2_epoch5_size200_lr1e-3_batch64_pow09.pth")
model.eval()
torch.set_grad_enabled(False)

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
import torch
from DataSet import MyDataSet

RGBimagePaths = [str(p) for p in Path("Photo/tensor_data/Gray/").glob("*.pt")]
DepthImagePaths = [str(p) for p in Path("Photo/tensor_data/Depth/").glob("*.pt")]

#ファイル名ソート用
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#ファイル名ソート(1,2,3,...23112)
RGBimagePaths = sorted(RGBimagePaths, key=natural_keys)
DepthImagePaths = sorted(DepthImagePaths, key=natural_keys)

dataset = MyDataSet(200)
RGBimage = torch.unsqueeze(dataset[3][0], 0)
DepthImage = torch.unsqueeze(dataset[3][1], 0)
print(RGBimage.shape)

# RGBimage = torch.randn(1, 1, 200, 200).cpu()
# DepthImage = torch.randn(1, 1, 200, 200).cpu()
res = model(RGBimage, DepthImage)
print(res)

torch.onnx.export(model, args=(RGBimage, DepthImage), f="C:/Users/yuma/Desktop/Fujimoto/LightPosEstimate/test4.onnx")