import torch.onnx
import GrayNet as Net

model = Net.MyResNet50(1).cpu()
model.load_state_dict(torch.load("drop3-2_epoch200_size200_lr1e-3_batch64_pow09.pth"))
model.eval()

RGBimage = torch.randn(1, 1, 200, 200).cpu()
DepthImage = torch.randn(1, 1, 200, 200).cpu()
res = model(RGBimage, DepthImage)
print(res)

torch.onnx.export(model, args=(RGBimage, DepthImage), f="C:/Users/yuma/Desktop/Fujimoto/LightPosEstimate/test.onnx", verbose=True)