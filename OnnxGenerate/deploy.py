# from backup.Small_UnetGated import UNetLWGated_lessc
from Small_UnetGated import UNetLWGated
import torch
from config import mdevice

width = 1280
height = 720
# width = 720
# height = 480

x=torch.randn(1,6,height,width).to(mdevice)
feature=torch.randn(1,6,height,width).to(mdevice)
mask=torch.randn(1,6,height,width).to(mdevice)
hisBuffer=torch.randn(3,4,height,width).to(mdevice)
model = UNetLWGated(18, 3)
# model.load_state_dict(torch.load("totalModel.290.pth.tar")["state_dict"])
model = model.to(mdevice)
model.eval()
input_names = [ "input_x","input_feature","input_mask","input_history"]
output_names = [ "output1"]
print(model.conv1.conv_feature.weight[0,0])

torch.onnx.export(model, (x,feature,mask,hisBuffer),
                  "UNetGated.onnx", verbose=True, input_names=input_names, output_names=output_names,opset_version=11)
