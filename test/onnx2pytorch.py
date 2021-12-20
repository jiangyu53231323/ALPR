import onnx
import torch

import onnx2pytorch
# from onnx2pytorch import ConvertModel
from onnx2pytorch.convert.model import ConvertModel
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from torchstat import stat

path_to_onnx_model = "E:\CodeDownload/picodet_l_320_coco.onnx"
onnx_model = onnx.load(path_to_onnx_model)
pytorch_model = ConvertModel(onnx_model)

x = torch.randn(1, 3, 64, 224)
flops, params = profile(pytorch_model, inputs=(x,))
stat(pytorch_model, (3, 64, 224))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')
total = sum([param.nelement() for param in pytorch_model.parameters()])  # 计算总参数量
print("Number of parameter: %.6f" % (total))  # 输出
flops = FlopCountAnalysis(pytorch_model, x)
print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')
print(flop_count_table(flops))