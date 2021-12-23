
from efficientnet_pytorch import EfficientNet
import torch
from thop import profile
from torchstat import stat

model = EfficientNet.from_pretrained('efficientnet-b0')
x = torch.randn(1,3,224,224)
y = model.extract_features(x)
print(y.size())
flops, params = profile(model, inputs=(x,))
stat(model, (3, 224, 224))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')
total = sum([param.nelement() for param in model.parameters()])  # 计算总参数量
print("Number of parameter: %.6f" % (total))  # 输出