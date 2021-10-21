# dropout练习
import torch
import torch.nn as nn

x = torch.Tensor([3, 1, 4, 1, 5])
layer = nn.Dropout(0.2)    # 被丢弃的概率为0.2，放大的倍数为1/(1-0.2)
y = layer(x)
print(y,"dropout后的x")