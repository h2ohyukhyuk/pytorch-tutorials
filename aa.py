

import torch

from torchvision import transforms

a = torch.randn(size=(3, 4))

print(a)

pred = a.max(1)
print(pred)

pred = a.max(1, keepdim=True)
print(pred)

pred = a.max(1, keepdim=True)[1]
print(pred)