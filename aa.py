

import torch

from torchvision import transforms

a = torch.zeros(size=(1,1,28,28))
b = torch.tile(a, (1,3,1,1))
print(b.shape)