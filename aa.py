

import torch

a = torch.randn((3,6))
b = torch.randn((3,6))

print(a.size(1))

c = torch.concatenate([a, b], dim=0)
t = torch.arange(0,3)

vars, idxs = torch.topk(a, k=5, dim=1, sorted=True, largest=True)

t = (idxs == t.unsqueeze(1)).to(torch.float)

print(t, t.dtype, t.shape)
print(t.sum(dim=0))
print(vars)
print(idxs)

import numpy as np

l = [np.arange(2, 6) for i in range(4)]
print(l)
l_c = np.concatenate(l)
print(l_c)