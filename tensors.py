import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)

x_ones = torch.ones_like(x_data)
x_rand =torch.rand_like(x_data, dtype=torch.float)

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(tensor.device)
tensor = tensor.to(device)
print(tensor.device)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1.shape)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print( torch.sum(y1 == y2) )
print( torch.sum(y1 != y2) )

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)