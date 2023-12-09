# https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

print(w.grad)
print(b.grad)
loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
gt = torch.rand_like(out, requires_grad=False)
loss = torch.mean(out - gt, dim=-1).pow(2)

print('loss shape: ', loss.shape)

loss.backward(torch.ones_like(loss), retain_graph=True)
print(f"First call\n{inp.grad}")

loss.backward(torch.ones_like(loss), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")

inp.grad.zero_()
loss.backward(torch.ones_like(loss), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")