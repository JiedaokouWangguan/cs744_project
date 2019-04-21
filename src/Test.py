import torch

a = torch.tensor([[1., -1.], [1., -1.]])
b = torch.tensor([[-2., -2.], [-2., -2.]])

a.add_(-1, b)
a = a * 10
print(a)
print(b)



