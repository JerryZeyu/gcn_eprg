import torch
a = torch.tensor([[1,2,3,4,5]])
b= torch.tensor([[[1,1],[1,1],[1,1],[1,1],[1,1]]])
print(a.unsqueeze(2).shape)
print(b.shape)
print(b*a.unsqueeze(2))
print((b*a.unsqueeze(2)).shape)