import torch
from torch import nn

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

print(input)
print(target)
print(torch.empty(10, dtype=torch.long).random_(10))