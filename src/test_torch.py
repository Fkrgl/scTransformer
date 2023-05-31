import torch
import numpy as np



true = torch.Tensor([5,3,0])
b = torch.Tensor([1,6,0])
c = torch.Tensor([5,3,0])
d = torch.Tensor([5,3,1])
e = torch.Tensor([100,100,100])
f = torch.Tensor([0,0,0])
g = torch.Tensor([0,0,0])
c2 = torch.Tensor([5,3,0, 0, 0])
d2 = torch.Tensor([5,3,1, 0, 0])

inputs = [b, c, d, e, f]

creterion_cce1 = torch.nn.MSELoss(reduction='sum')
creterion_cce2 = torch.nn.MSELoss()

print(torch.tensor([1,2,3]) + torch.Tensor([[1, 2, 3], [4, 5, 6]]))


a = torch.randint(0, 10, size=(2,3,4))
b = torch.randint(0, 10, size=(3,4))
c = torch.randint(0, 10, size=(3,4))
empty = torch.empty(size=(3,4))
l = [b, c]
print(torch.vstack(l).shape)



# # Example of target with class indices
# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(input)
# print(target)
# output = loss(input, target)
# output.backward()
# print(output)

# Boolean tensor indicating which elements to extract
# boolean_tensor = torch.tensor([True, False, True, False, True])
# labels = torch.tensor([[5,3,1], [3, 7, 9]])
# pred = torch.tensor([[2,2,4], [1, 1, 9]])
#
# b = torch.tensor([[True, False, True], [False, True, True]])
# # Extract numbers using boolean tensor
# binary_mask = torch.where(b, torch.tensor(1), torch.tensor(0))
# # Print the extracted numbers
# print(binary_mask * a)
# print(creterion_cce(pred, labels))
# print(creterion_cce(pred*a, labels*a))
#
# labels = torch.tensor([[5,1], [7, 9]])
# pred = torch.tensor([[2,4], [1, 9]])
# print(creterion_cce(pred, labels))


