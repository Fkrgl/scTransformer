import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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

a = torch.tensor([1,2,3])
print(torch.vstack([a,a]).shape)

one_hot = torch.nn.functional.one_hot(torch.arange(5), 5)
print(one_hot[0])
print(torch.zeros(3))
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

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(input)
# print(target)
# print(torch.transpose(input, 0, 1))

# look at network output
print()
label = torch.load('../data/val_input_multipleLeaveOut_100_epochs.pt', map_location=torch.device('cpu'))
pred = torch.load('../data/reconstructed_profiles_multipleLeaveOut_100_epochs.pt', map_location=torch.device('cpu'))
mask = torch.load('../data/masks_multipleLeaveOut_100_epochs.pt', map_location=torch.device('cpu'))
print(mask[1])
print(f'shape of mask: {mask[1].shape}')
l = label[1]
print(f'shape of label: {label.shape}')
p = pred[1]
print(f'shape of pred: {pred.shape}')
m = mask[1]
print(p[:,1])
print(l[m])
print(p[m,4])
print(p[m,4].shape)

s = nn.Softmax(dim=0)
# for i in range(pred.shape[0]):
#     print(s(pred[i, 0 ,:]))
#for i in range(4):
plt.hist(pred[:, :, 2].detach().numpy(), alpha=0.5)
plt.legend()
plt.show()

# print(torch.sum(p[m], dim=1))
# print(torch.sum(p[m], dim=1).shape)
# print(pred[0,1,:])
# print(torch.sum(pred[0,1,:]))

# a = np.array([1,2,3,4])
# print(np.random.choice(a, size=4, replace=False))
# b = np.array([3,0])
# print(a[b])