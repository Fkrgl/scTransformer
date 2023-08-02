import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


a = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(f'a normal:\n{a}')

mask = np.array([[True, False, True], [True, False, True], [True, False, True]])
print(f'mask:\n{mask}')

random = np.random.choice(np.arange(100), size=(3,3))
print(f'random:\n{random}')
a[mask] = random[mask]
print(f'a modified:\n{a}')

a[mask] = 0
print(f'a zero:\n{a}')

mask = np.array([0, 0, 1, 0, 1, 0], dtype=bool)
print(f'src_key_padding_mask:\n{mask}')
attn_mask = torch.zeros(size=(len(mask), len(mask)))
print(f'empty attn_mask: \n{attn_mask}')
attn_mask[mask, :] = 1
attn_mask[:, mask] = 1
print(f'attn_mask:\n{attn_mask} ')

# sequence_length = 10
# src_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)
# print(src_mask)
# src_mask = src_mask == 0  # Invert the src_mask (1s are replaced with 0s and vice versa)
# print(src_mask)
# src_mask = src_mask.unsqueeze(0).unsqueeze(1)
# print(src_mask)
#
# #a = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]])
# a = torch.tensor(a)
# print(a)
# stacked_tensor = torch.stack([a] * 3, dim=0)
# print(stacked_tensor)
# print(a.shape)
# print(stacked_tensor.shape)
# print(torch.vstack([a,a]).shape)
# print(torch.vstack([a,a]))

# a = torch.tensor([[[2,1], [2,2], [3,1]]])
print(a)
print(torch.FloatTensor(a.shape[0], a.shape[1]).uniform_(-2, 2))