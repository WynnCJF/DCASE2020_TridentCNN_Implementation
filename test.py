import torch
import torch.nn as nn
import numpy as np

'''
a = torch.Tensor([[0.1, 0.2, 0.4, 0.6, 0.8, 0.8, 0.9, 0.5, 0.5, 0.2]])

print(a.shape)

a_max, i = a.max(1)

print(i.item())
'''


save_list = [[1, 2], [3, 4], [-1, 5]]
save_list.append([3, 7])
save_arr = np.array(save_list)
np.savetxt("test_dict_save.csv", save_arr, delimiter=',')