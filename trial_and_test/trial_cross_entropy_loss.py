import torch
from cross_entropy_loss import CELoss

predict = torch.Tensor([[1, 1, 1], [1, 1, 1]])
predict.squeeze()
print(predict.shape)

target = torch.Tensor([[0, 0, 1], [0.2, 0.3, 0.5]])

celoss = CELoss()

loss = celoss(predict, target)

print(loss)

