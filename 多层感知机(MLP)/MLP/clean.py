import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

'''
MLP的简介实现
'''

#定义模型
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
model.apply(init_weights)

# 训练
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(model.parameters(), lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(model, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()