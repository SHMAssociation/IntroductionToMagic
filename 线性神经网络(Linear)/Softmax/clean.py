import torch
from torch import nn 
from d2l import torch as d2l

'''
softmax回归的简单实现
'''

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)) # 在Sequential中添加⼀个带有10个输出的全连接层
model.apply(init_weights) # 初始化权重
loss = nn.CrossEntropyLoss(reduction='none')  # 损失函数

#训练
num_epochs= 10
trainer = torch.optim.SGD(model.parameters(), lr=0.1)
d2l.train_ch3(model, train_iter, test_iter, loss, num_epochs, trainer)