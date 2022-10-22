import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

'''
线性回归的简洁实现
'''

# 生成数据集，生成y=Xw+b噪声
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 从均值为0，标准差为1的正态分布中提取随机数
    y = torch.matmul(X, w) + b  # matmul: 张量乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 读取数据集，构造⼀个PyTorch数据迭代器
def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2, 1))  # 定义模型
loss = nn.MSELoss()  # 定义损失函数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 定义优化算法

# 训练
num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差: ', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差: ', true_b - b)