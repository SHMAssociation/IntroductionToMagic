import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

'''
实现一个多层感知机(MLP)
'''

# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 定义模型
def model(X):
    X = X.reshape((-1, num_inputs))  # 使⽤reshape将每个⼆维图像转换为⼀个⻓度为num_inputs的向量
    H = relu(torch.matmul(X, W1) + b1)  # H = X*W1 + b1
    O = torch.matmul(H, W2) + b2  # O = H*W2 + b2
    return O

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256  # 设置输入层、输出层和隐藏层的维度

# 初始化输入层到隐藏层的权重参数和偏置参数
# 权重参数大小为784*256，偏置参数初始化为0
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 初始化隐藏层到输出层的权重参数和偏置参数
# 权重参数大小为256*10，偏置参数初始化为0
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]  # 参数集合
loss = nn.CrossEntropyLoss(reduction='none')  # 损失函数

# 训练，多层感知机的训练过程与softmax回归的训练过程完全相同
# 这里直接调用d2l的库函数，不再重复实现，具体可查阅softmax的有关内容
lr = 0.1
num_epochs= 10
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(model, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(model, test_iter)
plt.show()
