import torch
import inception as block
import matplotlib.pyplot as plt
from torch import nn 
from d2l import torch as d2l

'''
实现一个GoogLeNet网络
'''

# 第⼀个模块使⽤64个通道、7×7卷积层
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第⼆个模块使⽤两个卷积层：第⼀个卷积层是64个通道、1×1卷积层；第⼆个卷积层使⽤将通道数量增加三倍的3×3卷积层
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第三个模块串联两个完整的Inception块
b3 = nn.Sequential(
    block.Inception(192, 64, (96, 128), (16, 32), 32),
    block.Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第四模块串联5个Inception块
b4 = nn.Sequential(
    block.Inception(480, 192, (96, 208), (16, 48), 64),
    block.Inception(512, 160, (112, 224), (24, 64), 64),
    block.Inception(512, 128, (128, 256), (24, 64), 64),
    block.Inception(512, 112, (144, 288), (32, 64), 64),
    block.Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第五模块包含两个Inception块，其后紧跟输出层
b5 = nn.Sequential(
    block.Inception(832, 256, (160, 320), (32, 128), 128),
    block.Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))  # 构建网络

'''
# 测试
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
'''

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
