import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

'''
实现一个VGG网络
'''

# 实现一个VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# 实现一个VGG-11
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1

    # 添加卷积层
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    net = nn.Sequential(
        *conv_blks, nn.Flatten(),  # 卷积层部分
        
        # 全连接层
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), 
        nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(0.5),

        # 输出层
        nn.Linear(4096, 10)
    )
    
    return net

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

'''
# 测试
net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
'''

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

# 训练
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()
