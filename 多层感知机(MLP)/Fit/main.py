import math
import torch
import numpy as np
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
多项式拟合
'''

# 生成数据集
max_degree = 20 # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练集和测试集的大小
true_w= np.zeros(max_degree)
true_w[0: 4]= np.array([5, 1.2, -3.4, 5.6])  # 数据的真实值

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))

for i in range(max_degree):
    poly_features[:, 1] /= math.gamma(i + 1)  # 阶乘

labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# numpy的array转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]

'''
print(features[:2])
print(poly_features[:2, :])
print(labels[:2])
'''

# 对模型进行训练和测试
# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

#训练
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]

    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)  # 定义优化算法
    
    # 绘图
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epochs], 
                            ylim=[1e-3, 1e2], legend=['train', 'test'])
    
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 40 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_loss, test_loss))
            print("epoch", epoch + 1, ": " + "train_loss ", train_loss, "test_loss ", test_loss)
    
    print('weight:', net[0].weight.data.numpy())

# 三阶多项式拟合
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
plt.show()
