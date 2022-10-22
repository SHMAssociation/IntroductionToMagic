import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
from accumulator import Accumulator
from animator import Animator

'''
softmax回归，实现对Fashion-MNIST数据集的分类
'''

batch_size = 256  # 设置数据迭代器的批量⼤⼩为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 读取Fashion-MNIST数据集

# 定义softmax操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 对每一行求和
    result = X_exp/partition
    return result

# 定义模型
def model(X):
    return softmax(torch.matmul(X.reshape((-1,  W.shape[0])), W) + b)

# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 定义⼩批量随机梯度下降来优化模型的损失函数，设置学习率为0.1
lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# 定义训练
def train(model, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(model, torch.nn.Module):
        model.train()
    
    # 训练损失总和、准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        #计算梯度并更新参数
        y_hat = model(X)
        l = loss(y_hat, y)
        # 使⽤PyTorch内置的优化器和损失函数
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()  # 梯度清零
            l.mean().backward()  # 反向传播
            updater.step()  # 更新参数
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 模型训练
def train_softmax(model, train_iter, test_iter, loss, num_epochs, updater):
    # 利⽤Animator类来可视化训练进度
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train(model, train_iter, loss, updater)
        test_acc = d2l.evaluate_accuracy(model, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 预测标签
def predict(model, test_iter, n=5):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(model(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()

'''
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))
'''

# 初始化模型参数
num_inputs = 784  # 原始数据集中的每个样本都是28×28的图像，这里将展平每个图像，把它们看作⻓度为784的向量
num_outputs = 10  # 因为数据集有10个类别，所以⽹络输出维度为10
# 使⽤正态分布初始化我们的权重W，偏置b初始化为0
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 开始训练
num_epochs = 10
train_softmax(model, train_iter, test_iter, cross_entropy, num_epochs, updater)
# 预测
predict(model, test_iter)
