import random
import torch

'''
实现线性回归，包括数据流⽔线、模型、损失函数和⼩批量随机梯度下降优化器
'''

# 生成数据集，生成y=Xw+b噪声
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 从均值为0，标准差为1的正态分布中提取随机数
    y = torch.matmul(X, w) + b  # matmul: 张量乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 定义线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 定义平方损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降，lr代表学习率
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

batch_size = 10
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

'''
print('features:', features[0],'\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()
'''

'''
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
'''

# 初始化模型参数
# 通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
# 初始化超参数
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的⼩批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使⽤参数的梯度更新算法参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
