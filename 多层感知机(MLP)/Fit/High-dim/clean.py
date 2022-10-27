import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
高维线性回归的简洁实现
'''

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05  # 标准差为0.01, 偏置置为0.05
# 生成y=Xw+b+噪声
train_data = d2l.synthetic_data(true_w, true_b, n_train)  
test_data = d2l.synthetic_data(true_w, true_b, n_test)
train_iter = d2l.load_array(train_data, batch_size)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
num_epochs, lr = 100, 0.003

def train(wd):
    model = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in model.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([{"params":model[0].weight,'weight_decay': wd},
                               {"params":model[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
    xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(model(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(d2l.evaluate_loss(model, train_iter, loss),
                                    d2l.evaluate_loss(model, test_iter, loss)))
    print('w的L2范数: ', model[0].weight.norm().item())
    plt.show()

train(3)
