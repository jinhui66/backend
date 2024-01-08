import random
import torch
from d2l import torch as d2l
from torch import nn

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

net = nn.Sequential(nn.Linear(2,1))

lr = 0.03
num_epochs = 3
loss = nn.MSELoss()

for epoch in range(num_epochs):
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    for X, y in data_iter(batch_size, features, labels):
        optimizer.zero_grad()
        l = loss(net(X), y)  # X和y的小批量损失
        l.backward()
        optimizer.step()
    with torch.no_grad():
        train_l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 获取权重
print(net.state_dict())

torch.save(net, "net.pth")

# d2l.plt.show()
