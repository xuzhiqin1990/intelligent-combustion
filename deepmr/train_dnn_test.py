import numpy as np
import torch
import os
import time
import copy
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from dmgr.utils import *
from dmgr.base.DeePMR_base import DeePMR_base_class
from dmgr.visualization.plot_train_loss import plot_loss
from dmgr.data import load_dnn_data_x, load_dnn_data_y
from dmgr.models import three_layer_DNN
from tasks_combustion import *


x_train, x_test = load_dnn_data_x('x.npz')
y_train, y_test = load_dnn_data_y('y.npz')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
my_model = three_layer_DNN(427, [4000, 2000, 1000], 18).to(device)

# 定义损失和优化器
criterion = nn.MSELoss(reduction='mean')
# optimizer = optim.SGD(my_model.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

optimizer = optim.Adam(my_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

t0 = time.time()
epoch_index = []
train_his = []
test_his = []
batch_size = 512
batch_num = np.size(x_train, 0) // batch_size
for epoch in range(600):
    my_model.eval()    # 预测
    with torch.no_grad():
        y_dnn_test = my_model(torch.FloatTensor(x_test).to(device))
        test_loss = float(
            criterion(y_dnn_test, torch.FloatTensor(y_test).to(device)))
    my_model.train()   # 训练
    train_loss = 0
    for i in range(batch_num):  # 按照 batch 进行训练
        x_train_batch = x_train[i * batch_size: (i+1) * batch_size, :]
        y_train_batch = y_train[i * batch_size: (i+1) * batch_size, :]
        y_dnn_train_batch = my_model(
            torch.FloatTensor(x_train_batch).to(device))
        train_loss_batch = criterion(
            y_dnn_train_batch, torch.FloatTensor(y_train_batch).to(device))

        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()
        train_loss += float(train_loss_batch)

    scheduler.step()  # 调整学习率

    train_loss /= batch_num
    epoch_index.append(epoch)
    train_his.append(train_loss)
    test_his.append(test_loss)

    print('epoch: {}\t train loss: {:.4e}   test loss: {:.4e}   time cost: {} s   lr:{:.2e}'
            .format(epoch, train_loss, test_loss, int(time.time()-t0), optimizer.param_groups[0]['lr']))