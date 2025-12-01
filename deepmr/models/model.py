import torch.nn as nn
import numpy as np

class three_layer_DNN(nn.Module):
    r'''
    三层DNN结构，使用Kaiming初始化
    '''
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(three_layer_DNN, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.fc1.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc2.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc3.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc4.weight, a = np.sqrt(5))

        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu',nn.ReLU())
        self.block.add_module('linear4', self.fc4)
    def forward(self, x):
        return self.block(x)
    

class MLP_Layer(nn.Module):
    r'''
    多层MLP结构，使用Xavier初始化
    '''
    def __init__(self, input_dim = 100, hidden_units = [1000,800,500], output_dim = 36):
        super(MLP_Layer, self).__init__()
        # 构建网络
        self.embeding_layer = nn.ModuleList([nn.Linear(input_dim, hidden_units[0])])
        for i in range(len(hidden_units)-1):
            self.embeding_layer.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
        self.outlayer = nn.Linear(hidden_units[-1], output_dim, bias = False)

        # xavier初始化
        for layer in self.embeding_layer:
            nn.init.xavier_normal_(layer.weight, gain=1)
        nn.init.xavier_normal_(self.outlayer.weight, gain=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.embeding_layer:
            x = self.relu(layer(x))
        return self.outlayer(x)