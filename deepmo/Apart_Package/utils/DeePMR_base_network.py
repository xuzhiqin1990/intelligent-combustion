# -*- coding:utf-8 -*-
from math import gamma
import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F


# 设置神经网络结构, 只接受三层隐藏层的输入
class MyNet(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MyNet, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)
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


# 训练汇总的IDT数据
# Input: 01-vector (116 dim)
# Output: IDT data (num_T * num_P * num_phi dim)
class IDT_DNN(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(IDT_DNN, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)
        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('relu6', nn.ReLU6())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('relu6', nn.ReLU6())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu6',nn.ReLU6())
        self.block.add_module('linear4', self.fc4)
    def forward(self, x):
        return self.block(x)


'''============================================================================================================='''
'''                                         为训练好psr所做的不同尝试                                              '''
'''============================================================================================================='''
# 设置神经网络结构, 只接受三层隐藏层的输入
class DNN1(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(DNN1, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)
        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('sigmoid', nn.Sigmoid())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('sigmoid', nn.Sigmoid())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu',nn.ReLU())
        self.block.add_module('linear4', self.fc4)
    def forward(self, x):
        return self.block(x)

# 设置神经网络结构, 只接受三层隐藏层的输入
# 使用Kaiming初始化
class DNN2(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(DNN2, self).__init__()
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

        # 千万别改'sigmoid'这个字符串，不然没法加载以前的模型
        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('sigmoid', nn.ReLU()) # 不要改这个层的名字，不然有些模型加载的时候会报错
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('sigmoid', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu',nn.ReLU())
        self.block.add_module('linear4', self.fc4)
    def forward(self, x):
        return self.block(x)



# 设置神经网络结构, 只接受三层隐藏层的输入
# 使用Kaiming初始化
# 使用Dropout
class DNN3(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(DNN3, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)

        # dropout
        self.dropout = nn.Dropout(p = 0.2)

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.fc1.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc2.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc3.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc4.weight, a = np.sqrt(5))

        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('dropout1', self.dropout)
        self.block.add_module('sigmoid', nn.ReLU())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('dropout2', self.dropout)
        self.block.add_module('sigmoid', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('dropout3', self.dropout)
        self.block.add_module('relu',nn.ReLU())
        self.block.add_module('linear4', self.fc4)
        
    def forward(self, x):
        return self.block(x)


# 设置神经网络结构, 只接受三层隐藏层的输入
# 使用Kaiming初始化
# 使用Dropout
class DNN4(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(DNN4, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)

        # dropout
        self.dropout = nn.Dropout(p = 0.1)

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.fc1.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc2.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc3.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc4.weight, a = np.sqrt(5))

        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('dropout1', self.dropout)
        self.block.add_module('sigmoid', nn.ReLU())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('dropout2', self.dropout)
        self.block.add_module('sigmoid', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu',nn.ReLU())
        self.block.add_module('linear4', self.fc4)
        
    def forward(self, x):
        return self.block(x)



class focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
        :param labels:  实际类别. size:[B,N] or [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

# 用于 Regression 的 Focal Loss
class focal_loss_reg(nn.Module):    
    def __init__(self, ALPHA = 0.8, GAMMA = 2, task_num = 2, weight=None, size_average=True):
        super(focal_loss_reg, self).__init__()
        self.ALPHA = ALPHA; self.GAMMA = GAMMA; self.task_num = task_num

    def forward(self, inputs, targets):
        target_norm = [F.mse_loss(torch.zeros_like(inputs[k]), targets[k], reduction='mean') for k in range(self.task_num)]
        KPI = []
        for k in range(self.task_num):
            loss_k = F.mse_loss(inputs[k], targets[k], reduction='mean')
            KPI.append(((target_norm[k] - loss_k) / target_norm[k])**2)
        focal_loss = torch.sum([-(1-kpi)**gamma * torch.log(kpi) for kpi in KPI])
        return focal_loss





# 适用于猫狗分类问题的DNN
# 设置神经网络结构, 只接受三层隐藏层的输入
# 使用Kaiming初始化
class binary_classification_DNN(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(binary_classification_DNN, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim, bias = False)
        # self.fc5  = nn.Softmax(output_dim)

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.fc1.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc2.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc3.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc4.weight, a = np.sqrt(5))

        # 千万别改'sigmoid'这个字符串，不然没法加载以前的模型
        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('ReLU1', nn.ReLU()) # 不要改这个层的名字，不然有些模型加载的时候会报错
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('ReLU2', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('ReLU3',nn.ReLU())
        self.block.add_module('linear4', self.fc4)
        # self.block.add_module('softmax', self.fc5)
    def forward(self, x):
        return self.block(x)

# 设置神经网络结构, 只接受三层隐藏层的输入
# 使用Kaiming初始化
class DNN2_resnet(nn.Module):
    """
    DNN2 的 ResNet 版本，由一个拓宽层和四个基元组成，在拓宽层和第三基元前、第一基元后和第四基元前使用 skip connection 连接

    每个基元是一个两层 MLP ，宽度等同于输入

    拓宽层是一个单层 MLP，宽度很大用于特征提取，宽度等同于输入的 10 倍

    目前还接受 hidden units 的原因是懒得改代码
    """
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hiden_units, output_dim):
        # 定义隐藏层
        super().__init__()
        self.fc_flatten = nn.Linear(input_dim, 10 * input_dim, bias = False)

        self.fc1  = nn.Linear(10 * input_dim, input_dim)
        self.fcres1  = nn.Linear(10 * input_dim, input_dim)
        self.fc2  = nn.Linear(input_dim, input_dim)
        self.fc3  = nn.Linear(input_dim, input_dim)
        self.fc4  = nn.Linear(input_dim, output_dim)



        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)

    
    def forward(self, x):
        res2 = self.fc1(x)
        res1 = self.fcres1(x)
        x = self.fc2(res2)
        x = self.fc3(res1 + x)
        x = self.fc4(x + res2)
        return x


class DNN2_131(nn.Module):
    """
    APART 1.3.1 版本使用的网络结构\n
    基本网络结构由一个 base 层 + 三个分支层构成\n
    会同时获得 IDT PSR OH_mole 三者的输出\n
    训练网络的计划为:
        1. 首先将三者放在一起训练, loss 是三个分支层的叠加:
            loss = loss1 / loss1.detach() + loss2 / loss2.detach() + loss3 / loss3.detach()\n
        2. 之后的迭代将在这个基础上尝试进行 MMoE 方法 
    Args:
        input_dim: 输入层的宽度
        hidden_units: 隐藏层的宽度; 输入是一个列表，第一个元素是共同的 base 层宽度; 之后是一个三维矩阵代表分支层宽度
                      例如：[3000, [[2000,1000,500],[2000,1000,500],[2000,1000,500]]]
                      
        output_dim: 输出层的宽度，要求输入一个三元列表, 其中分别对应 IDT, PSR, mole 的输出宽度
    """
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim:int, hidden_units:list[int,list], output_dim:int):
        # 定义隐藏层
        super().__init__()
        base_layer_dim, branch_dim = hidden_units
        self.fc = nn.Sequential(
                          nn.Linear(input_dim, base_layer_dim, bias = False),
                          nn.ReLU(),
                          )

        self.IDTblock = nn.Sequential()
        self.IDTblock.add_module("IDT1", nn.Linear(base_layer_dim, branch_dim[0][0]))
        self.IDTblock.add_module("Gelu", nn.GELU())
        self.IDTblock.add_module("IDT2", nn.Linear(branch_dim[0][0], branch_dim[0][1]))
        self.IDTblock.add_module("Gelu", nn.GELU())
        self.IDTblock.add_module("IDT3", nn.Linear(branch_dim[0][1], branch_dim[0][2]))
        self.IDTblock.add_module("Gelu", nn.GELU())
        self.IDTblock.add_module("IDT4", nn.Linear(branch_dim[0][2], output_dim[0]))

        self.PSRblock = nn.Sequential()
        self.PSRblock.add_module("PSR1", nn.Linear(base_layer_dim, branch_dim[1][0]))
        self.PSRblock.add_module("Relu", nn.ReLU())
        self.PSRblock.add_module("PSR2", nn.Linear(branch_dim[1][0], branch_dim[1][1]))
        self.PSRblock.add_module("Relu", nn.ReLU())
        self.PSRblock.add_module("PSR3", nn.Linear(branch_dim[1][1], branch_dim[1][2]))
        self.PSRblock.add_module("Relu", nn.ReLU())
        self.PSRblock.add_module("PSR4", nn.Linear(branch_dim[1][2], output_dim[1]))

        self.moleblock = nn.Sequential()
        self.moleblock.add_module("mole1", nn.Linear(base_layer_dim, branch_dim[2][0]))
        self.moleblock.add_module("Gelu", nn.GELU())
        self.moleblock.add_module("mole2", nn.Linear(branch_dim[2][0], branch_dim[2][1]))
        self.moleblock.add_module("Gelu", nn.GELU())
        self.moleblock.add_module("mole3", nn.Linear(branch_dim[2][1], branch_dim[2][2]))
        self.moleblock.add_module("Gelu", nn.GELU())
        self.moleblock.add_module("mole4", nn.Linear(branch_dim[2][2], output_dim[2]))


        self._init() # 初始化参数
    
    def _init(self, ):
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)

        for name, param in self.IDTblock.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)    
        for name, param in self.PSRblock.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)   
        for name, param in self.moleblock.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)   

    
    def forward(self, x):
        x = self.fc(x)
        IDT = self.IDTblock(x)
        PSR = self.PSRblock(x)
        mole  = self.moleblock(x)
        return IDT, PSR, mole
    
    def get_IDT(self, x):
        x = self.fc(x)
        IDT = self.IDTblock(x)
        return IDT        
    
    def get_PSR(self, x):
        x = self.fc(x)
        PSR = self.PSRblock(x)
        return PSR       

    def get_mole(self, x):
        x = self.fc(x)
        mole = self.moleblock(x)
        return mole      

@DeprecationWarning
class DNN2_132(nn.Module):
    """
    APART 1.3.2 版本使用的网络结构\n
    基本网络结构由一个 base 层 + 2个分支层构成\n
    会同时获得 IDT PSR 2者的输出\n
    训练网络的计划为:
        1. 首先将三者放在一起训练, loss 是三个分支层的叠加:
            loss = loss1 / loss1.detach() + loss2 / loss2.detach()\n
        2. 之后的迭代将在这个基础上尝试进行 MMoE 方法 
    Args:
        input_dim: 输入层的宽度
        hidden_units: 隐藏层的宽度; 目前需要四元列表输入; 对于三个分支层结构相同, 前两层是共同的 base 层, 之后的层是分支层
        output_dim: 输出层的宽度，要求输入一个三元列表, 其中分别对应 IDT, PSR, FS 的输出宽度
    """
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        # 定义隐藏层
        super().__init__()
        self.fc_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.fc_flatten_activation = nn.ReLU()
        self.fc_flatten2 = nn.Linear(hidden_units[0], hidden_units[1], bias = False)

        self.IDTblock = nn.Sequential()
        self.IDTblock.add_module("IDT1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.IDTblock.add_module("Gelu", nn.GELU())
        self.IDTblock.add_module("IDT2", nn.Linear(hidden_units[2], output_dim[0]))

        self.PSRblock = nn.Sequential()
        self.PSRblock.add_module("PSR1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.PSRblock.add_module("Gelu", nn.GELU())
        self.PSRblock.add_module("PSR2", nn.Linear(hidden_units[2], output_dim[1]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.fc_flatten1.weight)
        nn.init.kaiming_normal_(self.fc_flatten2.weight)

    
    def forward(self, x):
        x = self.fc_flatten1(x)
        x = self.fc_flatten_activation(x)
        x = self.fc_flatten2(x)
        x = self.fc_flatten_activation(x)

        IDT = self.IDTblock(x)
        PSR = self.PSRblock(x)
        return IDT, PSR


class ANET_132(nn.Module):
    """
    APART 1.3.2 版本使用的网络结构其二\n
    基本网络结构由2个 base 层 + 1个分支层构成 + 反演的 autoencoder 结构 \n
    会同时获得 IDT PSR 2者的输出\n
    训练网络的计划为:
        1. 首先将三者放在一起训练, loss 是三个分支层的叠加:
            loss = loss1 / loss1.detach() + loss2 / loss2.detach()\n
        2. 训练完 encoder 之后训练 decoder
        3. 反问题中直接在 decoder 中寻找
    Args:
        input_dim: 输入层的宽度
        hidden_units: 隐藏层的宽度; 目前需要四元列表输入; 对于三个分支层结构相同, 前两层是共同的 base 层, 之后的层是分支层
        output_dim: 输出层的宽度，要求输入一个三元列表, 其中分别对应 IDT, PSR, FS 的输出宽度
    """
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super().__init__()
        # 定义 encoder 隐藏层
        self.encoder_fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_units[0], bias = False),
                    nn.ReLU(),
                    nn.Linear(hidden_units[0], hidden_units[1], bias = False),
        )


        self.forward_IDTblock = nn.Sequential()
        self.forward_IDTblock.add_module("IDT1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.forward_IDTblock.add_module("Gelu", nn.GELU())
        self.forward_IDTblock.add_module("IDT2", nn.Linear(hidden_units[2], output_dim[0]))

        self.forward_PSRblock = nn.Sequential()
        self.forward_PSRblock.add_module("PSR1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.forward_PSRblock.add_module("Gelu", nn.GELU())
        self.forward_PSRblock.add_module("PSR2", nn.Linear(hidden_units[2], output_dim[1]))

        # decoder 部分
        self.decoder_fc = nn.Sequential(
                    nn.Linear(2 * hidden_units[1], hidden_units[0], bias = False),
                    nn.ReLU(),
                    nn.Linear(hidden_units[0], input_dim, bias = False),
        )

        self.backward_IDTblock = nn.Sequential()
        self.backward_IDTblock.add_module("IDT-2", nn.Linear(output_dim[0], hidden_units[2]))
        self.backward_IDTblock.add_module("Gelu", nn.GELU())
        self.backward_IDTblock.add_module("IDT-1", nn.Linear(hidden_units[2], hidden_units[1]))

        self.backward_PSRblock = nn.Sequential()
        self.backward_PSRblock.add_module("PSR-2", nn.Linear(output_dim[1], hidden_units[2]))
        self.backward_PSRblock.add_module("Gelu", nn.GELU())
        self.backward_PSRblock.add_module("PSR-1", nn.Linear(hidden_units[2], hidden_units[1]))

        self._init() # 初始化参数
    
    def _init(self, ):
        for name, param in self.encoder_fc.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
        
        for name, param in self.decoder_fc.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
   
    def encode(self, x):
        x = self.encoder_fc(x)
        IDT = self.forward_IDTblock(x)
        PSR = self.forward_PSRblock(x)
        return IDT, PSR
    
    def decode(self, IDT, PSR):
        IDT = self.backward_IDTblock(IDT)
        PSR = self.backward_PSRblock(PSR)
        x = torch.cat((IDT, PSR),  dim = 1)
        x = self.decoder_fc(x)
        return x

    def forward(self, x):
        x = self.decode(*self.encode(x))
        return x

    def freeze_encoder(self, defreeze = False):
        """
        通过设置 require_grad = False 实现对于 encoder 所有层的冻结
        一般而言需要配合 optimizer 中 filter() 实现更新，否则可能冻结失败
        param:
            defreeze: 是否解冻，默认为冻结
        """
        for fc in self.encoder_fc.children():
            fc.requires_grad_(defreeze)
        for fc in self.forward_IDTblock.children():
            fc.requires_grad_(defreeze)
        for fc in self.forward_PSRblock.children():
            fc.requires_grad_(defreeze)
    
    def freeze_decoder(self, defreeze = False):
        """
        通过设置 require_grad = False 实现对于 decoder 所有层的冻结
        一般而言需要配合 optimizer 中 filter() 实现更新，否则可能冻结失败
        param:
            defreeze: 是否解冻，默认为冻结
        """
        for fc in self.decoder_fc.children():
            fc.requires_grad_(defreeze)
        for fc in self.backward_IDTblock.children():
            fc.requires_grad_(defreeze)
        for fc in self.backward_PSRblock.children():
            fc.requires_grad_(defreeze)