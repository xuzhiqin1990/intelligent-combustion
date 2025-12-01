# -*- coding:utf-8 -*-
import numpy as np
import torch.nn as nn
import torch, math
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss

def exp_kaiming_normal_(
    tensor, rate = 0.5,
    mode: str = "fan_in",
    generator = None,
):
    r"""Fill the input `Tensor` with values using a Kaiming normal distribution.

    The method is described in `Delving deep into rectifiers: Surpassing
    human-level performance on ImageNet classification` - He, K. et al. (2015).
    The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    if 0 in tensor.shape:
        print("Initializing zero-element tensors is a no-op")
        return tensor
    std = 1 / (tensor.numel() ** (1 / 2)) * rate
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)

"""=========================================================================================================="""
"""                                      Network Structure Definition                                        """          
"""=========================================================================================================="""

# 设置神经网络结构, 只接受三层隐藏层的输入
# 使用Kaiming初始化
class Network_PlainSingleHead(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, input_dim, hidden_units, output_dim):
        super(Network_PlainSingleHead, self).__init__()
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
        self.block.add_module("GELU", nn.GELU())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('GELU', nn.GELU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('GELU',nn.GELU())
        self.block.add_module('linear4', self.fc4)

    def forward(self, x):
        return self.block(x)


class one2oneLayer(nn.Module):
    """
    在输入和输出维度相同的情况下，生成对应分量 1 对 1 的网络层，不同的输入分量到输出相互独立不影响
    不能使用 nn.Linear 来实现，因为 nn.Linear 会将输入的所有分量都连接到输出的所有分量
    """
    def __init__(self, input_dim, index = 1):
        super().__init__()
        self.one2oneLayer_weight: torch.Tensor
        weight = nn.Parameter(torch.Tensor(input_dim))
        self.register_parameter('one2oneLayer_weight', weight)
        # 对 weight 和 bias 的梯度进行 clamp 限制;
        self.one2oneLayer_weight.register_hook(lambda grad: grad.clamp_(-index, index))

    def forward(self, x):
        # 返回 x * weight + bias
        return x * self.one2oneLayer_weight


class Network_PSR_MultiLayer(nn.Module):
    # 构造方法里面定义可学习参数的层
    def __init__(self, layer_nums, input_dim, hidden_units, output_dim):
        super(Network_PSR_MultiLayer, self).__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc2_duplicate  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc3_out  = nn.Linear(hidden_units[1], output_dim)
        self.one2oneLayer = nn.ModuleList([
            one2oneLayer(output_dim, layer_nums - index) for index in range(layer_nums - 1)
        ])
        self.activation = nn.GELU()

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.fc1.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc2.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc2_duplicate.weight, a = np.sqrt(5))
        nn.init.kaiming_normal_(self.fc3_out.weight, a = np.sqrt(5))
        for layer in self.one2oneLayer:
            nn.init.kaiming_normal_(layer.one2oneLayer_weight.unsqueeze(0), a = np.sqrt(5))

        self.block1 = nn.Sequential()
        self.block1.add_module('linear1', self.fc1)
        self.block1.add_module("GELU", nn.GELU())
        self.block1.add_module('linear2', self.fc2)
        self.block1.add_module('GELU', nn.GELU())
        self.block1.add_module('linear2_duplicate', self.fc2_duplicate)
        self.block1.add_module('GELU', nn.GELU())
        self.block1.add_module('linear3', self.fc3_out)

    def forward(self, x:torch.Tensor):
        x = self.block1(x)
        xlist = [x.item()]
        for layer in self.one2oneLayer:
            x = layer(self.activation(x))
            xlist.append(x.item())
        return torch.tensor(xlist)
    

    def calculate_loss(self, input:torch.Tensor, target:torch.Tensor):
        """
        计算损失函数; input 和 target 的第一维度是数据长度，第二维度是数据维度; 
        计算方式为, 将 input 经过 self.block1 后, 每经过一层 one2oneLayer 就计算一次损失函数
        最后将所有的损失函数加和
        """
        assert input.shape[0] == target.shape[0] and len(self.one2oneLayer) == target.shape[2] - 1, \
            f"input shape is {input.shape}, target shape is {target.shape}, one2oneLayer nums is {len(self.one2oneLayer)}"
        input = self.block1(input)
        loss = mse_loss(input, target[..., 0])
        for i, layer in enumerate(self.one2oneLayer):
            input = layer(input)
            loss += mse_loss(input, target[..., i + 1])
        return loss


class Network_PSR_PlainMLP(nn.Module):
    """
    PSR 通过简单的一层 MLP 连接起来的网络; 训练难度和深度都较大
    """
    def __init__(self, layer_nums, input_dim, hidden_units, output_dim):
        super().__init__()
        # 定义隐藏层
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc2_duplicate  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc3_out  = nn.Linear(hidden_units[1], output_dim)
        self.activation = nn.GELU()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.GELU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.GELU(),
            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.GELU(),
            nn.Linear(hidden_units[2], output_dim)
        )
        
        # 正确注册 PSRblock
        self.PSRblock = nn.ModuleList()
        for _ in range(layer_nums - 1):
            block = nn.Sequential(
                self.activation,
                nn.Linear(output_dim, int(hidden_units[2] / 2)),
                self.activation,
                nn.Linear(int(hidden_units[2] / 2), output_dim)
            )
            self.PSRblock.append(block)
        
        # 初始化权重
        for layer in self.block1:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=np.sqrt(3))
        for block in self.PSRblock:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, a=np.sqrt(3))
                    
                    
    def forward(self, input:torch.Tensor):
        x = self.block1(input)
        xlist = x.clone().unsqueeze(0)
        for PSRblock in self.PSRblock:
            x = PSRblock(x)
            xlist = torch.cat((xlist, x.unsqueeze(0)), dim=0)
        # 将第一个维度和第二个维度对换
        if input.dim() == 2:
            xlist = torch.permute(xlist, (1, 2, 0))
        else:
            xlist = torch.swapaxes(xlist, 0, 1)
        return xlist


    def forward_squeeze(self, input:torch.Tensor):
        x = self.block1(input)
        xlist = x.clone().unsqueeze(0)
        for PSRblock in self.PSRblock:
            x = PSRblock(x)
            xlist = torch.cat((xlist, x.unsqueeze(0)), dim=0)
        # 将第一个维度和第二个维度对换
        if input.dim() == 2:
            xlist = torch.permute(xlist, (1, 2, 0))
        else:
            xlist = torch.swapaxes(xlist, 0, 1)
        xlist = torch.reshape(xlist, (xlist.shape[0], -1))
        return torch.tensor(xlist)
    

    def calculate_loss(self, input:torch.Tensor, target:torch.Tensor):
        """
        计算损失函数; input 和 target 的第一维度是数据长度，第二维度是数据维度; 
        计算方式为, 将 input 经过 self.block1 后, 每经过一层 self.PSRblock 就计算一次损失函数
        最后将所有的损失函数加和
        """
        assert input.shape[0] == target.shape[0] and len(self.PSRblock) == target.shape[2] - 1, \
            f"input shape is {input.shape}, target shape is {target.shape}, self.PSRblock nums is {len(self.PSRblock)}"
        input = self.block1(input)
        loss = mse_loss(input, target[..., 0])
        for i, PSRblock in enumerate(self.PSRblock):
            input = PSRblock(input)
            loss += mse_loss(input, target[..., i + 1])
        return loss


class Network_PlainDoubleHead(nn.Module):
    """
    专属版优化两个指标的二头网络
    具体结构为 input 输入后进入 3000 层的全连接，而后分别进入 1 和 2 的网络中
    """
    def __init__(self, input_dim:int, 
                 hidden_units:list[int|list[int]], 
                 output_dim:list[int|int|int],):
        # 定义隐藏层
        super().__init__()
        self.Net1_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net1block = nn.Sequential()
        self.Net1block.add_module("Net1_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_2", nn.Linear(hidden_units[2], output_dim[0]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net1_flatten1.weight)
        # nn.init.kaiming_normal_(self.Net1block.weight)
        
        self.Net2_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net2block = nn.Sequential()
        self.Net2block.add_module("Net2_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_2", nn.Linear(hidden_units[2], output_dim[1]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net2_flatten1.weight)
        # nn.init.kaiming_normal_(self.Net2_flatten2.weight)    
    
    def forward(self, x):
        x1 = self.Net1_flatten1(x)
        x1 = self.Net1block(x1)
        x2 = self.Net2_flatten1(x)
        x2 = self.Net2block(x2)
        return x1, x2
    
    def forward_Net1(self, x):
        x = self.Net1_flatten1(x)
        x = self.Net1block(x)
        return x
    
    def forward_Net2(self, x):
        x = self.Net2_flatten1(x)
        x = self.Net2block(x)
        return x
    

class Network_PlainTripleHead(nn.Module):
    """
    专属版优化两个指标的二头网络
    具体结构为 input 输入后进入 3000 层的全连接，而后分别进入 1 和 2 的网络中
    """
    def __init__(self, input_dim:int, 
                 hidden_units:list[int|list[int]], 
                 output_dim:list[int|int|int],):
        # 定义隐藏层
        super().__init__()
        self.Net1_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net1block = nn.Sequential()
        self.Net1block.add_module("Net1_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_2", nn.Linear(hidden_units[2], output_dim[0]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net1_flatten1.weight)
        # nn.init.kaiming_normal_(self.Net1block.weight)
        
        self.Net2_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net2block = nn.Sequential()
        self.Net2block.add_module("Net2_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_2", nn.Linear(hidden_units[2], output_dim[1]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net2_flatten1.weight)
        # nn.init.kaiming_normal_(self.Net2_flatten2.weight)    

        self.Net3_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net3block = nn.Sequential()
        self.Net3block.add_module("Net3_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net3block.add_module("Gelu", nn.GELU())
        self.Net3block.add_module("Net3_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net3block.add_module("Gelu", nn.GELU())
        self.Net3block.add_module("Net3_2", nn.Linear(hidden_units[2], output_dim[2]))
        
        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net3_flatten1.weight)
    
    def forward(self, x):
        x1 = self.Net1_flatten1(x)
        x1 = self.Net1block(x1)
        x2 = self.Net2_flatten1(x)
        x2 = self.Net2block(x2)
        x3 = self.Net3_flatten1(x)
        x3 = self.Net3block(x3)
        return x1, x2, x3
    
    def forward_Net1(self, x):
        x = self.Net1_flatten1(x)
        x = self.Net1block(x)
        return x
    
    def forward_Net2(self, x):
        x = self.Net2_flatten1(x)
        x = self.Net2block(x)
        return x

    def forward_Net3(self, x):
        x = self.Net3_flatten1(x)
        x = self.Net3block(x)
        return x


class Network_PlainTriple_PSR(nn.Module):
    """
    专属版优化两个指标的三头网络 + PSR 网络 Network_PSR_PlainMLP
    将 Network_PlainTripleHead 和 Network_PSR_PlainMLP 结合起来
    """
    def __init__(self, layer_nums:int, input_dim:int, 
                 hidden_units:list[int|list[int]], 
                 output_dim:list[int|int|int|int],):
        # 定义隐藏层
        super().__init__()
        self.Net1_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net1block = nn.Sequential()
        self.Net1block.add_module("Net1_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net1block.add_module("GELU", nn.GELU())
        self.Net1block.add_module("Net1_2", nn.Linear(hidden_units[2], output_dim[0]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net1_flatten1.weight)
        
        self.Net2_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net2block = nn.Sequential()
        self.Net2block.add_module("Net2_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net2block.add_module("GELU", nn.GELU())
        self.Net2block.add_module("Net2_2", nn.Linear(hidden_units[2], output_dim[1]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net2_flatten1.weight)

        self.Net3_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net3block = nn.Sequential()
        self.Net3block.add_module("Net3_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net3block.add_module("Gelu", nn.GELU())
        self.Net3block.add_module("Net3_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net3block.add_module("GELU", nn.GELU())
        self.Net3block.add_module("Net3_2", nn.Linear(hidden_units[2], output_dim[2]))
        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net3_flatten1.weight)
        
        self.Network_PSR = Network_PSR_PlainMLP(
            layer_nums = layer_nums,
            input_dim = input_dim,
            hidden_units = hidden_units,
            output_dim = output_dim[3]
        )
        
    def forward(self, x):
        x1 = self.Net1_flatten1(x)
        x1 = self.Net1block(x1)
        x2 = self.Net2_flatten1(x)
        x2 = self.Net2block(x2)
        x3 = self.Net3_flatten1(x)
        x3 = self.Net3block(x3)
        x4 = self.Network_PSR(x)
        return x1, x2, x3, x4
    
    def forward_Net1(self, x):
        x = self.Net1_flatten1(x)
        x = self.Net1block(x)
        return x
    
    def forward_Net2(self, x):
        x = self.Net2_flatten1(x)
        x = self.Net2block(x)
        return x
    
    def forward_Net3(self, x):
        x = self.Net3_flatten1(x)
        x = self.Net3block(x)
        return x
    
    def forward_Net4(self, x):
        return self.Network_PSR(x)


# PSRconcentration 使用的网络

class Network_PlainDoubleHead_PSRconcentration(nn.Module):
    """
    第2个 network 是用于 PSRconcentration 的网络
    """
    def __init__(self, input_dim:int, 
                 hidden_units:list[int|list[int]], 
                 output_dim:list[int|int|int],):
        # 定义隐藏层
        super().__init__()
        self.Net1_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net1block = nn.Sequential()
        self.Net1block.add_module("Net1_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_2", nn.Linear(hidden_units[2], output_dim[0]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net1_flatten1.weight)
        # nn.init.kaiming_normal_(self.Net1block.weight)
        
        self.Net2_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net2block = nn.Sequential()
        self.Net2block.add_module("Net2_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net2block.add_module("ReLU", nn.ReLU())
        self.Net2block.add_module("Net2_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net2block.add_module("ReLU", nn.ReLU())
        self.Net2block.add_module("Net2_2", nn.Linear(hidden_units[2], output_dim[1]))
        self.Net2block.add_module("ReLU", nn.ReLU())

        # 使用Kaiming初始化
        exp_kaiming_normal_(self.Net2_flatten1.weight, rate = 0.9)
        for weight in self.Net2block:
            if isinstance(weight, nn.Linear):
                exp_kaiming_normal_(weight.weight, rate = 0.9)
        # nn.init.kaiming_normal_(self.Net2_flatten2.weight)    
    
    def forward(self, x):
        x1 = self.Net1_flatten1(x)
        x1 = self.Net1block(x1)
        x2 = self.Net2_flatten1(x)
        x2 = self.Net2block(x2)
        return x1, x2
    
    def forward_Net1(self, x):
        x = self.Net1_flatten1(x)
        x = self.Net1block(x)
        return x
    
    def forward_Net2(self, x):
        x = self.Net2_flatten1(x)
        x = self.Net2block(x)
        return x
class Network_PlainTripleHead_PSRconcentration(nn.Module):
    """
    第三个 network 是用于 PSRconcentration 的网络
    """
    def __init__(self, input_dim:int, 
                 hidden_units:list[int|list[int]], 
                 output_dim:list[int|int|int],):
        # 定义隐藏层
        super().__init__()
        self.Net1_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net1block = nn.Sequential()
        self.Net1block.add_module("Net1_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_2", nn.Linear(hidden_units[2], output_dim[0]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net1_flatten1.weight)
        # nn.init.kaiming_normal_(self.Net1block.weight)
        
        self.Net2_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net2block = nn.Sequential()
        self.Net2block.add_module("Net2_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_2", nn.Linear(hidden_units[2], output_dim[1]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net2_flatten1.weight)
        # nn.init.kaiming_normal_(self.Net2_flatten2.weight)    

        self.Net3_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net3block = nn.Sequential()
        self.Net3block.add_module("Net3_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net3block.add_module("Gelu", nn.ReLU())
        self.Net3block.add_module("Net3_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net3block.add_module("Gelu", nn.ReLU())
        self.Net3block.add_module("Net3_2", nn.Linear(hidden_units[2], output_dim[2]))
        self.Net3block.add_module("Relu", nn.ReLU())
        
        # 使用Kaiming初始化
        exp_kaiming_normal_(self.Net3_flatten1.weight, rate = 0.9)
        for weight in self.Net3block:
            if isinstance(weight, nn.Linear):
                exp_kaiming_normal_(weight.weight, rate = 0.9)
    
    def forward(self, x):
        x1 = self.Net1_flatten1(x)
        x1 = self.Net1block(x1)
        x2 = self.Net2_flatten1(x)
        x2 = self.Net2block(x2)
        x3 = self.Net3_flatten1(x)
        x3 = self.Net3block(x3)
        return x1, x2, x3
    
    def forward_Net1(self, x):
        x = self.Net1_flatten1(x)
        x = self.Net1block(x)
        return x
    
    def forward_Net2(self, x):
        x = self.Net2_flatten1(x)
        x = self.Net2block(x)
        return x

    def forward_Net3(self, x):
        x = self.Net3_flatten1(x)
        x = self.Net3block(x)
        return x
class Network_PlainTriple_PSR_PSRconcentration(nn.Module):
    """
    专属版优化两个指标的三头网络 + PSR 网络 Network_PSR_PlainMLP
    将 Network_PlainTripleHead 和 Network_PSR_PlainMLP 结合起来
    """
    def __init__(self, layer_nums:int, input_dim:int, 
                 hidden_units:list[int|list[int]], 
                 output_dim:list[int|int|int|int],):
        # 定义隐藏层
        super().__init__()
        self.Net1_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net1block = nn.Sequential()
        self.Net1block.add_module("Net1_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net1block.add_module("Gelu", nn.GELU())
        self.Net1block.add_module("Net1_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net1block.add_module("GELU", nn.GELU())
        self.Net1block.add_module("Net1_2", nn.Linear(hidden_units[2], output_dim[0]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net1_flatten1.weight)
        
        self.Net2_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net2block = nn.Sequential()
        self.Net2block.add_module("Net2_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net2block.add_module("Gelu", nn.GELU())
        self.Net2block.add_module("Net2_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net2block.add_module("GELU", nn.GELU())
        self.Net2block.add_module("Net2_2", nn.Linear(hidden_units[2], output_dim[1]))

        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.Net2_flatten1.weight)

        self.Net3_flatten1 = nn.Linear(input_dim, hidden_units[0], bias = False)
        self.Net3block = nn.Sequential()
        self.Net3block.add_module("Net3_0", nn.Linear(hidden_units[0], hidden_units[1]))
        self.Net3block.add_module("ReLU", nn.ReLU())
        self.Net3block.add_module("Net3_1", nn.Linear(hidden_units[1], hidden_units[2]))
        self.Net3block.add_module("ReLU", nn.ReLU())
        self.Net3block.add_module("Net3_2", nn.Linear(hidden_units[2], output_dim[2]))
        self.Net3block.add_module("ReLU", nn.ReLU())
        # 使用Kaiming初始化
        exp_kaiming_normal_(self.Net3_flatten1.weight, rate = 0.9)
        for weight in self.Net3block:
            if isinstance(weight, nn.Linear):
                exp_kaiming_normal_(weight.weight, rate = 0.9)
    
        
        self.Network_PSR = Network_PSR_PlainMLP(
            layer_nums = layer_nums,
            input_dim = input_dim,
            hidden_units = hidden_units,
            output_dim = output_dim[3]
        )
        
    def forward(self, x):
        x1 = self.Net1_flatten1(x)
        x1 = self.Net1block(x1)
        x2 = self.Net2_flatten1(x)
        x2 = self.Net2block(x2)
        x3 = self.Net3_flatten1(x)
        x3 = self.Net3block(x3)
        x4 = self.Network_PSR(x)
        return x1, x2, x3, x4
    
    def forward_Net1(self, x):
        x = self.Net1_flatten1(x)
        x = self.Net1block(x)
        return x
    
    def forward_Net2(self, x):
        x = self.Net2_flatten1(x)
        x = self.Net2block(x)
        return x
    
    def forward_Net3(self, x):
        x = self.Net3_flatten1(x)
        x = self.Net3block(x)
        return x
    
    def forward_Net4(self, x):
        return self.Network_PSR(x)

"""=========================================================================================================="""
"""                                      Dataloader Structure Definition                                     """          
"""=========================================================================================================="""


class DATASET_SingleHead(Dataset):
    """
    自定义的 dataset，要求是输入和输出（均是 tensor）的第一维度是数据长度且相同
    
    如果使用索引的话会返回1个值，分别是 A, IDT
    """
    def __init__(self, data_A:torch.Tensor, data_QoI:torch.Tensor, device:str = "cpu") -> None:
        super().__init__()
        self.data_device = device
        if device == "cpu":
            self.data_A = data_A; self.data_QoI = data_QoI
        else:
            self.data_A = data_A.to(device); self.data_QoI = data_QoI.to(device)

    def __len__(self,):
        assert len(self.data_A) == len(self.data_QoI)
        return len(self.data_A)

    def __getitem__(self, index):
        return self.data_A[index], self.data_QoI[index]


class DATASET_DoubleHead(Dataset):
    """
    自定义的 dataset，要求是输入和输出（均是 tensor）的第一维度是数据长度且相同
    
    如果使用索引的话会返回4个值，分别是 A, QoI1, QoI2, QoI3
    如果给定 device = cuda 的话，会将数据转移到 cuda 上; 此时 DataLoader 的 num_workers 必须为 0 并关闭 pin_memory
    如果不给定 device 的话，会将数据转移到 cpu 上; 建议 num_workers = cpu 核心数, pin_memory = True
    """
    def __init__(self, data_A:torch.Tensor, data_QoI1:torch.Tensor, data_QoI2:torch.Tensor,
                 device:str = "cpu") -> None:
        super().__init__()
        self.data_device = device
        if device == "cpu":
            self.data_A = data_A; self.data_QoI1 = data_QoI1; self.data_QoI2 = data_QoI2
        else:
            self.data_A = data_A.to(device); self.data_QoI1 = data_QoI1.to(device); self.data_QoI2 = data_QoI2.to(device)

    def __len__(self,):
        assert len(self.data_A) == len(self.data_QoI1) == len(self.data_QoI2)
        return len(self.data_A)

    def __getitem__(self, index):
        return self.data_A[index], self.data_QoI1[index], self.data_QoI2[index]
    

class DATASET_TripleHead(Dataset):
    """
    自定义的 dataset，要求是输入和输出（均是 tensor）的第一维度是数据长度且相同
    
    如果使用索引的话会返回4个值，分别是 A, QoI1, QoI2, QoI3
    如果给定 device = cuda 的话，会将数据转移到 cuda 上; 此时 DataLoader 的 num_workers 必须为 0 并关闭 pin_memory
    如果不给定 device 的话，会将数据转移到 cpu 上; 建议 num_workers = cpu 核心数, pin_memory = True
    """
    def __init__(self, data_A:torch.Tensor, data_QoI1:torch.Tensor, data_QoI2:torch.Tensor, data_QoI3:torch.Tensor,
                 device:str = "cpu") -> None:
        super().__init__()
        device = torch.device(device)
        if device == "cpu":
            self.data_A = data_A; self.data_QoI1 = data_QoI1; self.data_QoI2 = data_QoI2
            self.data_QoI3 = data_QoI3
        else:
            self.data_A = data_A.to(device); self.data_QoI1 = data_QoI1.to(device); self.data_QoI2 = data_QoI2.to(device)
            self.data_QoI3 = data_QoI3.to(device)

    def __len__(self,):
        assert len(self.data_A) == len(self.data_QoI1) == len(self.data_QoI2) == len(self.data_QoI3), \
            f"data_A shape is {self.data_A.shape},  data_QoI1 shape is {self.data_QoI1.shape}, " \
            f"data_QoI2 shape is {self.data_QoI2.shape}, data_QoI3 shape is {self.data_QoI3.shape}"
        return len(self.data_A)

    def __getitem__(self, index):
        return self.data_A[index], self.data_QoI1[index], self.data_QoI2[index], self.data_QoI3[index]

    
class DATASET_FourHead(Dataset):
    """
    自定义的 dataset，要求是输入和输出（均是 tensor）的第一维度是数据长度且相同
    
    如果使用索引的话会返回4个值，分别是 A, QoI1, QoI2, QoI3
    如果给定 device = cuda 的话，会将数据转移到 cuda 上; 此时 DataLoader 的 num_workers 必须为 0 并关闭 pin_memory
    如果不给定 device 的话，会将数据转移到 cpu 上; 建议 num_workers = cpu 核心数, pin_memory = True
    """
    def __init__(self, data_A:torch.Tensor, data_QoI1:torch.Tensor, data_QoI2:torch.Tensor, data_QoI3:torch.Tensor, data_QoI4:torch.Tensor,
                 device:str = "cpu") -> None:
        super().__init__()
        device = torch.device(device)
        if device == "cpu":
            self.data_A = data_A; self.data_QoI1 = data_QoI1; self.data_QoI2 = data_QoI2
            self.data_QoI3 = data_QoI3; self.data_QoI4 = data_QoI4
        else:
            self.data_A = data_A.to(device); self.data_QoI1 = data_QoI1.to(device); self.data_QoI2 = data_QoI2.to(device)
            self.data_QoI3 = data_QoI3.to(device); self.data_QoI4 = data_QoI4.to(device)

    def __len__(self,):
        assert len(self.data_A) == len(self.data_QoI1) == len(self.data_QoI2) == len(self.data_QoI3)
        return len(self.data_A)

    def __getitem__(self, index):
        return self.data_A[index], self.data_QoI1[index], self.data_QoI2[index], self.data_QoI3[index], self.data_QoI4[index]
    