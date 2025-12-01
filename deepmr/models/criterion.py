import torch.nn as nn

def get_criterion(
        criterion_type,
        ):
    if criterion_type == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
    elif criterion_type == 'L1':
        criterion = nn.L1Loss(reduction='mean')
    elif criterion_type == 'SmoothL1':
        criterion = nn.SmoothL1Loss(reduction='mean')
    else:
        raise ValueError(f'不支持的损失函数类型: {criterion_type}')
    return criterion
