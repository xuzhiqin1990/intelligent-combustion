import torch.optim as optim

def get_optimizer(
    optimizer_type,
    lr,
    model,
    ):
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'不支持的优化器类型: {optimizer_type}')
    return optimizer
