import time
import torch
import numpy as np
from .json_io import read_json_data
from scipy.sparse import csr_matrix
    


def get_timestr():
    return time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))

def get_loss(
        data1,
        data2,
        loss_type: str = 'relative',
        return_type: str = 'mean',
        ):
    r'''
    按照给定的loss类型，获取data1和data2之间的差距
    '''
    if loss_type == 'relative':
        loss = np.abs(data1 - data2) / (data1+1e-10)
    elif loss_type == 'absolute':
        loss = np.abs(data1 - data2)
    else:
        raise ValueError('loss_type not found!')

    if return_type == 'mean':
        loss = np.mean(loss)
    elif return_type == 'max':
        loss = np.max(loss)
    elif return_type == 'sum':
        loss = np.sum(loss)
    elif return_type == 'none':
        pass
    else:
        raise ValueError('return_type not found!')
    
    return loss

def output_loss(settings, simulation_datas, true_datas, return_loss_list = False):
    r'''
    计算并输出simulation_datas和true_datas之间的的loss
    '''
    loss = 0
    loss_list = []
    for i, indicator in enumerate(settings.indicators):
        config = getattr(settings, f'{indicator}_config')
        tmp_loss = get_loss(data1 = np.array(true_datas[indicator]), data2 = np.array(simulation_datas[indicator]), loss_type=config.loss_form)
        loss += settings.dsc_weight[i] * tmp_loss
        loss_list.append(tmp_loss)
    if return_loss_list:
        return loss, loss_list
    else:
        return loss

def compare_loss(settings, datas1, datas2):
    r'''
    比较datas1和datas2的loss，返回loss较小的那个
    '''
    working_dir = settings.working_dir
    true_datas = np.load(f'{working_dir}/data/true_data.npz')
    loss1 = output_loss(settings, datas1, true_datas)
    loss2 = output_loss(settings, datas2, true_datas)
    if loss1 < loss2:
        return datas1
    else:
        return datas2


def get_best_pth(model_path):
    Data = read_json_data(f'{model_path}/settings.json')
    best_epoch = int(Data['stop_epoch'])
    return f'{model_path}/model_pth/model{best_epoch}.pth'

def load_dnn(model, model_json_path, model_pth_path, device):
    json_data = read_json_data('%s/settings.json' % (model_json_path))
    my_model = model(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
    checkpoint = torch.load(model_pth_path, map_location = device)
    my_model.load_state_dict(checkpoint['model']) # 实例化网络
    return my_model

def load_dnn_cuda(model, model_json_path, model_pth_path):
    json_data = read_json_data('%s/settings.json' % (model_json_path))
    my_model = model(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).cuda()
    checkpoint = torch.load(model_pth_path)
    my_model.load_state_dict(checkpoint['model']) # 实例化网络
    return my_model


def load_vector_sp(
        vector_data_path,
        return_sparse: bool = False
        ):
    Data = np.load(vector_data_path)
    row = Data['vector_row']
    col = Data['vector_col']
    data = Data['vector_data']
    shape = Data['vector_shape']

    vector_sp = csr_matrix((data, (row, col)), shape=shape)
    if return_sparse:
        return vector_sp
    else:
        vector = vector_sp.toarray()
        return vector


