from .model import *

def load_dnn_model(
        model_name,
        hidden_units,
        input_dim,
        output_dim,
):
    r'''
    加载DNN模型
    Args:
        ``model_name``(``str``): 模型名称
    '''
    if model_name == 'three_layer_DNN':
        model = three_layer_DNN(input_dim, hidden_units, output_dim)
    elif model_name == 'MLP':
        model = MLP_Layer(input_dim, hidden_units, output_dim)
    else:
        raise ValueError(f'不支持的模型名称: {model_name}')
    
    return model