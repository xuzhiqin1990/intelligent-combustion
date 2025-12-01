import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

# 将当前数据转化成DNN的数据集
def generate_DNN_data(
        settings, 
        iteration, 
        rate = 0.8,
        sparse_form = False,
        ):
    r'''
    将当前simulation data数据转化成DNN的数据集
    Args:
        ``global_config``(``GlobalConfig``): 全局配置文件
        ``iteration``(``int``): 第几次迭代
        ``rate``(``float``): 训练集和测试集的比例   
        ``sparse_form``(``bool``): 若为True，返回的将是稀疏矩阵
    '''

    # 加载simulation data
    simulation_data_path = f'{settings.working_dir}/data/simulation_data/simulation_{iteration}.npz'
    datas = np.load(simulation_data_path)
    
    # 加载vector
    if sparse_form:
        vectors = csr_matrix((datas['vector_data'], (datas['vector_row'], datas['vector_col'])),
                                shape=datas['vector_shape'])
    else:
        vectors = np.array(datas['vector'])
        zero_num = np.sum(vectors, 1)
        zero_min, zero_max = int(np.min(zero_num)), int(np.max(zero_num))


    vector_size = np.size(vectors, 0)

    # 输入数据
    train_size = int(rate * vector_size)
    x_train = vectors[0: train_size, :]
    x_test = vectors[train_size:, :]


    if sparse_form:
        x_train_row, x_train_col, x_train_data = sp.find(x_train)
        x_test_row, x_test_col, x_test_data = sp.find(x_test)
        np.savez(f'{settings.working_dir}/data/dnn_data/input/dnn_{iteration}.npz',
                    count=iteration,
                    x_train_row=x_train_row,
                    x_train_col=x_train_col,
                    x_train_data=x_train_data,
                    x_train_shape=x_train.shape,
                    x_test_row=x_test_row,
                    x_test_col=x_test_col,
                    x_test_data=x_test_data,
                    x_test_shape=x_test.shape)
    else:
        np.savez(f'{settings.working_dir}/data/dnn_data/input/dnn_{iteration}.npz',
                    count=iteration,
                    x_train=x_train,
                    x_test=x_test,
                    zero_min = zero_min,
                    zero_max = zero_max)
    # 仅转化需要DNN辅助的指标
    for i, indicator in enumerate(settings.indicators):
        if settings.DNN_assist[i] == True:
            config = getattr(settings, f'{indicator}_config')
            
            simulation_data = np.array(datas[indicator])

            y_train = data_transform(simulation_data[0: train_size, :], config.data_transform)
            y_test = data_transform(simulation_data[train_size:, :], config.data_transform)

            np.savez(f'{settings.working_dir}/data/dnn_data/{indicator}/dnn_{iteration}.npz',
                        count=iteration,
                        y_train=y_train,
                        y_test=y_test)


def data_transform(data, transform_type):
    if transform_type == 'log10':
        data = np.log10(data+1e-10)
    elif transform_type == 'log10+1':
        data = np.log10(data + 1)
    elif transform_type == 'log10+1+abs':
        data = np.log10(np.abs(data) + 1)
    elif transform_type == 'none':
        pass
    else:
        raise ValueError(f'不支持的数据转换类型: {transform_type}')
    return data

def data_transform_reverse(data, transform_type):
    if transform_type == 'log10':
        data = np.power(10, data) - 1e-10
    elif transform_type == 'log10+1':
        data = np.power(10, data) - 1
    elif transform_type == 'log10+1+abs':
        data = np.power(10, data) - 1
        data = np.sign(data) * (np.abs(data) - 1)
    elif transform_type == 'none':
        pass
    else:
        raise ValueError(f'不支持的数据转换类型: {transform_type}')
    return data


def load_dnn_data_x(
        dnn_data_path: str, 
        sparse_form: bool = False
        ):
    r'''
    加载DNN训练数据的输入
    Args:
        ``dnn_data_path``(``str``): DNN训练数据的路径
        ``sparse_form``(``bool``): 若为True，返回的将是稀疏矩阵
    '''
    if sparse_form:
        Data = np.load(dnn_data_path)
        x_train = csr_matrix((Data['x_train_data'], (Data['x_train_row'], Data['x_train_col'])),
                             shape=Data['x_train_shape'])
        x_test = csr_matrix((Data['x_test_data'], (Data['x_test_row'], Data['x_test_col'])),
                            shape=Data['x_test_shape'])
        vector_length = x_train.shape[1]
    else:
        Data = np.load(dnn_data_path)
        x_train, x_test = Data['x_train'], Data['x_test']
        vector_length = np.size(x_train, 1)

    return x_train, x_test, vector_length

def load_dnn_data_y(
        dnn_data_path: str
        ):
    r'''
    加载DNN训练数据的标签
    Args:
        ``dnn_data_path``(``str``): DNN训练数据的路径
    '''
    Data = np.load(dnn_data_path)
    y_train, y_test = Data['y_train'], Data['y_test']
    vector_length = np.size(y_train, 1)

    return y_train, y_test, vector_length