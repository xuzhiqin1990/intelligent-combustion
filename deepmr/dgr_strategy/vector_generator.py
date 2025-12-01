import numpy as np
from typing import Union, Optional
import scipy.sparse as sp
from scipy.sparse import csr_matrix

def random_vector_generator(
        vector_lenth: int,
        vector_size: int,
        one_num: int,
        retained_index: list = [],
        weight: Union[list, np.ndarray, None] = None,
        save: bool = False,
        save_path: Optional[str] = None,
):
    r'''
    根据权重生成向量
    Args:
        ``vector_lenth``(``int``): 向量长度
        ``vector_size``(``int``): 向量数量
        ``one_num``(``int``): 1的数量
        ``retained_index``(``list``): 保留的组分索引
        ``weight``(``list``): 组分权重，和可以不为1，将用于生成选取组分的概率密度函数
    '''
    # 生成向量
    vector = np.zeros((vector_size, vector_lenth))

    # 对于retained_index必定保留
    vector[:, retained_index] = 1

    # 非必要保留的组分的索引
    zero_index = np.nonzero(vector[0] == 0)[0]

    # 需要生成的非必要保留组分的数量
    one_num = one_num - len(retained_index)

    # 生成选取组分的概率密度函数
    # 这个概率函数只考虑非必要保留的组分的权重
    if weight is None:
        weight = np.ones(len(zero_index))
    else:
        weight = np.array(weight[zero_index])

    weight = weight / np.sum(weight)

    # 按照概率密度函数选取组分
    for i in range(vector_size):
        index = np.random.choice(len(zero_index), one_num, replace=False, p=weight)
        for j in index:
            vector[i, zero_index[j]] = 1

    # 以稀疏矩阵的形式保存
    vector_sp = sp.coo_matrix(vector)

    if save:
        np.savez(save_path, 
                 vector_row = vector_sp.row,
                 vector_col = vector_sp.col,
                 vector_data = vector_sp.data,
                 vector_shape = vector_sp.shape,
                 weight = weight)

    return vector