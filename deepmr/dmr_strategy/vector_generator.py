import numpy as np



def random_del_species_from_father_sample(
        father_sample: np.ndarray,
        target_size: int,
        del_zero_min: int,
        del_zero_max: int,
        retained_index: list
        ):
    r'''
    从父样本中随机删除组分，生成目标向量
    Args:
        ``father_sample``(``np.ndarray``): 父样本
        ``target_size``(``int``): 目标向量的数量
        ``del_zero_min``(``int``): 删除的最小组分数
        ``del_zero_max``(``int``): 删除的最大组分数
        ``retained_index``(``list``): 保留的组分索引
    Returns:
        ``target_vector``(``np.ndarray``): 目标向量
    '''
    
    father_sample_size = np.size(father_sample, 0)
    
    # 从father_sample里随机选向量随机删除组分获得数据集
    target_vector = np.ones((target_size, np.size(father_sample, 1)))
    
    for i in range(target_size):
        # 随机选取一个father_sample
        index = np.random.randint(father_sample_size)
        target_vector[i] = father_sample[index]

        # 对于retained_index不做操作(先设为0取得其他组分的nonzero_index，再设回1)
        target_vector[i, retained_index] = 0
        
        nonzero_index = np.nonzero(target_vector[i])[0]
        nonzero_size = np.size(nonzero_index)

        zero_num = np.random.randint(del_zero_min, del_zero_max + 1)
        # 如果要删除的组分数大于非零组分数，则只删除非零组分数个组分
        zero_num = min(nonzero_size, zero_num)
        
        index = np.random.choice(nonzero_size, zero_num, replace=False)
        for j in index:
            target_vector[i, nonzero_index[j]] = 0

    target_vector[:, retained_index] = 1
    return target_vector