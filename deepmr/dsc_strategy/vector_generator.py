import numpy as np
import os
import itertools as it
import random

def delete_combination(settings):
    working_dir = settings.working_dir

    try:
        Data = np.load(os.path.join(working_dir, 'best_vector.npz'))
    except:
        Data = np.load(os.path.join(working_dir, 'ini_vector.npz'))
    vector = Data['vector']

    # settings.retained_species中的组分不能删除，先设为0，删完了再设回1
    vector[settings.retained_index] = 0
    nonzero_index = np.nonzero(vector)[0]    # 可删组分的编号

    del_max = min(np.sum(nonzero_index)-1, settings.del_per_iteration_max)
    
    # 所有可能情况的组合
    Delete_Index = []
    for del_num in range(1, del_max+1):
        delete_index = list(it.combinations(nonzero_index, del_num))  # nonzero_index中取del_num个数的所有情况
        Delete_Index.extend(delete_index)
    
    random.shuffle(Delete_Index)
    vector_size = len(Delete_Index)
    all_vector = np.ones((vector_size, 1)) * vector

    for i in range(vector_size):
        all_vector[i, list(Delete_Index[i])] = 0

    all_vector[:, settings.retained_index] = 1
    
    return all_vector
