import numpy as np
import os

from .vector_generator import random_del_species_from_father_sample
from .DNN_training import dnn_training
from .DNN_assist_screening import dnn_assist_screening
from .find_good_result import find_good_result

from dmgr.base import DeePMR_base_class
from dmgr.utils import *
from dmgr.data import gendata_MPI, gendata_multiprocessing, \
                gather_data, generate_DNN_data

def dmr_ini(settings, get_data):
    r'''
    初始化工作目录，并生成真实数据和第一批子代
    '''

    # 生成文件夹结构
    working_dir = settings.working_dir
    mr = DeePMR_base_class(settings)
    ini_vector = [1] * settings.species_num

    # 生成真实数据
    if not os.path.exists(f'{working_dir}/data/true_data.npz'):
        true_data_dict = get_data(ini_vector, settings, return_type='specific')
        save_simulation_to_json(true_data_dict, f'{working_dir}/data/true_data.json')
        # 读取并保存为npz文件
        true_data = get_data(ini_vector, settings, return_mode='concerned', 
                              exist_data_path=f'{working_dir}/data/true_data.json')
        np.savez(f'{working_dir}/data/true_data.npz', **true_data)
    
    try:
        start_vector = np.load(settings.start_vector_path)['vector']
    except:
        start_vector = ini_vector

    # 生成第一批子代
    del_zero_min = int(settings.sp_num_decay_min * np.sum(start_vector))
    del_zero_max = int(settings.sp_num_decay_max * np.sum(start_vector))
    vector = random_del_species_from_father_sample(
            np.array([start_vector]),
            settings.size_per_iteration,
            del_zero_min,
            del_zero_max,
            settings.retained_index)
    np.savez(f'{working_dir}/data/vector_data/vector_{0}.npz',
             vector = vector,
             vector_size = np.size(vector, 0),
             zero_min = del_zero_min,
             zero_max = del_zero_max)
    print(np.shape(vector), np.sum(start_vector), del_zero_min, del_zero_max)


def dmr_prepare(settings, iteration:int = None):
    r'''
    准备工作，生成用于存放模拟数据的临时文件夹
    '''
    working_dir = settings.working_dir
    os.makedirs(f'{working_dir}/data/simulation_data/tmp', exist_ok = True)
    my_logger = Log(f'{working_dir}/log/gendata/iter_{iteration}.log', mode = 'w')


def dmr_gen_data(settings, get_data, iteration:int = None, batch_id:int = 0, batch_num:int = 1):
    r'''
    使用multiprocessing或MPI并行生成数据
    '''
    # 并行生成数据
    if settings.parallel_type == 'MPI':
        gendata_MPI(settings, get_data, iteration, batch_id=batch_id, batch_num=batch_num)
    elif settings.parallel_type == 'multiprocessing':
        gendata_multiprocessing(settings, get_data, iteration)


def dmr_gather_data(settings, iteration:int = None):
    # 汇总数据
    gather_data(settings, count = iteration)
    
def dmr_convert_data(settings, iteration:int = None):
    # 将模拟数据转化为DNN训练数据，仅转化需要DNN辅助的指标
    generate_DNN_data(settings, iteration, rate = 0.8)


def dmr_gather_and_convert_data(settings, iteration:int = None):
    r'''
    汇总模拟数据，并将模拟数据转化为DNN训练数据
    '''
    # 汇总数据
    dmr_gather_data(settings, iteration)

    # 将模拟数据转化为DNN训练数据，仅转化需要DNN辅助的指标
    dmr_convert_data(settings, iteration)


def dmr_train_dnn(settings, iteration:int = None):
    r'''
    训练神经网络
    '''
    dnn_training(iteration, settings)


def dmr_dnn_screen(settings, iteration:int = None):
    r'''
    使用神经网络进行辅助筛选
    '''
    dnn_assist_screening(iteration, settings)


def dmr_train_and_screen(settings, iteration:int = None):
    r'''
    训练神经网络，并使用神经网络进行辅助筛选
    '''  
    # 训练神经网络
    dnn_training(iteration, settings)

    # 使用神经网络完成辅助预测
    dnn_assist_screening(iteration, settings)


def dmr_find_good_result(settings, start_iteration, end_iteration, mode: str = 'find'):
    r'''
    寻找最优结果
    '''
    find_good_result(settings, start_iteration, end_iteration, mode)