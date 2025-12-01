import numpy as np
import os

from .vector_generator import random_vector_generator
from .DNN_training import dnn_training

from dmgr.base import DeePGR_base_class
from dmgr.utils import *
from dmgr.data import gendata_MPI, gendata_multiprocessing, \
                gather_data, generate_DNN_data

def dgr_ini(settings, get_data):
    r'''
    初始化工作目录，并生成真实数据和第一批子代
    '''

    # 生成文件夹结构
    working_dir = settings.working_dir
    gr = DeePGR_base_class(settings)

    ini_vector = [1] * settings.species_num

    # 生成真实数据
    if not os.path.exists(f'{working_dir}/data/true_data.npz'):
        true_data_dict = get_data(ini_vector, settings, return_type='specific')
        save_simulation_to_json(true_data_dict, f'{working_dir}/data/true_data.json')
        # 读取并保存为npz文件
        true_data = get_data(ini_vector, settings, return_mode='concerned', 
                              exist_data_path=f'{working_dir}/data/true_data.json')
        np.savez(f'{working_dir}/data/true_data.npz', **true_data)
    

def dgr_gen_vector(settings, iteration:int = None):
    r'''
    生成机理向量，第一次的向量不存在weight，后续的向量需要weight
    '''
    working_dir = settings.working_dir
    
    weight = np.load(f'{working_dir}/data/weight_data/weight_{iteration}.npy')

    # 生成机理向量并保存
    random_vector_generator(
            vector_lenth = settings.species_num,
            vector_size = settings.data_size,
            one_num = settings.expected_mechanism_size,
            retained_index = settings.retained_index,
            weight = weight,
            save = True,
            save_path = f'{working_dir}/data/vector_data/vector_{iteration}.npz',
    )

def dgr_prepare(settings, iteration:int = None):
    r'''
    准备工作，生成用于存放模拟数据的临时文件夹
    '''
    working_dir = settings.working_dir
    os.makedirs(f'{working_dir}/data/simulation_data/tmp', exist_ok = True)
    my_logger = Log(f'{working_dir}/log/gendata/iter_{iteration}.log', mode = 'w')


def dgr_gen_data(settings, get_data, iteration:int = None):
    r'''
    使用multiprocessing或MPI并行生成数据
    '''

    # 并行生成数据
    if settings.parallel_type == 'MPI':
        gendata_MPI(settings, get_data, iteration)
    elif settings.parallel_type == 'multiprocessing':
        gendata_multiprocessing(settings, get_data, iteration)


def dgr_gather_data(settings, iteration:int = None):
    # 汇总数据
    gather_data(settings, count = iteration, sparse_form = True)
    
def dgr_convert_data(settings, iteration:int = None):
    # 将模拟数据转化为DNN训练数据，仅转化需要DNN辅助的指标
    generate_DNN_data(settings, iteration, rate=0.8, sparse_form = True)
    
def dgr_gather_and_convert_data(settings, iteration:int = None):
    r'''
    汇总模拟数据，并将模拟数据转化为DNN训练数据
    '''
    # 汇总数据
    dgr_gather_data(settings, iteration)

    # 将模拟数据转化为DNN训练数据，仅转化需要DNN辅助的指标
    dgr_convert_data(settings, iteration)



def dgr_train_dnn(settings, iteration:int = None):
    r'''
    训练神经网络，并使用神经网络进行辅助筛选
    '''    
    # 训练神经网络
    dnn_training(iteration, settings, sparse_form = True)


def dgr_dnn_evaluate(settings, iteration:int = None):
    r'''
    按照weight生成新的机理，并用DNN进行打分
    计算组分的得分情况，获取下次生成数据的概率密度函数weight_{iteration+1}.npy
    '''
    working_dir = settings.working_dir
    if iteration == 0:
        weight = None
    else:
        weight = np.load(f'{working_dir}/data/weight_data/weight_{iteration}.npy')

    pass

