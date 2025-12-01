import os, sys
sys.path.append('..')
import numpy as np
from .dsc_utils import update_reduced_mechanism_mpi, get_best_vector_info, gather_data
from dmgr.utils import save_simulation_to_json

r'''
这个代码用于机理简化完成后，通过列举组合删除组分的方式，进一步删减无用机理。
适用于机理简化后，机理规模仍然较大的情况。
working_dir中包含[chem].yaml, config.yaml, ini_vector.npz, data/true_data.npz
在执行完删除组分后，最好的结果被保存在best_vector.npz中。
当想继续进一步删除组分时，可以将best_vector.npz中的最优结果作为ini_vector.npz的初始值。
'''


def dsc_ini(settings, get_data, get_species_name, get_reduced_sp_reac):
    r'''
    1. 计算真实数据
    2. 计算并保存初始简化机理与真实数据的误差
    '''
    working_dir = settings.working_dir
    os.makedirs(f'{working_dir}/data/dsc_data', exist_ok=True)

    # 如果没有ini_vector，则将全1向量保存为ini_vector
    detailed_vector = [1] * settings.species_num
    
    # 生成真实数据
    if not os.path.exists(f'{working_dir}/data/true_data.npz'):
        true_datas = get_data(detailed_vector, settings, return_type='specific')
        save_simulation_to_json(true_datas, f'{working_dir}/data/true_data.json')

        # 读取并保存为npz文件
        true_data = get_data(detailed_vector, settings, mode = 'simple', return_mode='concerned', 
                            true_data_path=f'{working_dir}/data/true_data.json',
                            exist_data_path=f'{working_dir}/data/true_data.json')
        np.savez(f'{working_dir}/data/true_data.npz', **true_data)
    
    # 如果没有ini_vector，则将全1向量保存为ini_vector，同时没必要再计算误差（因为误差是0）
    if not os.path.exists(f'{working_dir}/ini_vector.npz'):
        np.savez(f'{working_dir}/ini_vector.npz', vector = detailed_vector)
    # 如果是第一次运行(没有best_vector)，需要计算并保存初始简化机理与真实数据的误差
    if not os.path.exists(f'{working_dir}/best_vector.npz'):
        ini_vector = np.load(f'{working_dir}/ini_vector.npz')['vector']
        simulation_data = get_data(ini_vector, settings, mode = 'simple')
        get_best_vector_info(settings, simulation_data, get_species_name, get_reduced_sp_reac)


def dsc_update_reduced_mechanism(settings, get_data, get_species_name, get_reduced_sp_reac):
    r'''
    做一次迭代，组合删除组分，保存并更新最好的机理。
    '''
    update_reduced_mechanism_mpi(settings, get_data, get_species_name, get_reduced_sp_reac)



def dsc_gather_data(settings, get_species_name, get_reduced_sp_reac):

    gather_data(settings, get_species_name, get_reduced_sp_reac)