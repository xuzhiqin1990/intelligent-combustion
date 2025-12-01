import os
import numpy as np
from dmgr.utils import get_loss, save_simulation_to_json
from dmgr.data import gendata_MPI, gather_data


r'''
DIS: detect ignorable species
这个代码用于初步检测哪些组分是相对不重要的。
可以在应用DeePMR和DeePGR之前使用，以减少计算量。
其方法是先通过逐一删除组分获得所有可能的机理。筛选出一些可以忽略的组分。
验证忽略全部这些组分后机理的性能。
'''

def dis_gen_vector(settings):
    r'''
    逐一删除组分获得所有可能的机理。
    强制保留的组分不用删除。
    '''
    working_dir = settings.working_dir
    vector = []
    for i in range(settings.species_num):
        if i in settings.retained_index:
            continue
        else:
            tmp_vector = np.ones(settings.species_num)
            tmp_vector[i] = 0
            vector.append(tmp_vector)

    os.makedirs(f'{working_dir}/data/del_one_sp', exist_ok=True)
    np.savez(f'{working_dir}/data/del_one_sp/vector.npz', vector = vector)


def dis_calculate_indicator(settings, get_data):
    r'''
    计算简化机理的指标
    '''
    working_dir = settings.working_dir
    if settings.parallel_type == 'MPI':
        gendata_MPI(settings, get_data, 
                    vector_path=f'{working_dir}/data/del_one_sp/vector.npz',
                    log_path=f'{working_dir}/data/del_one_sp/logger.log', 
                    save_path=f'{working_dir}/data/del_one_sp')



def dis_gather_simulation_data(settings):
    r'''
    汇总数据
    '''
    working_dir = settings.working_dir
    gather_data(settings, save_path = f'{working_dir}/data/del_one_sp', del_tmp_path=False)


def dis_del_and_retest(settings, get_data):
    r'''
    删除所有不重要的组分并回测得到的简化机理
    筛选评判标准由loss_form和dis_tol_list决定
    '''
    working_dir = settings.working_dir
    # 加载数据
    vector = np.ones(settings.species_num)
    true_data = np.load(f'{working_dir}/data/true_data.npz')
    red_data = np.load(f'{working_dir}/data/del_one_sp/simulation.npz')
    red_vector = red_data['vector']

    # 筛选不重要的组分
    negligible_sp_index = []
    for i in range(np.size(red_vector, 0)):
        tmp_vector = red_vector[i]
        this_one_is_good = True
        for j, indicator in enumerate(settings.indicators):
            config = getattr(settings, f'{indicator}_config')
            err = get_loss(np.array(true_data[indicator]), red_data[indicator][i], config.loss_form, 'max')

            if err > settings.dis_tol_list[j]:
                this_one_is_good = False
                break
        
        # 机理好说明删除这个组分没有影响
        if this_one_is_good:
            sp_index = np.nonzero(1-tmp_vector)[0]
            negligible_sp_index.extend(sp_index)

    # 删除不重要组分
    vector = np.ones(settings.species_num)
    vector[negligible_sp_index] = 0
    sp_num = int(np.sum(vector))
    print(f'delete {len(negligible_sp_index)} species, remain {sp_num} species.')

    # 回测
    datas = get_data(vector, settings, mode = 'simple', return_type = 'specific', true_data_path=f'{working_dir}/data/true_data.json')

    # 保存数据
    save_simulation_to_json(datas, f'{working_dir}/data/del_one_sp/{sp_num}sp_data.json', del_vector=False)

    # 读取数据，获取一个包含计算指标的简单字典，方便比较误差和保存为npz
    true_data = get_data(vector, settings, mode = 'simple', 
                     exist_data_path=f'{working_dir}/data/true_data.json')
    datas = get_data(vector, settings, mode = 'simple',
                     true_data_path=f'{working_dir}/data/true_data.json',
                     exist_data_path=f'{working_dir}/data/del_one_sp/{sp_num}sp_data.json')

    # 计算并输出误差
    for indicator in settings.indicators:
        config = getattr(settings, f'{indicator}_config')
        err = get_loss(np.array(true_data[indicator]), np.array(datas[indicator]), config.loss_form, 'max')
        print(f'{indicator} error: {err}')

    # 保存为npz文件
    np.savez(f'{working_dir}/data/del_one_sp/{sp_num}sp_data.npz', **datas)