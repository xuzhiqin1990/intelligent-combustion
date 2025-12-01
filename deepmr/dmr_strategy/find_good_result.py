import os
import numpy as np
from dmgr.utils import get_loss, Log

def find_good_result(settings, start_iteration, end_iteration, mode = 'find'):
    r'''
    find: 在所有数据中寻找好的结果
    debug: 如果一直找不到，则使用debug模式，逐个检查组分符合要求的机理的误差是什么样子
    '''

    my_logger = Log(f'{settings.working_dir}/log/find_good_result.log', mode = 'w')

    # 加载真实数据
    true_data = np.load(f'{settings.working_dir}/data/true_data.npz')

    # 寻找符合预期的结果
    good_result = []
    result_error = []
    # 加载所有simulation data
    simulation_data_path = f'{settings.working_dir}/data/simulation_data'

    if mode == 'find':
        # for files in os.listdir(simulation_data_path):
        for k in range(start_iteration, end_iteration + 1):
            my_logger.info(f'loading {simulation_data_path}/simulation_{k}.npz')

            # 加载simulation data
            tmp_data_path = f'{simulation_data_path}/simulation_{k}.npz'
            datas = np.load(tmp_data_path)

            vector = datas['vector']
            vector_size = np.size(vector, 0)
            sp_num = np.sum(vector, 1)

            for i in range(vector_size):
                is_good_result = True
                if sp_num[i] >= settings.fgr_tol_list[-1]:
                    is_good_result = False
                    continue

                # 计算误差，如果超过阈值则跳过
                err_list = []
                for j, indicator in enumerate(settings.indicators):
                    config = getattr(settings, f'{indicator}_config')
                    err = get_loss(np.array(true_data[indicator]), np.array(datas[indicator][i]), 
                                    config.loss_form, 'mean')
                    if err >= settings.fgr_tol_list[j]:
                        is_good_result = False
                        break
                    else:
                        err_list.append(err)

                if is_good_result:
                    good_result.append(vector[i])
                    result_error.append(err_list)

                    tmp_sp_num = int(np.sum(vector[i]))

                    my_logger.info(f'species num:{tmp_sp_num}, error:{err_list}')
        # 去重
        unique_result, unique_index = np.unique(good_result, axis=0, return_index = True)
        unique_result_error = [result_error[i] for i in unique_index]

        # 保存结果
        np.savez(f'{settings.working_dir}/data/good_result_{len(good_result)}.npz', 
            vector = unique_result, error = unique_result_error)
 
        # 打印结果
        my_logger.info(f'共找到{len(good_result)}个符合预期的结果，不重复结果有{len(unique_result)}个')
        for i, vector in enumerate(unique_result):
            sp_num = int(np.sum(vector))
            my_logger.info(f'index: {i}, species num: {sp_num}')
            for j, indicator in enumerate(settings.indicators):
                my_logger.info(f'{indicator}: {unique_result_error[i][j]:.6f}')
    
    elif mode == 'debug':
        # debug模式
        for k in range(start_iteration, end_iteration + 1):
            # 加载simulation data
            tmp_data_path = f'{simulation_data_path}/simulation_{k}.npz'
            datas = np.load(tmp_data_path)

            vector = datas['vector']
            vector_size = np.size(vector, 0)
            sp_num = np.sum(vector, 1)

            # 打印出组分符合预期的所有机理的误差
            for i in range(vector_size):
                if sp_num[i] <= settings.fgr_tol_list[-1]:

                    # 计算误差，如果超过阈值则跳过
                    err_list = []
                    for j, indicator in enumerate(settings.indicators):
                        config = getattr(settings, f'{indicator}_config')
                        err = get_loss(np.array(true_data[indicator]), np.array(datas[indicator][i]), 
                                        config.loss_form, 'mean')
                        err_list.append(err)
                    my_logger.info(int(sp_num[i]), err_list)



