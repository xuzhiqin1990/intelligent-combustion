import os
import numpy as np
import torch
import os
import time
from dmgr.utils import *
from dmgr.data import *
from dmgr.base.DeePMR_base import DeePMR_base_class
from .father_sample_select import father_sample_select
from .vector_generator import random_del_species_from_father_sample
from dmgr.data import load_dnn_data_x, load_dnn_data_y
from dmgr.models import *

def data_diff(predict, true_data, loss_form):
    predict = np.array(predict)
    true_data = np.array(true_data)

    if loss_form == 'abs_log10':
        diff = np.abs(np.log10(predict) - np.log10(true_data))
    elif loss_form == 'abs_none':
        diff = np.abs(predict - true_data)
    elif loss_form == 'abs_log':
        diff = np.abs(np.log(predict) - np.log(true_data))
    elif loss_form == 'rel_none':
        diff = np.abs(predict - true_data) / (true_data+1e-10)
    else:
        raise ValueError(f'不支持的loss_form: {loss_form}')
    return diff


def dnn_assist_screening(
        count,
        settings,
        ):
    r'''
    DNN辅助筛选
    Args:
        ``mr``(``DeePMR_base_class``): DeePMR的基类
        ``count``(``int``): 第几次迭代
        ``global_config``(``GlobalConfig``): 全局配置文件
        ``indicator_config_list``(``list``): 指标配置文件列表
    '''
    mr = DeePMR_base_class(settings)

    # 批量生成备选向量，防止内存不足
    batch_size = settings.gen_batch

    # 期望最终获得的简化机理的数量
    target_size = settings.size_per_iteration
    
    # 权重列表，将用于father_sample的选择
    weight_list = settings.weight_list

    screening_logger = Log(f"{mr.log_screening_path}/iter_{count}.log")

    # 加载上一代向量，获取zero_num
    last_vector = np.load(f'{mr.vector_data_path}/vector_{count}.npz')
    zero_min = last_vector['zero_min']
    zero_max = last_vector['zero_max']
    
    # 动态调整del_zero_step
    sp_num_min = settings.species_num - zero_max
    sp_num_max = settings.species_num - zero_min
    del_zero_min = int(settings.sp_num_decay_min * sp_num_min)
    del_zero_max = int(settings.sp_num_decay_max * sp_num_max)

    # device = torch.device(settings.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载真实数据
    true_data = np.load(f'{mr.working_dir}/data/true_data.npz')

    # 生成父样本
    father_sample = father_sample_select(
        settings,
        count,
        weight_list,
        save_path = mr.father_sample_data_path)

    # 获取输入维度
    _, _, input_dim = load_dnn_data_x(mr.dnn_input_data_path + f'/dnn_{count}.npz')

    # 激活训练的神经网络
    act_model_list = []
    for i, indicator in enumerate(settings.indicators):

        if settings.DNN_assist[i] == True:
            # 加载数据
            _, _, output_dim = load_dnn_data_y(mr.dnn_data_path[indicator] + f'/dnn_{count}.npz')

            # 激活网络
            config = getattr(settings, f'{indicator}_config')
            
            model = load_dnn_model(config.model_name, config.hidden_units, input_dim, output_dim)

            model = model.to(device)
            checkpoint = torch.load(
                f'{mr.model_pth_path[indicator]}/model_{count+1}.pth', map_location=device)
            model.load_state_dict(checkpoint['model'])  # 实例化网络
            act_model_list.append(model)
        else:
            act_model_list.append(None)

    t0 = time.time()
    screen_count = 0  # 记录预测的次数
    target_vector = []

    while len(target_vector) < target_size + 1:

        # 在原来的数据集上删组分获得备用数据集
        pre_input = random_del_species_from_father_sample(
            father_sample, batch_size, del_zero_min, del_zero_max, settings.retained_index)
        pre_input_list = pre_input.tolist()

        Remain = np.ones(np.size(pre_input, 0))
        for i, indicator in enumerate(settings.indicators):

            # 计算预测区间
            bound = max(config.bound_ini - (config.bound_decay_rate * count), config.bound_min)

            # 如果过长时间都没有预测到符合条件的机理，则适当放宽bound
            # TODO: 应该将其设置为超参数
            bound += screen_count // 2000 * 0.05

            tmp_true_data = true_data[indicator]

            # 仅对需要DNN筛选的指标进行操作
            if settings.DNN_assist[i] == True: 
                config = getattr(settings, f'{indicator}_config')
                
                # 获取tol，tol为超过预设边界的工况个数
                tol = config.tolerance * len(true_data[indicator])

                model = act_model_list[i]
                # 备用数据集经过迭代的DNN筛选获得数据集
                predict = model(torch.FloatTensor(pre_input).to(
                    device)).cpu().detach().numpy()
                
                # 不同指标对于true_data的数据处理方式不同
                predict = data_transform_reverse(predict, config.data_transform)

                # TODO: 有些指标的预测值需要经过数据变换才能和真实值比较，选择标准代码需要进一步优化
                err = data_diff(predict, tmp_true_data, config.dnn_screen_loss_form)
                exceed_bound = np.sum(1 * (err > bound), 1)  # batch_size维
                # 超过预设边界的工况个数不超过tol的则被采用
                remain = 1 * (exceed_bound <= tol)
                Remain *= remain


        # 将符合条件的机理加入target_vector
        for i in range(np.size(pre_input, 0)):
            if Remain[i] == 1:
                target_vector.append(pre_input_list[i])
        screen_count += 1
        screening_logger.info(
            f'{screen_count}: vector size: {len(target_vector)}')

    target_vector = target_vector[: target_size]

    # 计算zero_num的最大值和最小值以便保存
    zero_num = mr.species_num - np.sum(np.array(target_vector), 1)
    zero_min, zero_max = int(np.min(zero_num)), int(np.max(zero_num))

    vector_size = len(target_vector)
    screening_logger.info(f'{vector_size} vectors are selected, time cost: {time.time()-t0:.2f} s')
    vector_path = f'{mr.vector_data_path}/vector_{count+1}.npz'

    np.savez(vector_path,
            vector=target_vector, 
            vector_size=vector_size,
            zero_min=zero_min, 
            zero_max=zero_max)
    
    species_num_min = settings.species_num - zero_max
    species_num_max = settings.species_num - zero_min

    # 在全局log文件中保存信息，以便后续分析
    info_logger = Log(f'{mr.working_dir}/log/info.log')
    info_logger.info(
    f'iter {count} vector_size: {vector_size} zero_num: {zero_min} - {zero_max} species_num: {species_num_min} - {species_num_max} vector_path: {vector_path}'
    )
    
