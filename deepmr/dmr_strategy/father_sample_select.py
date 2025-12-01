import numpy as np
from dmgr.utils import get_loss, Log

def father_sample_select(
        settings,
        count: int, 
        weight_list: list,
        save_path: str # self.father_sample_data_path
        ):
    r"""
    选择父代样本，筛选标准为加权误差最小的样本
    Args:
        ``settings``: 全局配置文件
        ``count``(``int``): 第几次迭代
        ``weight_list``(``list``): 权重列表
        ``save_path``(``str``): 保存路径
    Returns:
        ``father_sample``(``np.ndarray``): 父代样本
    """
    working_dir = settings.working_dir

    # 加载真实数据
    true_data = np.load(f'{working_dir}/data/true_data.npz')
    simulation_data = np.load(f'{working_dir}/data/simulation_data/simulation_{count}.npz')

    vector = simulation_data['vector']
    vector_size = np.size(vector, 0)

    father_sample_size = int(vector_size * 0.05)
    sample_size_per_weight = int(father_sample_size / len(weight_list))


    # 在不同权重下寻找最优的father_sample
    argsort_list = []
    for weight in weight_list:
        loss = np.zeros(vector_size)
        for i, indicator in enumerate(settings.indicators):
            config = getattr(settings, f'{indicator}_config')
            tmp_loss = get_loss(data1 = true_data[indicator], data2 = simulation_data[indicator], 
                                loss_type = config.loss_form)
            loss += weight[i] * tmp_loss
        
        argsort = np.argsort(loss)[: sample_size_per_weight].tolist()
        argsort_list.extend(argsort)

    argsort = np.array(argsort_list)
    father_sample = vector[argsort]

    
    # 保存father_sample
    Father_sample = {}
    Father_sample['father_sample'] = father_sample
    for indicator in settings.indicators:
        Father_sample[indicator] = simulation_data[indicator][argsort]
    np.savez(f'{save_path}/{count}.npz', **Father_sample) 

    # TODO: 在info.log中输出最好的father_sample各指标的loss信息
    # info_logger = Log(f'{working_dir}/log/info.log')
    # # 对每个指标输出误差
    # for i, indicator in enumerate(settings.indicators):
    #     config = getattr(settings, f'{indicator}_config')
    #     if config.loss_form == 'relative':
    #         info_logger.info(f'{indicator} loss: {100*loss_list[i]:.2f} %, loss form: {config.loss_form}')
    #     elif config.loss_form == 'absolute':
    #         info_logger.info(f'{indicator} loss: {loss_list[i]:.2f}, loss form: {config.loss_form}')

    return father_sample