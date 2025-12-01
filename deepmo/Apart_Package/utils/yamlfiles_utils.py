import yaml, re
import numpy as np
import cantera as ct
from typing import Callable, Iterable, Union, Tuple
import copy, torch
# 导入 Python 的 Number 数据类型
from numbers import Number

"""===================================================================================================="""
"""                                           Sampling                                                 """
"""===================================================================================================="""

def sample_constant_A(size:int, A0:np.ndarray, l_alpha:float|np.ndarray|list|tuple[float,float]|tuple[list, list], r_alpha:float|np.ndarray|list|tuple[float,float]|tuple[list, list], save_path = None, n_latin:int = 0, reduced_mech_A0 = None, max_uncertainty_range = None, **kwargs):
    """ 
    简单地根据区间范围均匀采样
    params:
        size: 采得最终的样本量
        A0: A 的初始值

        l_alpha: 左侧采样界/可以是和 A0 相同大小的 array/float/2个元素tuple
        r_alpha: 右侧采样界/可以是和 A0 相同大小的 array/float/2个元素tuple
        n_latin: 是否进行拉丁超立方采样; 0 为不进行拉丁超立方采样, 若为其他值则进行拉丁超立方采样且在每个维度的采样间隔点为 n_latin
        save_path: 默认为None, 保存采样数据路径
    """
    # 生成 size 个以 A0 为中心的随机数
    if reduced_mech_A0 is None or max_uncertainty_range is None:
        if n_latin == 0:
            if isinstance(l_alpha, float) and isinstance(r_alpha, float):
                samples = ((r_alpha - l_alpha) * np.random.random_sample(size = (size, len(A0))) + l_alpha) + A0
            elif isinstance(l_alpha, (np.ndarray, list)) and isinstance(r_alpha, (np.ndarray, list)):
                # 若 l_alpha 和 r_alpha 是 array, 且维度和 A0 相同, 则直接采样
                assert len(l_alpha) == len(A0) and len(r_alpha) == len(A0), f"l_alpha and r_alpha should have the same shape as A0, but got {len(l_alpha)} and {len(r_alpha)} and A0: {len(A0)}"
                samples = ((r_alpha - l_alpha) * np.random.random_sample(size = (size, len(A0))) + l_alpha) + A0
            elif isinstance(l_alpha, tuple) and isinstance(r_alpha, tuple):
                # 若 l_alpha 和 r_alpha 是 tuple, 且只有两个元素，则在 (A0 - l_alpha[0], A0 + l_alpha[1]), (A0 - r_alpha[0], A0 + r_alpha[1]) 之间采样
                if len(l_alpha) == 2 and len(r_alpha) == 2:
                    ll_alpha, lr_alpha = l_alpha; rl_alpha, rr_alpha = r_alpha
                    if not isinstance(ll_alpha, Iterable):
                        r_sample_size = int(size * (lr_alpha - ll_alpha) / (rr_alpha - rl_alpha + lr_alpha - ll_alpha))
                        samples1 = ((rr_alpha - rl_alpha) * np.random.random_sample(size = (r_sample_size, len(A0))) + rl_alpha) + A0
                        samples2 = ((lr_alpha - ll_alpha) * np.random.random_sample(size = (size - r_sample_size, len(A0))) + ll_alpha) + A0
                        samples = np.concatenate((samples1, samples2), axis = 0)
                    else:
                        try:
                            r_sample_size = int(size * (np.mean(lr_alpha) - np.mean(ll_alpha)) / 
                                                (np.mean(rr_alpha) - np.mean(rl_alpha) + np.mean(lr_alpha) - np.mean(ll_alpha)))
                        except:
                            r_sample_size = int(0.5 * size)
                        samples1 = ((rr_alpha - rl_alpha) * np.random.random_sample(size = (r_sample_size, len(A0))) + rl_alpha) + A0
                        samples2 = ((lr_alpha - ll_alpha) * np.random.random_sample(size = (size - r_sample_size, len(A0))) + ll_alpha) + A0
                        samples = np.concatenate((samples1, samples2), axis = 0)

                else:
                    raise ValueError("l_alpha and r_alpha should have the same shape as 2")
        else:
            n_latin = size 
            # 使用 latin 超立方采样方法, 在每个维度划分均匀网格后采样
            if isinstance(l_alpha, Number) and isinstance(r_alpha, Number):
                l_alpha = np.array([l_alpha] * len(A0)); r_alpha = np.array([r_alpha] * len(A0))
            samples = np.empty(shape = (size, len(A0)))
            for ind in range(len(A0)):
                grid_step_size = (r_alpha[ind] - l_alpha[ind]) / (n_latin + 1)
                samples[:,ind] = np.linspace(l_alpha[ind], r_alpha[ind], n_latin + 1)[:-1] + A0[ind] + np.random.random_sample(size = n_latin) * grid_step_size
         
    else:
        assert len(max_uncertainty_range) == len(A0), f"max_uncertainty_range should have the same shape as A0, but got {len(max_uncertainty_range)} and A0: {len(A0)}"
        total_samples = []
        while len(total_samples) < size:
            samples = ((r_alpha - l_alpha) * np.random.random_sample(size = (2 * size, len(A0))) + l_alpha) + A0
            # 判断 samples 是否在 [reduced_mech_A0 - max_uncertainty_range, reduced_mech_A0 + max_uncertainty_range] 范围内，并筛选掉超出的样本
            valid_samples = np.all((samples >= (reduced_mech_A0 - max_uncertainty_range)) & (samples <= (reduced_mech_A0 + max_uncertainty_range)), axis = 1)
            samples = samples[valid_samples, :]
            total_samples.extend(samples.tolist())
        samples = np.array(total_samples[:size])
    if save_path is not None:
        np.save(save_path, samples)
    return samples

@torch.no_grad()
def SampleAWithNet(Net: Callable, 
                  true_data: np.ndarray, 
                  threshold: Union[float, np.ndarray], 
                  father_samples: np.ndarray = None,
                  loss_func: Callable = None,
                  reduced_data: np.ndarray = None,
                  reduced_threshold: float = None,
                  passing_rate_upper_limit: float = 1,
                  ord = 2,
                  debug = False,
                  reverse_DataNormalization:Callable = None,
                  reduced_mech_A0: np.ndarray = None,
                  max_uncertainty_range: np.ndarray = None,
                  **kwargs):
    """ 
    根据已经训练部分的网络反过来指导采样流程，采样流程和之前完全一样，但是增加了网络预测筛选过程
    Net: 已经训练好的网络; 注意网络的位置应该在 CPU 上
    true_data: Net 要与之比较的真实值; IDT 请先做 log 对数
    threshold: int or ndarray, critorion 允许的误差最大值; 如果输入为 ndarray 则必须与 predict 维度相同
    father_sample: 是否从 father_sample 中筛选
    loss_func: 用于计算误差的函数; 默认为 None, 则使用 Net 作为误差函数
    scalar_input，mean_input: 在 Net 的输出为标准化后的值时, 需要输入 scalar_input 和 mean_input 用于反标准化
    passing_rate_upper_limit: 采样通过率的上限, 用于控制采样的样本量
    kwargs: 在father_sample 为 None 时, father_sample会自动调用函数 sample_constant_A, kwargs 为相同的 Input
        size: 采得最终的样本量
        A0: A 的初始值
        l_alpha: 左侧采样界
        r_alpha: 右侧采样界
        debug: 将返回每次采样的误差值, 用于检查bug位置

    20230309: 增加 reduced_data 和 reduced_threshold 的输入，功能是要求预测值不仅满足真实值的要求，且预测值最好满足：
        | pred - true | < reduced_threshold * | reduced - true |
    如果输入均为 None 则不加入此项筛选
    20230604：增加 scalar_input 和 mean_input 用于反标准化
    20230730: 增加 pass_rate
    20230817: 增加 loss_func 筛选
    """
    if father_samples is None:
        pass_nums = int(kwargs.get('size') * passing_rate_upper_limit)
        samples = sample_constant_A(reduced_mech_A0 = reduced_mech_A0, max_uncertainty_range = max_uncertainty_range, **kwargs) 
    else:
        if len(father_samples) == 0:
            return np.array([]) if not debug else (np.array([]), np.array([]))
        samples = father_samples
        pass_nums = int(len(father_samples) * passing_rate_upper_limit)

    pred_data = Net(torch.tensor(samples, dtype = torch.float)).detach().numpy()
    if not reverse_DataNormalization is None:
        pred_data = pred_data.reshape(pred_data.shape[0], -1)  # 确保 pred_data 是二维的
        pred_data = reverse_DataNormalization(pred_data)
        
    if loss_func is None:
        Rlos = np.abs(pred_data - true_data); 
    else:
        Rlos = loss_func(pred_data, true_data)
    sorted_Rlos_index = np.argsort(np.linalg.norm(Rlos, axis = 1, ord = ord))
    samples = samples[sorted_Rlos_index, :]; pred_data = pred_data[sorted_Rlos_index, :]; Rlos = Rlos[sorted_Rlos_index, :]
    filtered_data = samples[np.all(Rlos < threshold, axis = 1), :]

    # 简化机理 data 筛选
    if not reduced_data is None and not reduced_threshold is None:
        reduced_error = reduced_threshold * np.abs(reduced_data - true_data)
        filtered_data = filtered_data[np.all(Rlos < reduced_error, axis = 1), :]

    # PASSING rate work here
    filtered_data = filtered_data[:min(pass_nums, len(filtered_data)), :]

    # # 如果全部通过，强制取前 50%
    # if father_samples is None and len(filtered_data) == kwargs.get('size', None):
    #     filtered_data = filtered_data[:int(kwargs.get('size') / 2), :]

    if debug:
        return filtered_data, pred_data
    return filtered_data


    
"""===================================================================================================="""
"""                                 CANTERA  YAML  SEETINGS                                            """
"""===================================================================================================="""

def concatenate_eq_dict(eq_dict:dict, need_flatten = False):
    """
    将 eq_dict 中的所有 value 拼接起来
    """
    Alist = np.array([])
    if need_flatten:
        copy_eq_dict = copy.deepcopy(eq_dict)
        for eq in eq_dict:
            try:
                Alist = np.r_[Alist, np.ravel(eq_dict[eq])]
                copy_eq_dict[eq] = np.ravel(eq_dict[eq])
            except:
                Alist = np.r_[Alist, np.concatenate(eq_dict[eq])]
                copy_eq_dict[eq] = np.concatenate(eq_dict[eq])
        return Alist, copy_eq_dict
    else:    
        for eq in eq_dict:
            try:
                Alist = np.r_[Alist, np.ravel(eq_dict[eq])]
            except:
                try:
                    tmp_a = np.concatenate(eq_dict[eq])
                    Alist = np.r_[Alist, tmp_a]
                except:
                    print(eq)
                    print(eq_dict[eq])
                    print(Alist.shape)
                    print(np.array(eq_dict[eq]).shape)
                    exit()
        return Alist


def Alist2eq_dict(Alist: list, ori_eq_dict:dict):
    """
    将 Alist 转化为 eq_dict 的工具函数
    """
    # 使用 Alist 替换 eq_dict 中的值
    # 没有任何 duplicate 和 Troe 问题的情况
    eq_dict = ori_eq_dict.copy()
    if len(eq_dict) == len(Alist):
        eq_dict = dict(zip(eq_dict.keys(), Alist)) 
    else:
        count_num = 0
        for key in eq_dict.keys():
            current_eq = eq_dict[f'{key}']
            # 正常情况
            if not isinstance(current_eq, Iterable):
                eq_dict[f'{key}'] = Alist[count_num]
                count_num += 1
                continue
            # Falloff
            elif isinstance(current_eq[0], Iterable) and len(current_eq) == 1 and len(current_eq[0]) == 2:
                eq_dict[f'{key}'] = [[Alist[count_num], Alist[count_num + 1]]]
                count_num += 2          
            # duplicate &  Plog
            elif not isinstance(current_eq[0], Iterable) or len(current_eq) >= 2:
                if not isinstance(current_eq[0], Iterable): # Normal duplicate & NO duplicate Plog
                    eq_dict[f'{key}'] = list(Alist[count_num: count_num + len(eq_dict[f'{key}'])])
                    count_num += len(eq_dict[f'{key}'])
                else: # duplicate Plog
                    eq_dict[f'{key}'] = []
                    for tmp_current_eq in current_eq:
                        eq_dict[f'{key}'].append(Alist[count_num: count_num + len(tmp_current_eq)])
                        count_num += len(tmp_current_eq)
          
            else:
                raise ValueError("Fatal: No suitable match! Please checkout the thermo file .yaml")
    
    return eq_dict
            

def eq_dict2Alist(eq_dict:dict, benchmark_eq_dict:dict = None):
    """
    将 eq_dict 转化为 Alist 的工具函数；若 benchmark_eq_dict 不为 None, 
    则我们会将 eq_dict 通过复制 value 的方式扩展为 benchmark_eq_dict 后再转化为 Alist
    params:
        eq_dict: 需要转化的 eq_dict
        benchmark_eq_dict: 用于扩展 eq_dict 的 benchmark_eq_dict
    return:
        Alist: 转化后的 Alist
    """
    Alist = np.array([])       
    if not benchmark_eq_dict is None:
        _, benchmark_eq_dict = concatenate_eq_dict(benchmark_eq_dict, need_flatten=True)
        if eq_dict.keys() != benchmark_eq_dict.keys():
            # 剔除 eq_dict 中不属于 benchmark_eq_dict 的 key
            eq_dict = {key: eq_dict[key] for key in benchmark_eq_dict.keys() if key in eq_dict.keys()}
        for key in benchmark_eq_dict.keys():
            # print("len of eq_dict and benchmark:" , eq_dict[key], benchmark_eq_dict[key])
            if isinstance(eq_dict[key], Iterable):
                Alist = np.r_[Alist, np.ravel(eq_dict[key])]
            else:
                Alist = np.r_[Alist, eq_dict[key] * np.ones_like(benchmark_eq_dict[key]).flatten()]
    else:
        Alist = concatenate_eq_dict(eq_dict)
    return Alist


def eq_dict_broadcast2Alist(eq_dict:dict, benchmark_eq_dict:dict):
    """
    将 eq_dict 先 broadcast 到 benchmark_eq_dict 再转化为 Alist 的工具函数；broadcast 的方式是复制
    params:
        eq_dict: 需要转化的 eq_dict
        benchmark_eq_dict: 用于扩展 eq_dict 的 benchmark_eq_dict
    return:
        Alist: 转化后的 Alist
    """
    Alist = []; _, tmp_benchmark_eq_dict = concatenate_eq_dict(benchmark_eq_dict, need_flatten=True)
    for eq in tmp_benchmark_eq_dict:
        if isinstance(tmp_benchmark_eq_dict[eq], Iterable):
            length = len(np.array(tmp_benchmark_eq_dict[eq]))
            Alist.extend([eq_dict[eq]] * length)
        else:
            Alist.append(eq_dict[eq])
    return np.array(Alist)


# Legacy; 不再维护
def A2yaml(original_chem_path: str, chem_path: str, Alist: np.ndarray)  -> None:
    """根据Alist得到YAML"""   
    raise DeprecationWarning("This function is deprecated, please use Adict2yaml instead")
    # 加载原文件
    file = open(original_chem_path, 'r', encoding="utf-8")         # 获取yaml文件数据
    chem_file = yaml.load(file.read(), Loader=yaml.FullLoader)       # 将yaml数据转化为字典
    file.close()

    Aindex = 0
    for _, tmp_dict in enumerate(chem_file['reactions']):
    # 修改第ind号反应的反应常数，有两种类型的反应常数
        if 'rate-constant' in list(tmp_dict.keys()):
            tmp_dict['rate-constant']['A']  = float(Alist[Aindex])
            Aindex += 1
        elif 'high-P-rate-constant' in list(tmp_dict.keys()):
            tmp_dict['high-P-rate-constant']['A']  = float(Alist[Aindex])
            tmp_dict['low-P-rate-constant']['A']  = float(Alist[Aindex + 1])
            Aindex += 2
        else:
            print(f"KeyNotFoundError at {tmp_dict['equation']} when changing")
    # 写入 yaml 文件
    with open(chem_path, "w") as yaml_file:
        yaml.dump(chem_file, yaml_file, )


def ctSolution2A(yaml_file:str, **kwargs) -> Tuple[np.ndarray, dict]:
    """
    return:
        Alist, eq_dict
    将 yaml 加载为第一个反应-A值键值对字典和一个特定排序的 A 值列表
    字典对满足如下顺序
    {
        "equation": [[55, 11]] # 若这个反应是三体 falloff 反应; high 在前 low 在后
        "equation": [55, 11]   # 若这个反应是复试反应(duplicated reactions) 或 PlogReaction
        "equation": 55   # 若这个反应是正常反应
    }, 注意所有字典内的值都是经过 log10 操作的！暂时不支持其他变换格式
    列表则是字典直接全部降维并 extend 得到的, 因此也是 log 尺度
    """
    raise DeprecationWarning("This function is deprecated, please use yaml directly convert to Alist and eq_dict")
    reactions = ct.Solution(yaml_file).reactions(); eq_dict = {}
    plog_reaction_upper_pressure = kwargs.get('plog_reaction_upper_pressure', 5.6625e6)
    plog_reaction_lower_pressure = kwargs.get('plog_reaction_lower_pressure', 1e5)
    for reac in reactions:
        match type(reac):
            case ct._cantera.Reaction | ct._cantera.ThreeBodyReaction:
                if isinstance(reac.rate, ct.PlogRate):
                    match reac.duplicate:
                        case True:
                            # 第2次见这个重复反应
                            if reac.equation in eq_dict.keys():
                                tmp_A = eq_dict[reac.equation]; tmp_B = []
                                for Aslice in reac.rate.rates: 
                                    if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                        tmp_B.append(np.log10(Aslice[1].pre_exponential_factor)) 
                                eq_dict.update({reac.equation: [tmp_A, tmp_B]})                                         
                            # 第1次见这个重复反应
                            else:
                                tmp_A = []
                                # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                for Aslice in reac.rate.rates: 
                                    if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                        tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                eq_dict.update({reac.equation: tmp_A})
                        case False:
                            tmp_A = []
                            # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                            for Aslice in reac.rate.rates: 
                                tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                            eq_dict.update({reac.equation: tmp_A})                            
                else:
                    tmp_A = np.log10(reac.rate.pre_exponential_factor)
                    # match 重复反应
                    match reac.duplicate:
                        case True:
                            # 第2次见这个重复反应
                            if reac.equation in eq_dict.keys():
                                eq_dict[reac.equation].append(tmp_A)
                            # 第1次见这个重复反应
                            else:
                                eq_dict.update({reac.equation: [tmp_A]})
                        case False:
                            eq_dict.update({reac.equation: tmp_A})
            # match Falloff反应
            case ct._cantera.FalloffReaction | ct.ChemicallyActivatedReaction:
                tmp_A_high =  np.log10(reac.rate.high_rate.pre_exponential_factor)
                tmp_A_low =  np.log10(reac.rate.low_rate.pre_exponential_factor)
                # Falloff默认不会 duplicate
                match reac.duplicate:
                    case True:
                        raise ValueError("Duplicate in the three body Troe equation!")
                    case False:
                        eq_dict.update({reac.equation: [[tmp_A_high, tmp_A_low]]})
            # Plog Reaction 特殊识别 / 由于已经被 cantera 所 depreacated, 因此不再维护 
            case ct._cantera.PlogReaction: 
                tmp_A = []
                for Aslice in reac.rates:
                    if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                        tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                eq_dict.update({reac.equation: tmp_A})
            case default:
                raise ValueError(f"The Input of reactions is {default} neither Reaction nor FalloffReaction!")
        
    Alist = eq_dict2Alist(eq_dict)
    return Alist, eq_dict


def yaml_key2A(yaml_file: str, rea_keywords: list[str] = None, pro_keywords: list[str] = None, **kwargs) -> Tuple[np.ndarray, dict]:
    r"""
    读取机理的指前因子A; 只读取反应物包含 rea_keyword 和产物包含 pro_keyword 的方程式的 A 值
    目前仅仅支持 log 格式的 scalar

    如果 rea / pro 均是 None, 那么含义是将不对 反应物 / 生成物 进行任何限制

    目前支持反应: Reaction; PlogReaction; ThreeBodyReaction; FalloffReaction; Duplicate reactions
    params:
        chem_path: YAML 文件地址
        rea_keywords: 反应物的关键词
        pro_keywords: 生成物的关键词
        kwargs: 
            plog_reaction_upper_pressure: Plog 反应的读取上限压强
            plog_reaction_lower_pressure: Plog 反应的读取下限压强
    return:
        Alist, eq_dict
        Alist: 返回对应的指前因子 A
        eq_dict: dict, 用于之后的写入指前因子步骤
        格式：{
            "equation": [[55, 11]] # 若这个反应是三体 falloff 反应; high 在前 low 在后
            "equation": [55, 11]   # 若这个反应是复试反应(duplicated reactions) 或 PlogReaction
            "equation": 55   # 若这个反应是正常反应
            }
    """
    gas:ct.Solution = ct.Solution(yaml_file)
    reactions:ct.Reaction = gas.reactions(); eq_dict = {}
    # 若其中有 None 则为全集
    if pro_keywords is None: pro_keywords = [k.name for k in gas.species()]
    if rea_keywords is None: rea_keywords = [k.name for k in gas.species()]
    plog_reaction_upper_pressure = kwargs.get('plog_reaction_upper_pressure', 5.6625e6)
    plog_reaction_lower_pressure = kwargs.get('plog_reaction_lower_pressure', 1e5)
    for reac in reactions:
        reactor = list(reac.reactants.keys())
        producer = list(reac.products.keys())
        if any([keyword in pro_keywords for keyword in producer]) and \
            any([keyword in rea_keywords for keyword in reactor]):
            match type(reac):
                case ct._cantera.Reaction | ct._cantera.ThreeBodyReaction:
                    if isinstance(reac.rate, ct.PlogRate):
                        match reac.duplicate:
                            case True:
                                # 第2次见这个重复反应
                                if reac.equation in eq_dict.keys():
                                    tmp_A = eq_dict[reac.equation]; tmp_B = []
                                    for Aslice in reac.rate.rates: 
                                        if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                            tmp_B.append(np.log10(Aslice[1].pre_exponential_factor)) 
                                    eq_dict.update({reac.equation: [tmp_A, tmp_B]})                                         
                                # 第1次见这个重复反应
                                else:
                                    tmp_A = []
                                    # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                    for Aslice in reac.rate.rates: 
                                        # 我们只取 PlogReaction 中压强范围在 lower_pressure 和 upper_pressure 之间的反应
                                        if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                            tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                    eq_dict.update({reac.equation: tmp_A})
                            case False:
                                tmp_A = []
                                # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                for Aslice in reac.rate.rates: 
                                    if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                        tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                eq_dict.update({reac.equation: tmp_A})
                    else:
                        tmp_A = np.log10(reac.rate.pre_exponential_factor)
                        # match 重复反应
                        match reac.duplicate:
                            case True:
                                # 第2次见这个重复反应
                                if reac.equation in eq_dict.keys():
                                    eq_dict[reac.equation].append(tmp_A)
                                # 第1次见这个重复反应
                                else:
                                    eq_dict.update({reac.equation: [tmp_A]})
                            case False:
                                eq_dict.update({reac.equation: tmp_A})
                # match Falloff反应
                case ct._cantera.FalloffReaction | ct.ChemicallyActivatedReaction:
                    tmp_A_high =  np.log10(reac.rate.high_rate.pre_exponential_factor)
                    tmp_A_low =  np.log10(reac.rate.low_rate.pre_exponential_factor)
                    # Falloff默认不会 duplicate
                    match reac.duplicate:
                        case True:
                            # raise ValueError("Duplicate in the Falloff equation!")
                            # 第2次见这个重复反应
                            if reac.equation in eq_dict.keys():
                                eq_dict[reac.equation].append([tmp_A_high, tmp_A_low])
                            # 第1次见这个重复反应
                            else:
                                eq_dict.update({reac.equation: [[tmp_A_high, tmp_A_low]]})
                        case False:
                            eq_dict.update({reac.equation: [[tmp_A_high, tmp_A_low]]})
                # Plog Reaction 特殊识别 / 由于已经被 cantera 所 depreacated, 因此不再维护 
                case ct._cantera.PlogReaction: 
                    if reac.duplicate:
                        raise ValueError("Duplicate in the Plog equation is NOT supported YET!")
                    tmp_A = []
                    for Aslice in reac.rates:
                        tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                    eq_dict.update({reac.equation: tmp_A})
                case default:
                    raise ValueError(f"The Input of reactions is {default} neither Reaction nor FalloffReaction!")
    Alist = eq_dict2Alist(eq_dict)
    return Alist, eq_dict


def yaml_keychain2A(yaml_file: str, *chains:list[str], **kwargs) -> Tuple[np.ndarray, dict]:
    """
    读取机理的指前因子A; 只读取一条或多条特定路径的 A 值, 
    目前仅仅支持 log 格式的 scalar

    目前支持反应: Reaction; PlogReaction; ThreeBodyReaction; FalloffReaction; Duplicate reactions

    params:
        chem_path: YAML 文件地址
        chains: 一个或多个 list，每个list代表一条chain，要求是在每个chain存在以相邻两个元素前者反应物后者生成物的方程式

    return:
        Alist, eq_dict
        Alist: 返回对应的指前因子 A
        eq_dict: dict, 用于之后的写入指前因子步骤
        格式：{
            "equation": [[55, 11]] # 若这个反应是三体 falloff 反应; high 在前 low 在后
            "equation": [55, 11]   # 若这个反应是复试反应(duplicated reactions)/没有 duplicate 的 PlogReactionRate
            "equation": [
                [55, 11],
                [66, 22]
            ]   # 若这个反应是 duplicate 的 PlogReactionRate
            "equation": 55   # 若这个反应是正常反应
            }
    """
    
    ct.suppress_thermo_warnings()
    # 加载原文件
    gas:ct.Solution = ct.Solution(yaml_file)
    reactions:ct.Reaction = gas.reactions(); eq_dict = {}
    plog_reaction_upper_pressure = kwargs.get('plog_reaction_upper_pressure', 5.6625e6)
    plog_reaction_lower_pressure = kwargs.get('plog_reaction_lower_pressure', 1e5)
    # 匹配多条 chains
    for chain in chains:
        # 匹配前后两个元素
        for chain_index in range(len(chain) - 1):
            path_status = False # 用于检测是否是通路
            # 遍历所有的反应
            for reac in reactions:
                reactor = list(reac.reactants.keys())
                producer = list(reac.products.keys())
                # 将 chain 的两端匹配 reactor, producer 
                if chain[chain_index] in reactor and chain[chain_index + 1] in producer:
                    path_status = True
                    match type(reac):
                        case ct._cantera.Reaction | ct._cantera.ThreeBodyReaction:
                            # Plog Rate 特殊识别
                            if isinstance(reac.rate, ct.PlogRate):
                                match reac.duplicate:
                                    case True:
                                        # 第2次见这个重复反应
                                        if reac.equation in eq_dict.keys():
                                            tmp_A = eq_dict[reac.equation]; tmp_B = []
                                            for Aslice in reac.rate.rates: 
                                                if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                                    tmp_B.append(np.log10(Aslice[1].pre_exponential_factor)) 
                                            eq_dict.update({reac.equation: [tmp_A, tmp_B]})                                         
                                        # 第1次见这个重复反应
                                        else:
                                            tmp_A = []
                                            # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                            for Aslice in reac.rate.rates: 
                                                if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                                    tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                            eq_dict.update({reac.equation: tmp_A})
                                    case False:
                                        tmp_A = []
                                        # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                        for Aslice in reac.rate.rates: 
                                            if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                                tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                        eq_dict.update({reac.equation: tmp_A})
                            else:
                                tmp_A = np.log10(reac.rate.pre_exponential_factor)
                                # match 重复反应
                                match reac.duplicate:
                                    case True:
                                        # 第2次见这个重复反应
                                        if reac.equation in eq_dict.keys():
                                            eq_dict[reac.equation].append(tmp_A)
                                        # 第1次见这个重复反应
                                        else:
                                            eq_dict.update({reac.equation: [tmp_A]})
                                    case False:
                                        eq_dict.update({reac.equation: tmp_A})
                        # match Falloff反应
                        case ct._cantera.FalloffReaction | ct.ChemicallyActivatedReaction:
                            tmp_A_high =  np.log10(reac.rate.high_rate.pre_exponential_factor)
                            tmp_A_low =  np.log10(reac.rate.low_rate.pre_exponential_factor)
                            # Falloff默认不会 duplicate
                            match reac.duplicate:
                                case True:
                                    # raise ValueError("Duplicate in the Falloff equation!")
                                    # 第2次见这个重复反应
                                    if reac.equation in eq_dict.keys():
                                        eq_dict[reac.equation].append([tmp_A_high, tmp_A_low])
                                    # 第1次见这个重复反应
                                    else:
                                        eq_dict.update({reac.equation: [[tmp_A_high, tmp_A_low]]})
                                case False:
                                    eq_dict.update({reac.equation: [[tmp_A_high, tmp_A_low]]})
                        # Plog Reaction 特殊识别 / 由于已经被 cantera 所 depreacated, 因此不再维护 
                        case ct._cantera.PlogReaction: 
                            match reac.duplicate:
                                case True:
                                    raise ValueError("Duplicate in the Plog equation is NOT supported YET!")
                                case False:
                                    tmp_A = []
                                    # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                    for Aslice in reac.rates: 
                                        tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                    eq_dict.update({reac.equation: tmp_A})
                        case default:
                            raise ValueError(f"The Input of reactions is {default} neither Reaction nor FalloffReaction!")
            if not path_status:
                raise KeyError("The chain is not a currect reaction Path!")
    
    Alist = eq_dict2Alist(eq_dict)   
    return Alist, eq_dict


def yaml_eq2A(yaml_file: str, *equations:list[str], **kwargs):
    """
    通过 equation 来找到合适的 A 值; 没有加入 PlogReaction

    目前支持反应: Reaction; ThreeBodyReaction; FalloffReaction; Duplicate reactions
    params:
        yaml_file: 输入的 yaml 文件
        equations: 需要获得 A 值的反应; 反应必须存在于 YAML 文件中并完全对应 ct.Reaction.equation; 
        kwargs
    return:
        Alist, eq_dict
    """
    def contains_specific_m(equation):
        return bool(re.findall(r"\+M\|\b M \b|\(\+(.*?)\)|\+ M(?!\w)", equation))

    if equations == []:
        return np.array([]), {}
    ct.suppress_thermo_warnings()
    # 加载原文件
    gas:ct.Solution = ct.Solution(yaml_file)
    reaction_list:np.ndarray = np.array([reac.equation for reac in gas.reactions()]); eq_dict = {}
    equations:list = list(equations)
    plog_reaction_upper_pressure = kwargs.get('plog_reaction_upper_pressure', 5.6625e6)
    plog_reaction_lower_pressure = kwargs.get('plog_reaction_lower_pressure', 1e5)
    # 匹配多条 chains
    for equation in equations:
        # 因为 cantera 里面的独特转换格式会使得反应物物质对调之类，
        # 因此需要将 equation 先输入 cantera 中调整到 cantera 认可的形式
        # 这里的反应常数只是占位用的
        
        # 判断反应类型
        # if " M " in equation or ' (+M) ' in equation:
        if contains_specific_m(equation):
            try:
                tmp_reaction = ct.FalloffReaction(
                    equation = equation,
                    rate = ct.LindemannRate(
                        low = ct.Arrhenius(1, 1, 1),
                        high = ct.Arrhenius(1, 1, 1),
                    )
                )
            except:
                tmp_reaction = ct.ThreeBodyReaction(equation = equation, rate = {
                    "A": 1,
                    "b": 1,
                    "Ea": 1,
                }, efficiencies = {"O2":1}, kinetics = gas)
        elif equation not in reaction_list:
            tmp_reaction = ct.Reaction(equation = equation, rate = ct.ArrheniusRate(1, 1, 1))
            equation = tmp_reaction.equation

        # 找到所有相同的反应方程索引
        equation_indies = np.where(reaction_list == equation)[0]
        assert equation_indies.tolist() != [], f"The equation {equation} is not in the GAS file!" 
        for equation_index in equation_indies:
            reac = gas.reaction(equation_index)
            # 将 chain 的两端匹配 reactor, producer 
            match type(reac):
                case ct._cantera.Reaction | ct._cantera.ThreeBodyReaction:
                    if isinstance(reac.rate, ct.PlogRate):
                        match reac.duplicate:
                            case True:
                                # 第2次见这个重复反应
                                if reac.equation in eq_dict.keys():
                                    tmp_A = eq_dict[reac.equation]; tmp_B = []
                                    for Aslice in reac.rate.rates:
                                        if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure: 
                                            tmp_B.append(np.log10(Aslice[1].pre_exponential_factor)) 
                                    eq_dict.update({reac.equation: [tmp_A, tmp_B]})                                         
                                # 第1次见这个重复反应
                                else:
                                    tmp_A = []
                                    # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                    for Aslice in reac.rate.rates: 
                                        if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                            tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                    eq_dict.update({reac.equation: tmp_A})
                            case False:
                                tmp_A = []
                                # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                                for Aslice in reac.rate.rates: 
                                    if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                        tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                                eq_dict.update({reac.equation: tmp_A})
                    else:
                        tmp_A = np.log10(reac.rate.pre_exponential_factor)
                        # match 重复反应
                        match reac.duplicate:
                            case True:
                                # 第2次见这个重复反应
                                if reac.equation in eq_dict.keys():
                                    eq_dict[reac.equation].append(tmp_A)
                                # 第1次见这个重复反应
                                else:
                                    eq_dict.update({reac.equation: [tmp_A]})
                            case False:
                                eq_dict.update({reac.equation: tmp_A})
                # match Falloff反应
                case ct._cantera.FalloffReaction | ct.ChemicallyActivatedReaction:
                    tmp_A_high =  np.log10(reac.rate.high_rate.pre_exponential_factor)
                    tmp_A_low =  np.log10(reac.rate.low_rate.pre_exponential_factor)
                    # Falloff默认不会 duplicate
                    match reac.duplicate:
                        case True:
                            # raise ValueError("Duplicate in the Falloff equation!")
                            # 第2次见这个重复反应
                            if reac.equation in eq_dict.keys():
                                eq_dict[reac.equation].append([tmp_A_high, tmp_A_low])
                            # 第1次见这个重复反应
                            else:
                                eq_dict.update({reac.equation: [[tmp_A_high, tmp_A_low]]})
                        case False:
                            eq_dict.update({reac.equation: [[tmp_A_high, tmp_A_low]]})
                case ct._cantera.PlogReaction:
                    # match 重复反应
                    match reac.duplicate:
                        case True:
                            raise ValueError("Duplicate in the Plog equation is NOT supported YET!")
                        case False:
                            tmp_A = []
                            # 遍历 PlogReaction 中的所有tuple 类键值对 (压强范围, (A,b, Ea))
                            for Aslice in reac.rate.rates: 
                                if Aslice[0] >= plog_reaction_lower_pressure and Aslice[0] <= plog_reaction_upper_pressure:
                                    tmp_A.append(np.log10(Aslice[1].pre_exponential_factor))
                            eq_dict.update({reac.equation: tmp_A})
                case default:
                    raise ValueError(f"The Input of reactions is {default} neither Reaction nor FalloffReaction!")
    # 根据函数输入 equations 的顺序，调整 eq_dict 的顺序
    # eq_dict = dict(sorted(eq_dict.items(), key=lambda item: equations.index(item[0])))
    Alist = eq_dict2Alist(eq_dict)   
    return Alist, eq_dict


def Adict2yaml(original_chem_path: str, chem_path: str, eq_dict: dict = None, Alist:list = None, 
               rea_keywords = None, pro_keywards = None, **kwargs)  -> None:
    """
    根据 ctSolution2A 函数得到的 dict 反推回 yaml 文件，需要提供 eq_dict 或者 Alist 中的一个

    注意：不需要在转换之前修改 eq_dict 中的 A 值，因为 log 还原已经集成在本函数中
    
    params:
        original_chem_path: 原始的 yaml 文件位置
        chem_path: 更新后 yaml 文件位置
        eq_dict: dict, 格式为 
        {
            "equation": [[55, 11]] # 若这个反应是三体 falloff 反应; high 在前 low 在后
            "equation": [55, 11]   # 若这个反应是复试反应(duplicated reactions)或者 Plog Funtion
            "equation": 55         # 若这个反应是正常反应
        }, 注意每一个数值应该是 log 格式下的数值！
        Alist: 默认为None, 若不为 None, 将使用 Alist 替换 eq_dict 中的值之后再保存
        rea_keywords: 反应关键词, 默认为 None, 只有在 eq_dict 为 None 时会使用在 yaml_key2A 函数中
        pro_keywards: 生成物关键词, 默认为 None, 只有在 eq_dict 为 None 时会使用在 yaml_key2A 函数中
    return:
        chem_path
        保存文件 chem_path
    """
    # 加载原文件
    origin_gas = ct.Solution(original_chem_path); new_reactions = []; 
    if eq_dict is None:
        _, eq_dict = yaml_key2A(original_chem_path, rea_keywords = rea_keywords, pro_keywards = pro_keywards)
    if Alist is not None:
        # 防止之后的 pop 操作影响到 eq_dict
        Alist = np.array(Alist)
        # 确保 eq_dict 生成的 alist 与输入的 Alist 一致
        assert eq_dict2Alist(eq_dict).shape == Alist.shape, ValueError(f"The shape of Alist and eq_dict is not consistent! Shape of Alist is {Alist.shape} but shape of eq_dict is {eq_dict2Alist(eq_dict).shape}!")
        
        # 使用 Alist 替换 eq_dict 中的值
        eq_dict = Alist2eq_dict(Alist, ori_eq_dict = copy.deepcopy(eq_dict))
    plog_reaction_upper_pressure = kwargs.get('plog_reaction_upper_pressure', 5.6625e6)
    plog_reaction_lower_pressure = kwargs.get('plog_reaction_lower_pressure', 1e5)
    for reac in origin_gas.reactions():
        # 将 original 里面的 equation 和输入的 eq_dict 匹配
        equation:str = reac.equation; 
        if equation in eq_dict.keys():
            match type(reac):
                # 基元反应
                case ct._cantera.Reaction:
                    # 新式 PlogRate
                    if isinstance(reac.rate, ct.PlogRate):
                        match reac.duplicate:
                            case False:
                                plograte = []; ind = 0
                                for tmp_rate in reac.rates:
                                    tmp_p, tmp_rate = tmp_rate
                                    if tmp_p >= plog_reaction_lower_pressure and tmp_p <= plog_reaction_upper_pressure:   
                                        plograte.append((tmp_p, ct.Arrhenius(
                                            10**eq_dict[equation][ind],
                                            tmp_rate.temperature_exponent,
                                            tmp_rate.activation_energy
                                        )))
                                        ind += 1
                                tmp_reaction = ct.Reaction(equation = equation, rate = ct.PlogRate(plograte))
                                new_reactions.append(tmp_reaction)
                            case True:
                                # 将第一个元素[即 duplicate 的第一个] pop 出来
                                twinA = eq_dict[equation].pop(0)
                                plograte = []; ind = 0
                                for tmp_rate in reac.rates:
                                    tmp_p, tmp_rate = tmp_rate
                                    if tmp_p >= plog_reaction_lower_pressure and tmp_p <= plog_reaction_upper_pressure:
                                        plograte.append((tmp_p, ct.Arrhenius(
                                            10**twinA[ind],
                                            tmp_rate.temperature_exponent,
                                            tmp_rate.activation_energy
                                        )))
                                        ind += 1
                                tmp_reaction = ct.Reaction(equation = equation, rate = ct.PlogRate(plograte))
                                tmp_reaction.duplicate = True
                                new_reactions.append(tmp_reaction)                                
                    else:
                        b = reac.rate.temperature_exponent; Ea = reac.rate.activation_energy
                        # match 重复反应
                        match reac.duplicate:
                            case True:
                                # 将第一个元素 pop 出来
                                twinA = eq_dict[equation].pop(0)
                                if isinstance(twinA, Iterable): twinA = twinA[0]
                                # 制造两个新反应且设置 duplicate flag
                                tmp_reaction = ct.Reaction(equation = equation, rate = {
                                    "A": 10**twinA,
                                    "b": b,
                                    "Ea": Ea,
                                })
                                tmp_reaction.duplicate = True
                                new_reactions.append(tmp_reaction)
                            case False:
                                tmp_reaction = ct.Reaction(equation = equation, rate = {
                                        "A": 10**eq_dict[equation],
                                        "b": b,
                                        "Ea": Ea,
                                    })
                                new_reactions.append(tmp_reaction)
                # 三体反应
                case ct._cantera.ThreeBodyReaction:
                    b = reac.rate.temperature_exponent; Ea = reac.rate.activation_energy
                    efficiencies = reac.efficiencies
                    # match 重复反应
                    match reac.duplicate:
                        case True:
                            # pop 出列表中第一个元素
                            tmp_A =  eq_dict[equation].pop(0)
                            # 制造两个新反应且设置 duplicate flag
                            tmp_reaction = ct.ThreeBodyReaction(equation = equation, rate = {
                                "A": 10**tmp_A,
                                "b": b,
                                "Ea": Ea,
                            }, efficiencies = efficiencies, kinetics = origin_gas)
                            tmp_reaction.duplicate = True
                            new_reactions.append(tmp_reaction)
                        case False:
                            tmp_reaction = ct.ThreeBodyReaction(equation = equation, rate = {
                                    "A": 10**eq_dict[equation],
                                    "b": b,
                                    "Ea": Ea,
                                }, efficiencies = efficiencies, kinetics = origin_gas)
                            new_reactions.append(tmp_reaction)  
                # Falloff反应
                case ct._cantera.FalloffReaction | ct.ChemicallyActivatedReaction:
                    # A_low = 10**eq_dict[equation][0][1]; A_high = 10**eq_dict[equation][0][0]
                    b_low = reac.rate.low_rate.temperature_exponent; b_high = reac.rate.high_rate.temperature_exponent
                    Ea_low = reac.rate.low_rate.activation_energy; Ea_high = reac.rate.high_rate.activation_energy
                    # Falloff默认不会 duplicate
                    match reac.duplicate:
                        case True:
                            twinA = eq_dict[equation].pop(0)
                            A_low = 10**twinA[1]; A_high = 10**twinA[0]
                        case False:
                            A_low = 10**eq_dict[equation][0][1]; A_high = 10**eq_dict[equation][0][0]
                    match type(reac.rate):
                        case ct.LindemannRate:
                            tmp_reaction = ct.FalloffReaction(
                                equation = equation,
                                efficiencies = reac.efficiencies,
                                rate = ct.LindemannRate(
                                    low = ct.Arrhenius(A_low, b_low, Ea_low),
                                    high = ct.Arrhenius(A_high, b_high, Ea_high),
                                )
                            )
                                                              
                        case ct.TroeRate:
                            tmp_reaction = ct.FalloffReaction(
                                equation = equation,
                                efficiencies = reac.efficiencies,
                                rate = ct.TroeRate(
                                    low = ct.Arrhenius(A_low, b_low, Ea_low),
                                    high = ct.Arrhenius(A_high, b_high, Ea_high),
                                    falloff_coeffs = reac.rate.falloff_coeffs
                                )
                            )
                            
                        case ct.SriRate:
                            tmp_reaction = ct.FalloffReaction(
                                equation = equation,
                                efficiencies = reac.efficiencies,
                                rate = ct.SriRate(
                                    low = ct.Arrhenius(A_low, b_low, Ea_low),
                                    high = ct.Arrhenius(A_high, b_high, Ea_high),
                                    falloff_coeffs = reac.rate.falloff_coeffs
                                )
                            )
                            
                        case ct.TsangRate:
                            tmp_reaction = ct.FalloffReaction(
                                equation = equation,
                                efficiencies = reac.efficiencies,
                                rate = ct.TsangRate(
                                    low = ct.Arrhenius(A_low, b_low, Ea_low),
                                    high = ct.Arrhenius(A_high, b_high, Ea_high),
                                    falloff_coeffs = reac.rate.falloff_coeffs
                                )
                            )
                    tmp_reaction.duplicate = True if reac.duplicate else False        
                    new_reactions.append(tmp_reaction)
                # Plog Reaction 特殊识别
                case ct._cantera.PlogReaction: 
                    # match 重复反应
                    match reac.duplicate:
                        case True:
                            raise ValueError("Duplicate in the Plog equation is not Supported!")
                        case False:
                            plograte = []
                            for ind, tmp_rate in enumerate(reac.rates):
                                tmp_p, tmp_rate = tmp_rate
                                plograte.append(tuple(tmp_p, ct.Arrhenius(
                                    10**eq_dict[equation][ind],
                                    tmp_rate.temperature_exponent,
                                    tmp_rate.activation_energy
                                )))
                            tmp_reaction = ct.PlogReaction(equation = equation, rate = plograte, kinetics = origin_gas)
                            new_reactions.append(tmp_reaction)                    
                # 其他反应暂时不支持
                case default:
                    raise ValueError(f"The Input of reactions is {default} neither Reaction nor FalloffReaction!")
        else:
            # 不对应则直接 append 进来
            new_reactions.append(reac)

    new_gas =  ct.Solution(thermo = 'ideal-gas', species = origin_gas.species(), reactions = new_reactions, kinetics = 'gas',)
    new_gas.write_yaml(filename = chem_path) # 写入 yaml 文件
    return chem_path


def reactants_division_by_C_num(chemfile, C_number = None):
    """
    将 chemfile 中涉及到的所有物质按照碳原子数进行分组
    params:
        chemfile: Cantera 的 yaml 文件
        C_number: 指定碳原子数，如果为 None 则默认为所有物质. 例如：[1, 2, 4]
    return:
        reactants_dicts: 按照碳原子数分组的物质字典构成的数组，例如在 C_number = [1, 2, 4] 时
        reactants_dicts = [{<C1 的所有物质}, {C1~C2 的所有物质}, {C2~C4 的所有物质}, {C4~ 的所有物质}]
    """
    # 读取 yaml 文件
    gas = ct.Solution(chemfile)
    # 获取所有物质
    species = gas.species()
    # 获取所有物质中的含碳量
    C_num = [spec.composition.get('C', 0) for spec in species]
    # 将对应 C_num 相同的物质放在一起
    reactants_dict = {}
    for ind, num in enumerate(C_num):
        if num in reactants_dict.keys():
            reactants_dict[int(num)].append(species[ind].name)
        else:
            reactants_dict[int(num)] = [species[ind].name]
    # 按照 C_number 合并 reactants_dict 中的物质
    reactants_dicts = []
    if C_number is None:
        return reactants_dict.values()
    else:
        C_max = max(reactants_dict.keys()); C_number.append(C_max + 1)
        # 将 np.arange(Cmax) 中的数字按照 C_number 切片
        for indnd, ind in enumerate(C_number):
            if ind == C_max + 1: break
            if indnd == 0:
                tmp_reactants_dict = []
                [tmp_reactants_dict.extend(reactants_dict[i]) for i in np.arange(ind)]
            else:
                tmp_reactants_dict = []
                [tmp_reactants_dict.extend(reactants_dict[i]) for i in np.arange(ind, C_number[indnd + 1])]
            # 删除 tmp_reaction_list 中的重复项
            tmp_reaction_list = list(set(tmp_reaction_list))    
            reactants_dicts.append(tmp_reactants_dict)
    # 去掉 C_number 中的 最后一个元素
    if reactants_dicts[-1] == []:
        C_number.pop()
        reactants_dicts.pop()
    return reactants_dicts


def reactions_division_by_C_num(chemfile, C_number = None, original_eq_list: dict = None):
    """
    将 chemfile 中涉及到的所有反应按照最高碳原子数的反应物含有的碳原子数进行分组
    params:
        chemfile: Cantera 的 yaml 文件
        C_number: 指定碳原子数，如果为 None 则默认为所有物质. 例如：[1, 2, 4]
        original_eq_list: 原始的反应方程式列表; 只保留存在 original_eq_list 中的反应; 
        如果为 None 则默认为所有反应 gas.reaction_equations()
    return:
        reactants_dicts: 按照碳原子数分组的物质字典构成的数组，例如在 C_number = [1, 2, 4] 时
        reactants_dicts = [{<C1 的所有物质}, {C1~C2 的所有物质}, {C2~C4 的所有物质}, {C4~ 的所有物质}]
    """

    # 读取 yaml 文件
    gas = ct.Solution(chemfile)
    # 获取所有物质
    species = gas.species()
    if original_eq_list is None: original_eq_list = gas.reaction_equations()
    # 获取所有物质中的含碳量
    C_num = [spec.composition.get('C', spec.composition.get('c', 0)) for spec in species]; C_max = max(C_num)
    C_number.append(C_max + 1)
    reaction_list = []
    for indnd, ind in enumerate(C_number):
        tmp_reaction_list = []
        for reaction in gas.reactions():
            if reaction.equation in original_eq_list:
                # 获取反应物
                reactants = list(reaction.reactants.keys())
                # 获取反应物中的最高碳原子数
                reactants_C_num = [gas.species(gas.species_index(reactant)).composition.get('C', 0) 
                                for reactant in reactants]
                reactants_C_max = int(max(reactants_C_num))
                # 将 reactants_C_max 与 C_number 中的数字进行比较
                if reactants_C_max <= ind:
                    if indnd == 0:
                        tmp_reaction_list.append(reaction.equation)
                    elif reactants_C_max > C_number[indnd - 1]:
                        tmp_reaction_list.append(reaction.equation)
                # 删除 tmp_reaction_list 中的重复项; 且保证 tmp_reaction_list 中顺序不变
                tmp_reaction_list = np.unique(np.array(tmp_reaction_list)).tolist()
        reaction_list.append(tmp_reaction_list)        
    # 若 reaction_list 中的最后一个元素为空，则去掉 C_number 中的 最后一个元素
    if reaction_list[-1] == []:
        C_number.pop()
        reaction_list.pop()
    return reaction_list


def convert_from_lowercase_to_upper(chem_file, output_file):
    """
    将 chem_file 中的所有物质名称转换为大写
    """
    # 使用 cantera 加载 chem_file
    gas = ct.Solution(chem_file)
    species = gas.species()
    species_names_mapping = {
        spec.name: spec.name.upper() for spec in species
    }
    lowercase_species_names = list(species_names_mapping.keys())
    # 直接用 txt 读取机理
    with open(chem_file, 'r') as f:
        lines = f.readlines()
    
    # 将所有的物质名称转换为大写
    for ind, line in enumerate(lines):
        if any([name in line for name in lowercase_species_names]):
            new_line = re.split(r'([ ,:\[\]\n]+)', line)
            new_line = [
                species_names_mapping[name] if name in lowercase_species_names else name for name in new_line
            ]
            lines[ind] = ''.join(new_line)
            
                
    # 将修改后的机理写入新的文件
    with open(output_file, 'w') as f:
        f.writelines(lines)
        
    # 测试新的机理文件
    ct.Solution(output_file)