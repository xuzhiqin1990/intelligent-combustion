# -*- coding:utf-8 -*-

import os, time, shutil, psutil, GPUtil, traceback, subprocess
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 在程序运行之前且导入 torch 之前先确定是否使用 GPU
try:
    device = GPUtil.getFirstAvailable(maxMemory=0.5, maxLoad=0.5)
    print("Avaliable GPU is ", device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
except:
    pass

import numpy as np, seaborn as sns, torch.nn as nn
import matplotlib.pyplot as plt
from typing import Iterable
from torch.utils.data import DataLoader, random_split
from torch import optim
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .basic_set import _DeePMO, gatherAPARTdata
from APART_base import _GenOneLFS
from APART_plot.APART_plot import compare_nn_train3, sample_distribution_IDT
from utils.cantera_utils import *
from utils.cantera_multiprocess_utils import yaml2LFS_sensitivity_Multiprocess, yaml2FS_Mcondition
from utils.setting_utils import *
from utils.yamlfiles_utils import * 
from APART_plot.Result_plot import CompareDRO_LFS_lineplot
from .DeePMO_V1_Network import Network_PlainSingleHead, DATASET_SingleHead


class DeePMO_LFS(_DeePMO):
    """
    使用 Network_PlainSingleHead 网络结构进行 DeePMO 训练: 考虑 LFS 的具体数值，同时考虑当前 LFS 的熄火极限
    """
    def __init__(self, circ = 0, basic_set = True, GenASampleRange = True, setup_file: str = './settings/setup.yaml', 
                 cond_file: str = "./data/true_data/true_lfs_data.npz", SetAdjustableReactions_mode:int = None, GenASampleRange_mode = None,
                 previous_best_chem:str = None, **kwargs) -> None:
        
        basic_set = kwargs.pop('basic_set', basic_set)
        super().__init__(circ, basic_set, setup_file, cond_file, 
                         SetAdjustableReactions_mode = SetAdjustableReactions_mode, **kwargs)
        GenASampleRange_mode = GenASampleRange_mode if GenASampleRange_mode is not None else self.APART_args.get('GenASampleRange_mode', 'default')
        
        if GenASampleRange:
            if previous_best_chem is None:
                if self.circ == 0: 
                    previous_best_chem = self.reduced_mech
                    previous_best_chem_LFS = self.reduced_lfs_data
                else:
                    # 加载上一个循环的 lfs_mean 和 lfs_std
                    previous_best_chem, previous_best_chem_LFS = self.SortALIST(
                        apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                        experiment_time = 1,
                        return_all = True,
                    )
                    previous_best_chem = np.squeeze(previous_best_chem)
                    previous_eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ - 1}.json")
                    previous_best_chem = Adict2yaml(self.reduced_mech, f"./data/APART_data/reduced_data/previous_best_chem_circ={self.circ}.yaml", previous_eq_dict, previous_best_chem)
            # previous_best_chem = f"./inverse_skip/circ={self.circ}/0/optim_chem.yaml" if previous_best_chem is None else previous_best_chem
            self.GenASampleRange(mode = GenASampleRange_mode, target_chem = previous_best_chem, **kwargs)
            self.GenASampleThreshold( 
                best_chem_LFS = previous_best_chem_LFS,
                **kwargs
            )
        else:
            # 在不调用 GenAsampleRange 的情况下, 需要根据之前生成的 eq_dict 更新 self.eq_dict 和 self.A0
            self.eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json")
            self.A0 = eq_dict2Alist(self.eq_dict)
        # 更新 Current APART args
        # 
        self.WriteCurrentAPART_args(
            GenASampleRange_mode = GenASampleRange_mode,
            GenASampleRange = GenASampleRange,
            # **currentAPART_args
        )


    def GenASampleRange(self, adjustable_reactions = None, l_alpha = None, r_alpha = None, mode = 'default', 
                        target_chem = None, **kwargs):    
        """
        生成采样的范围; 根据 mode 选择采样范围的生成依据. 需要在 init 中就启动
        params:
            adjustable_reactions: eq_dict, 可调节的反应的字典
            l_alpha: float, list[=len(self.A0)], 采样的左边界
            r_alpha: float, list[=len(self.A0)], 采样的右边界
            mode: str, 采样范围的生成依据
                'default': 默认的采样范围
                'C_numbers': 根据 reactants 中的含碳量来确定采样范围; 
                    l_alpha, r_alpha 为 list, 且长度与划分的 C_numbers 相同
                    target_chem: 默认为 self.reduced_mech, 在不同的 circ 需要自己指定; 采样的中心点
                    kwargs:
                        C_numbers: list, 每个反应物中的碳原子数的划分; 形如 [1, 2, 3, 4]
                                    表示 <C1, C1~C2, C2~C3, C3~C4, >C4 采样的范围
                    requirement: self.APART_args 需要以下 keys: 
                        C_numbers, C{num}_{l/r}_alpha
                    return:
                        重写 self.eq_dict, self.A0, 增加 self.alpha_dict, self.l_alpha, self.r_alpha
                'locally_sensitivity': 在 'C_numbers' 的基础上针对无碳、低碳根据敏感性的倒数来确定采样范围
                    target_chem: 默认为 self.reduced_mech, 在不同的 circ 需要自己指定; 采样的中心点
                    kwargs:
                        save_jsonpath: 保存灵敏度的 json 路径
                    requirement: self.APART_args 需要以下 keys: LFS_{first/midst/miden/last}_{l/r}_alpha
        """
        adjustable_reactions = adjustable_reactions if adjustable_reactions is not None else list(self.eq_dict.keys())
        
        C_numbers = self.APART_args.get('C_numbers', None)
        target_chem = target_chem if target_chem is not None else self.reduced_mech
        match mode:
            case 'default':
                A0 = eq_dict2Alist({
                    eq: self.eq_dict[eq] for eq in adjustable_reactions
                })
                tmp_l_alpha = self.APART_args['l_alpha']; tmp_r_alpha = self.APART_args['r_alpha']
                if isinstance(tmp_l_alpha, Iterable):
                    if len(tmp_l_alpha) != len(A0):
                        tmp_l_alpha = tmp_l_alpha[self.circ]
                        tmp_r_alpha = tmp_r_alpha[self.circ]
                l_alpha = l_alpha if l_alpha is not None else tmp_l_alpha
                r_alpha = r_alpha if r_alpha is not None else tmp_r_alpha
                self.l_alpha = l_alpha * np.ones_like(A0)
                self.r_alpha = r_alpha * np.ones_like(A0)
                self.alpha_dict = {key: (l_alpha, r_alpha) for key in self.eq_dict.keys()}
            case 'C_numbers':
                C_numbers = kwargs.get('C_numbers', None) if C_numbers is None else C_numbers
                assert C_numbers is not None, "C_numbers is None, while please use default mode"
                assert isinstance(C_numbers, Iterable), "C_numbers should be Iterable"
                self.l_alpha, self.r_alpha = [], []; self.alpha_dict = {}; self.eq_dict = {}
                # 根据 C_numbers 划分反应物组
                reactions_group = reactions_division_by_C_num(self.reduced_mech, C_numbers, adjustable_reactions)
                # 如果 max(C_numbers) 对应的 {C_numbers}_l_alpha 和 _r_alpha 不存在 self.APART_args 中，则使用默认值
                # C_max = max([spec.composition.get('C', 0) for spec in ct.Solution(self.reduced_mech).species()])
                # if f'{C_max}_l_alpha' not in self.APART_args:
                #     self.APART_args[f'{C_max}_l_alpha'] = self.APART_args[f'{max(C_numbers)}_l_alpha']
                #     self.APART_args[f'{C_max}_r_alpha'] = self.APART_args[f'{max(C_numbers)}_r_alpha']
                print(f"anet.reduced_mech: {self.reduced_mech}, C_numbers: {C_numbers}, reactions_group: {reactions_group}")
                for i, num in enumerate(sorted(C_numbers)):
                    num = int(num)
                    tmp_l_alpha = self.APART_args[f'C{num}_l_alpha']
                    tmp_r_alpha = self.APART_args[f'C{num}_r_alpha']
                    tmp_l_alpha = tmp_l_alpha[self.circ] if isinstance(tmp_l_alpha, Iterable) else tmp_l_alpha
                    tmp_r_alpha = tmp_r_alpha[self.circ] if isinstance(tmp_r_alpha, Iterable) else tmp_r_alpha
                    tmp_A0, tmp_eq_dict = yaml_eq2A(
                        target_chem, *reactions_group[i], 
                    )
                    self.eq_dict.update(tmp_eq_dict)
                    self.alpha_dict.update(
                        {key: (tmp_l_alpha, tmp_r_alpha) for key in tmp_eq_dict.keys()}
                    )
                    self.l_alpha.extend([tmp_l_alpha] * len(tmp_A0))
                    self.r_alpha.extend([tmp_r_alpha] * len(tmp_A0))
    
            case 'locally_LFS_sensitivity':  
                original_eq_dict = copy.deepcopy(self.eq_dict)
                self.l_alpha, self.r_alpha = [], []; self.alpha_dict = {}; self.eq_dict = {}
                save_jsonpath = kwargs.get('save_jsonpath', None)  
                LFSsensitivity_json_path = f"./data/APART_data/reduced_data/LFS_sensitivity_circ={self.circ}.json"  
                # 计算 original_eq_dict 中所有反应关于 LFS 的敏感度
                if not os.path.exists(LFSsensitivity_json_path):
                    LFS_sensitivity = yaml2LFS_sensitivity_Multiprocess(
                                target_chem,
                                LFS_condition = self.LFS_condition,
                                fuel = self.fuel, oxidizer = self.oxidizer,
                                save_path = save_jsonpath,
                                specific_reactions = list(original_eq_dict.keys()),
                            )      
                    print(LFS_sensitivity)    
                    # LFS_sensitivity 内所有 value 取绝对值后求平均值，替换原来的位置
                    LFS_sensitivity = {k: np.mean(np.abs(v)) for k, v in LFS_sensitivity.items()}
                    # 所有的 value 标准化: value - min(value) / (max(value) - min(value))
                    LFS_sensitivity = {k: (v - min(LFS_sensitivity.values())) / (max(LFS_sensitivity.values()) - min(LFS_sensitivity.values())) for k, v in LFS_sensitivity.items()}                      
                    # LFS_sensitivity 按照 value 重排序并保存
                    LFS_sensitivity = {k: v for k, v in sorted(LFS_sensitivity.items(),
                                                                key=lambda item: item[1], reverse=True)}
                    write_json_data(LFSsensitivity_json_path, LFS_sensitivity)
                else:
                    LFS_sensitivity = read_json_data(LFSsensitivity_json_path)
                # 根据 LFS_sensitivity 中的 value 对 original_eq_dict 进行分组: value < 1e-4, 1e-4 < value < 1e-2, 1e-2 < value < 1e-1, 1e-1 < value < 1
                reactions_group = [
                        [k for k, v in LFS_sensitivity.items() if 1e-1 < v < 1 + 1e-3],
                        [k for k, v in LFS_sensitivity.items() if 1e-2 < v < 1e-1],
                        [k for k, v in LFS_sensitivity.items() if 1e-4 < v < 1e-2],
                        [k for k, v in LFS_sensitivity.items() if v < 1e-4],
                    ]
                # 读取 APART_args 中三组 l_alpha 与 r_alpha 数值: 
                # LFSse_last_l_alpha, LFSse_last_r_alpha 用于统一赋值给第一组 reactions_group[-1]
                # LFSse_miden_l_alpha, LFSse_miden_r_alpha 用于给第二组 reactions_group[-2] 作为最大调整范围
                # LFSse_midst_l_alpha, LFSse_midst_r_alpha 用于给第2组 reactions_group[-2] 作为最小调整范围
                # LFSse_first_l_alpha, LFSse_first_r_alpha 用于统一赋值给第3组 reactions_group[-3]
                # 第四组 reactions_group[-4] 不做调整 (采样区间为 0)
                for i, sind in enumerate(['overse', 'first', ['midst', 'miden'], 'last']):
                    tmp_A0, tmp_eq_dict = yaml_eq2A(target_chem, *reactions_group[i], )
                    # 检测 tmp_eq_dict 是否为空
                    if len(reactions_group[i]) == 0:
                        self.GenAPARTDataLogger.info(f"reactions_group[{i}] is empty, break from the GenASampleRange Process!")
                        continue
                    if i != 2 and i != 0:
                        tmp_l_alpha = self.APART_args[f'LFSse_{sind}_l_alpha']
                        tmp_r_alpha = self.APART_args[f'LFSse_{sind}_r_alpha']
                        tmp_l_alpha = tmp_l_alpha[self.circ] if isinstance(tmp_l_alpha, Iterable) else tmp_l_alpha
                        tmp_r_alpha = tmp_r_alpha[self.circ] if isinstance(tmp_r_alpha, Iterable) else tmp_r_alpha
                        # if tmp_l_alpha == 0 and tmp_r_alpha == 0:
                        #     break # 如果 l_alpha 和 r_alpha 都为 0，则不做调整，因此不写如 eq_dict 中
                        self.alpha_dict.update(
                            {key: (tmp_l_alpha, tmp_r_alpha) for key in tmp_eq_dict.keys()}
                        )       
                        self.l_alpha.extend([tmp_l_alpha] * len(tmp_A0))
                        self.r_alpha.extend([tmp_r_alpha] * len(tmp_A0))                 

                    elif i == 2:
                        ## 计算 reactions_group[2] 中的最大敏感度和最小敏感度 的 log 值
                        max_LFS_mid = np.log10(np.max([LFS_sensitivity[k] for k in reactions_group[2]]))
                        min_LFS_mid = np.log10(np.min([LFS_sensitivity[k] for k in reactions_group[2]]))
                        if max_LFS_mid == min_LFS_mid:
                            max_LFS_mid += 1e-5
                        tmp_l_alpha1 = self.APART_args[f'LFSse_midst_l_alpha']
                        tmp_r_alpha1 = self.APART_args[f'LFSse_midst_r_alpha']
                        tmp_l_alpha1 = tmp_l_alpha1[self.circ] if isinstance(tmp_l_alpha1, Iterable) else tmp_l_alpha1
                        tmp_r_alpha1 = tmp_r_alpha1[self.circ] if isinstance(tmp_r_alpha1, Iterable) else tmp_r_alpha1
                        tmp_l_alpha2 = self.APART_args[f'LFSse_miden_l_alpha']
                        tmp_r_alpha2 = self.APART_args[f'LFSse_miden_r_alpha']
                        tmp_l_alpha2 = tmp_l_alpha2[self.circ] if isinstance(tmp_l_alpha2, Iterable) else tmp_l_alpha2
                        tmp_r_alpha2 = tmp_r_alpha2[self.circ] if isinstance(tmp_r_alpha2, Iterable) else tmp_r_alpha2
                    
                        # 打印存在于 reactions_group 但是不存在于 tmp_eq_dict 中的key
                        print(f"reactions_group[2] - tmp_eq_dict.keys(): {set(reactions_group[2]) - set(tmp_eq_dict.keys())}")

                        for reac in reactions_group[2]:
                            A0_len = len(np.array(tmp_eq_dict[reac]).flatten())
                            sensitivity = np.log10(LFS_sensitivity[reac])
                            tmp_l_alpha = tmp_l_alpha1 + (tmp_l_alpha2 - tmp_l_alpha1) * (max_LFS_mid - sensitivity) / (max_LFS_mid - min_LFS_mid)
                            tmp_r_alpha = tmp_r_alpha1 + (tmp_r_alpha2 - tmp_r_alpha1) * (max_LFS_mid - sensitivity) / (max_LFS_mid - min_LFS_mid)
                            self.alpha_dict.update(
                                {reac: (tmp_l_alpha, tmp_r_alpha)}
                            )
                            self.l_alpha.extend([tmp_l_alpha] * A0_len)
                            self.r_alpha.extend([tmp_r_alpha] * A0_len)
                    # 第 0 类推荐完全不调整; 但是依然可以设置 overse_l_alpha 和 overse_r_alpha 来调整
                    elif i == 0:
                        # 获取 overse_l_alpha 和 overse_r_alpha
                        overse_l_alpha = self.APART_args.get('LFSse_overse_l_alpha', 0)
                        overse_r_alpha = self.APART_args.get('LFSse_overse_r_alpha', 0)
                        overse_l_alpha = overse_l_alpha[self.circ] if isinstance(overse_l_alpha, Iterable) else overse_l_alpha  
                        overse_r_alpha = overse_r_alpha[self.circ] if isinstance(overse_r_alpha, Iterable) else overse_r_alpha
                        self.alpha_dict.update(
                            {key: (overse_l_alpha, overse_r_alpha) for key in tmp_eq_dict.keys()}
                        ) 
                        self.l_alpha.extend([overse_l_alpha] * len(tmp_A0))
                        self.r_alpha.extend([overse_r_alpha] * len(tmp_A0))  
                    # 更新 self.eq_dict
                    self.eq_dict.update(tmp_eq_dict)
                # self.l_alpha 按照 original_eq_dict 的顺序排列，跳过不存在的反应
                self.eq_dict = {key: self.eq_dict[key] for key in original_eq_dict.keys() if key in self.eq_dict.keys()}
                l_alpha_dict = {key: self.alpha_dict[key][0] for key in original_eq_dict.keys()}
                r_alpha_dict = {key: self.alpha_dict[key][1] for key in original_eq_dict.keys()}
                self.l_alpha = eq_dict_broadcast2Alist(l_alpha_dict, self.eq_dict)
                self.r_alpha = eq_dict_broadcast2Alist(r_alpha_dict, self.eq_dict)

        self.A0 = eq_dict2Alist(self.eq_dict)  
        self.APART_args['eq_dict'] = self.eq_dict
        self.l_alpha = np.array(self.l_alpha); self.r_alpha = np.array(self.r_alpha)
        self.GenAPARTDataLogger.info(f"Current CIRC alphas are {np.unique(self.l_alpha)} ~ {np.unique(self.r_alpha)}")
        write_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json", self.eq_dict, cover = True)
        write_json_data(f"./data/APART_data/reduced_data/alpha_dict_circ={self.circ}.json", self.alpha_dict, cover = True)
        self.WriteCurrentAPART_args(eq_dict = self.eq_dict, l_alpha = self.l_alpha, r_alpha = self.r_alpha)


    def GenASampleThreshold(self, best_chem_LFS, threshold_expand_factor = None, **kwargs):
        """
        生成采样的阈值; 根据 best_chem_LFS 生成采样阈值. 需要在 init 中就启动
        生成的具体策略为: 
            根据 best_chem_IDT/LFS/LFS 中的每一项，计算其与真实值的误差 Rlos，乘以系数 threshold_expand_factor 后，
            以此作为 IDT/LFS/LFS 采样阈值；同时，如果 thresholds 是一个列表，将阈值设置为
            Rlos * threshold_expand_factor * thresholds[i] / thresholds[i - 1]
        add
            self.idt_threshold, self.lfs_threshold, self.lfs_threshold
        """
        self.GenAPARTDataLogger.info(f"Start GenASampleThreshold with best_chem_lfs: best_chem_LFS: {best_chem_LFS}")
        lfs_threshold = self.APART_args['lfs_threshold']
        lfs_threshold = np.array(lfs_threshold)[self.circ - 1] if isinstance(lfs_threshold, Iterable) else lfs_threshold 
        threshold_expand_factor = self.APART_args.get('threshold_expand_factor', 1.5) if threshold_expand_factor is None else threshold_expand_factor
        lfs_Rlos = np.linalg.norm(best_chem_LFS - self.true_lfs_data)
        self.GenAPARTDataLogger.info(f"Current CIRC Rlos are {lfs_Rlos}")
        if self.circ >= 1:
            if isinstance(lfs_threshold, Iterable):
                self.lfs_threshold = min(lfs_Rlos * threshold_expand_factor * lfs_threshold[self.circ] / lfs_threshold[self.circ - 1], lfs_threshold)
                self.lfs_threshold = max([min(self.lfs_threshold, lfs_threshold), 0.1])
            else:
                self.lfs_threshold = max([min(lfs_Rlos * threshold_expand_factor, lfs_threshold), 0.1])
        else:
            lfs_threshold = np.array(lfs_threshold)[0] if isinstance(lfs_threshold, Iterable) else lfs_threshold 
            self.lfs_threshold = max(min(lfs_Rlos * threshold_expand_factor, lfs_threshold), 0.1)

        self.GenAPARTDataLogger.info(f"Current CIRC thresholds are fixed {self.lfs_threshold}")
        self.GenAPARTDataLogger.info("=" * 50)


    def SortALIST(self, apart_data_path = None, experiment_time = 50, cluster_ratio = False, 
                LFS_reduced_threshold = None, return_all = False, father_sample_save_path = None,
                    logger = None, ord = np.inf, **kwargs) -> np.ndarray:
        """
        从 apart_data.npz 中筛选出来最接近真实 lfs 的采样点，以反问题权重作为 lfs 和 LFS 的筛选权重
        增加一个限制： 筛选出的结果到真实值的距离不能比 Reduced 结果差大于 lfs_reduced_threshold 和 LFS_reduced_threshold 倍
        1. 以此作为反问题初值; 2. 用于 ASample 筛选
        params:
            w1, w2: lfs/LFS 的权重。
            apart_data_path: 输入 apart_data.npz
            experiment_time: 最后返回的列表大小
            cluster_ratio: 是否使用聚类模式; 如果为 int 类型，则表示聚类中初始点的数量
            need_lfs: 如果需要筛选出的 Alist 对应的 apart_data， 将此键值改为 True
            father_sample_save_path: 保存 father_sample 的路径
            SortALIST_T_threshold: 筛选出的结果中，温度误差不能高于 SortALIST_T_threshold
            SortALIST_T_threshold_ratio: 筛选出的结果中，
                        温度误差不能高于 SortALIST_T_threshold_ratio * self.Temperature_Diff; 优先级高于 SortALIST_T_threshold
            lfs_mean; lfs_std
            need_lfs: 如果需要筛选出的 Alist 对应的 apart_data， 将此键值改为 True
        """
        if apart_data_path is None: apart_data_path = self.apart_data_path
        if logger is None: logger = self.GenAPARTDataLogger
        assert self.LFS_mode is True
        apart_data = np.load(apart_data_path)
        Alist = apart_data['Alist']; 
        apart_data_lfs = apart_data['all_lfs_data']
        true_lfs = self.true_lfs_data; reduced_lfs = self.reduced_lfs_data

        # experiment_time 可以存放比例值
        if experiment_time < 1: experiment_time = int(experiment_time * apart_data_lfs.shape[0])

        assert Alist.shape[0] == apart_data_lfs.shape[0] != 0, "Alist or apart_data_lfs is empty!"
        # 初始化筛选向量 if_lfs_pass
        if_lfs_pass = np.ones(apart_data_lfs.shape[0], dtype = bool)
        # 筛选出的结果不能比 Reduced 结果差大于  LFS_reduced_threshold 倍
        if not LFS_reduced_threshold is None:
            if_lfs_pass *= np.all(np.abs(true_lfs - apart_data_lfs) <= LFS_reduced_threshold * np.abs(reduced_lfs - true_lfs), axis = 1)
            if if_lfs_pass.sum() == 0: logger.warning(
                f"No sample pass the LFS_reduced_threshold filter! apart_data - true = {np.mean(np.abs(true_lfs - apart_data_lfs))} and true - reduce = {np.abs(reduced_lfs - true_lfs)}"
                )

        # 计算 diff_lfs 并筛选出其值最小的样本集合
        apart_data_lfs_log2 = np.maximum(apart_data_lfs, self.true_lfs_data)
        diff_lfs = np.abs(apart_data_lfs_log2 - self.true_lfs_data)
        ## 筛选出 diff_lfs 最小的样本集合
        if_lfs_pass *= np.all(diff_lfs <= np.amin(diff_lfs, axis = 0) + 1, axis = 1)
        if if_lfs_pass.sum() == 0: logger.warning(
                f"No sample pass the lfs filter! apart_data_lfs = {apart_data_lfs} and true_lfs_data = {self.true_lfs_data}"
                )
        diff_lfs = np.linalg.norm(apart_data_lfs_log2 - self.true_lfs_data, axis = 1, ord = ord)
        # 计算 DIFF_lfs 和 DIFF_lfs
        diff_lfs = np.linalg.norm(
            self.LFS_weight * (apart_data_lfs - self.true_lfs_data), 
            axis = 1, ord = ord)
        
        Alist = Alist[if_lfs_pass, :]; apart_data_lfs = apart_data_lfs[if_lfs_pass, :]
        diff_lfs = diff_lfs[if_lfs_pass]; apart_data_lfs = apart_data_lfs[if_lfs_pass, :]; apart_data_lfs = apart_data_lfs[if_lfs_pass, :]
        diff_lfs = diff_lfs[if_lfs_pass]
        # DIFF_lfs 暂时删除或者只有 diff_lfs 非常小才启用
        diff = diff_lfs 
        index = np.argsort(diff); Alist = Alist[index,:]
        apart_data_lfs = apart_data_lfs[index,:]; apart_data_lfs = apart_data_lfs[index,:]

        # 如果使用聚类模式
        if cluster_ratio:
            cluster_weight = kwargs.get('cluster_weight', 0.1)
            cluster_ratio = int(0.5 * experiment_time) if cluster_ratio is True else cluster_ratio
            cluster_ratio = min(cluster_ratio, int(0.8 * experiment_time))
            init_Alist = Alist[0:int(experiment_time - cluster_ratio), :]
            unchecked_Alist = Alist[int(experiment_time - cluster_ratio):, :]
            init_Alist_mean = np.mean(init_Alist, axis = 0)
            cluster_loss = 1 / np.linalg.norm(unchecked_Alist - init_Alist_mean, axis = 1, ord = ord)
            cluster_loss /= np.amax(cluster_loss)
            # diff = diff[index][int(experiment_time - cluster_ratio):] 加上 cluster_loss 再排序
            diff = diff[index][int(experiment_time - cluster_ratio):] + cluster_loss * cluster_weight
            # Alist = init_Alist + unchecked_Alist[np.argsort(diff), :]
            Alist = np.vstack((init_Alist, unchecked_Alist[np.argsort(diff), :]))


        # 若 Alist 为空则报错
        if Alist.shape[0] == 0: raise ValueError("Alist is empty in the Function SortAList!")
        if not father_sample_save_path is None:
           np.savez(father_sample_save_path, Alist = Alist[0:experiment_time, :], LFS = apart_data_lfs[0:experiment_time, :]) 
        if return_all:
            return Alist[0:experiment_time, :], apart_data_lfs[0:experiment_time, :]
        else:
            return Alist[0:experiment_time, :]
        

    def ASample(self, sample_size = None, coreAlist = None, passing_rate_upper_limit = None, LFS_reduced_threshold = None, shrink_delta = None,
                cluster_ratio = False, father_sample_save_path = None, start_circ = 0, 
                shrink_ratio = None, average_timeout_time = 0.072, sampling_expand_factor = None, **kwargs):
        """
        重写了 APART.ASample_lfs_LFS 函数，主要是为了增加针对不同反应调节采样范围的函数
        202230730: 增加了通过率上限 passing_rate_upper_limit, 是否启用 LFS 和 LFS 训练的 if_LFS_filter 与 if_LFS_filter
        """
        assert self.LFS_mode == True
        np.set_printoptions(precision = 2, suppress = True); t0 = time.time()

        # 预设置
        # 提取采样的左右界限 + 采样阈值
        # 检测类中是否存在 self.lfs_threshold, self.lfs_threshold
        if not hasattr(self, 'lfs_threshold') or not hasattr(self, 'lfs_threshold'):
            lfs_threshold = self.APART_args['lfs_threshold']
            self.lfs_threshold = np.array(lfs_threshold)[self.circ - 1] if isinstance(lfs_threshold, Iterable) else lfs_threshold

        sample_size = self.APART_args['sample_size'] if sample_size is None else sample_size
        core_size = self.APART_args.get('father_sample_size', 1)
        passing_rate_upper_limit = self.APART_args.get('passing_rate_upper_limit', 0.5) if passing_rate_upper_limit is None else passing_rate_upper_limit

        sample_size = np.array(sample_size)[self.circ] if isinstance(sample_size, Iterable) else sample_size    
        core_size = int(np.array(core_size)[self.circ]) if isinstance(core_size, Iterable) else int(core_size)
        passing_rate_upper_limit = np.array(passing_rate_upper_limit)[self.circ - 1] if isinstance(passing_rate_upper_limit, Iterable) else passing_rate_upper_limit
        
        shrink_delta = self.APART_args.get('shrink_delta', 0) if shrink_delta is None else shrink_delta
        shrink_strategy = False if shrink_delta == 0 else self.APART_args.get('shrink_strategy', False)
        shrink_ratio = self.APART_args.get('shrink_delta', 0.01) if shrink_ratio is None else shrink_ratio
        
        threshold_expand_factor = self.APART_args.get('threshold_expand_factor', 1.5)
        sampling_expand_factor = self.APART_args.get('sampling_expand_factor', 2 / threshold_expand_factor) if sampling_expand_factor is None else sampling_expand_factor

        # 将YAML文件的A值提取并构建均匀采样点
        if self.circ == 0 or self.circ == start_circ:
            self.samples = sample_constant_A(sample_size, self.reduced_mech_A0, self.l_alpha, self.r_alpha)
            if shrink_strategy:
                self.boarder_samples = sample_constant_A(int(sample_size * shrink_ratio), self.reduced_mech_A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha),)
        else:
            lfs_net = load_best_dnn(Network_PlainSingleHead, self.model_previous_json, device = 'cpu')

            self.GenAPARTDataLogger.info(f"lfs_threshold: {np.amin(self.lfs_threshold)} ~ {np.amax(self.lfs_threshold)}; sample_size: {sample_size}; father sample size: {core_size}")
            self.GenAPARTDataLogger.info(f"shrink_delta: {shrink_delta}; shrink_strategy: {shrink_strategy}")
            self.GenAPARTDataLogger.info(f"passing_rate_upper_limit: {passing_rate_upper_limit}")
            self.GenAPARTDataLogger.info("="*100)

            # 读取之前的 apart_data.npz 从中选择前 1% 的最优采样点作为核心; 依然选择 IDT 作为指标，不涉及 LFS

            if coreAlist is None:
                cluster_weight = kwargs.get('cluster_weight', 0.1)
                previous_coreAlist = self.SortALIST(
                        apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                        experiment_time = core_size,
                        father_sample_save_path = father_sample_save_path, 
                        logger = self.GenAPARTDataLogger, cluster_ratio = cluster_ratio, cluster_weight = cluster_weight, **kwargs)
                previous_eq_dict = read_json_data(os.path.dirname(self.apart_data_path) + f'/reduced_data/eq_dict_circ={self.circ - 1}.json')
                coreAlist = []
                for A0 in previous_coreAlist:
                    previous_eq_dict = Alist2eq_dict(A0, previous_eq_dict)
                    # 将 self.eq_dict 中与 previous_eq_dict 相同的项替换为 previous_eq_dict 中的值
                    tmp_eq_dict = {
                        key: previous_eq_dict[key] if key in previous_eq_dict else self.eq_dict[key] for key in self.eq_dict.keys()
                    }
                    coreAlist.append(eq_dict2Alist(tmp_eq_dict))
            else:
                core_size = len(coreAlist)
            
            coreAlist = np.array(coreAlist); self.best_sample = coreAlist[0,:]
            np.save(os.path.dirname(self.apart_data_path) + f"/best_sample_circ={self.circ}.npy", coreAlist)
                
            t0 = time.time(); lfs_zero_father_sample_times = 0
            self.samples = []; self.boarder_samples = []; tmp_sample_size = int(2 * (sample_size) // core_size); tmp_sample = []  
            while len(self.samples) < sample_size:
                # 每次采样的样本点不能太少，也不能太多；因此创建自适应调节机制
                if len(tmp_sample) >= sample_size * 0.02:
                    tmp_sample_size = int(sampling_expand_factor * (sample_size - len(self.samples)) // core_size)
                else:
                    tmp_sample_size = int(sampling_expand_factor * sample_size // core_size)
                for A0 in coreAlist:
                    try:  
                        tmp_sample, lfs_pred_data = SampleAWithNet(lfs_net, self.true_lfs_data, threshold = self.lfs_threshold, 
                                                    size = tmp_sample_size,  A0 = A0, l_alpha = self.l_alpha,  r_alpha = self.r_alpha, save_path = None, passing_rate_upper_limit = passing_rate_upper_limit, debug = True,
                                                    reduced_data = self.reduced_lfs_data)
                        if shrink_strategy:
                            tmp_boarder_sample = SampleAWithNet(lfs_net, self.true_lfs_data, threshold = self.lfs_threshold + 1, 
                                                    size = int(tmp_sample_size * shrink_ratio), A0 = A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                    r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha), 
                                                    reduced_data = self.reduced_lfs_data, reduced_threshold = None, debug = False)
                            self.boarder_samples.extend(tmp_boarder_sample.tolist())
                        
                        lfs_Rlos = np.abs(np.mean(lfs_pred_data - self.true_lfs_data, axis = 0))
                        self.GenAPARTDataLogger.info(f"LFS: On Average, Working Condition Index which not satisfy threshold IS {np.where(lfs_Rlos > self.true_lfs_data)[0]}")
                        # self.GenAPARTDataLogger.info(f"LFS: On Average, Working Condition which not satisfy threshold IS {np.unique(np.tile(self.LFS_condition, (1,3))[np.where(lfs_Rlos > true_lfs)[0],:], axis = 0)}")
                        self.GenAPARTDataLogger.info(f"true LFS is {self.true_lfs_data}")
                        self.GenAPARTDataLogger.info(f"First element of sample prediction LFS is {lfs_pred_data[0]}")
                        self.GenAPARTDataLogger.info(f"Abs mean Difference between true LFS and sample prediction is {lfs_Rlos}")
                        self.GenAPARTDataLogger.info(f"LFS: Sampled {len(tmp_sample)} in this iter, pass Rate is {len(tmp_sample) / tmp_sample_size * 100} %, total size is {len(self.samples) + len(tmp_sample)}! cost {time.time() - t0:.2f}s; Memory usage: {psutil.Process(os.getpid()).memory_info().rss:.2e} B")  
                        self.GenAPARTDataLogger.info(f"LFS: Sampled {len(tmp_boarder_sample)} Boarder Data in this iter, total size is None! Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024:.2e} GB")  
                        self.GenAPARTDataLogger.info("——"*100)
                        if len(tmp_sample) == 0:
                            raise ValueError("tmp_sample is Empty!")
                    except ValueError or IndexError:
                        self.GenAPARTDataLogger.warning(f"LFS: tmp_sample is Empty in This circle! lfs_zero_father_sample_times = {lfs_zero_father_sample_times}")
                        self.GenAPARTDataLogger.info("——"*100)
                        lfs_zero_father_sample_times += 1
                        continue

                    self.samples.extend(tmp_sample.tolist())
                    pass_rate = len(tmp_sample) / tmp_sample_size
                    # pass_rate 太高调小 lfs_threshold 和 lfs_threshold
                    if pass_rate > passing_rate_upper_limit * 0.9 and pass_rate > 0.5:
                        self.lfs_threshold = self.lfs_threshold * 0.8
                        self.GenAPARTDataLogger.warning(f"LFS: pass_rate is {pass_rate * 100} %, which is too high! lfs_threshold have changed to log2({self.lfs_threshold}) = {self.lfs_threshold}")
                    
                    # 样本太多立即退出
                    if len(self.samples) >= sample_size * 1.2:
                        self.GenAPARTDataLogger.warning(f"LFS: Stop the Sampling Process, Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                        self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                        break

                # 时间过长退出机制
                if time.time() - t0 > sample_size * average_timeout_time:
                    self.GenAPARTDataLogger.warning("Error: Function Asample has timed out!")
                    break
                self.GenAPARTDataLogger.info(f"In this Iteration, LFS: Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                # 如果过长时间无法筛选到合适的点，稍微拓宽采样的 threshold
                if lfs_zero_father_sample_times > 10* len(coreAlist):
                    self.lfs_threshold = 2 * self.lfs_threshold
                    self.GenAPARTDataLogger.warning(f"LFS: lfs_threshold: log2 {self.lfs_threshold} = {self.lfs_threshold};")
                    lfs_zero_father_sample_times = 0
        
        self.samples = np.array(self.samples)
        self.father_samples = coreAlist
        if len(self.samples) == 0:
            raise ValueError("self.samples is Empty! End the whole process! Please check the threshold!")
        np.save(f'./data/APART_data/Asamples_{self.circ}.npy', self.samples)  
        self.APART_args['lfs_threshold'] = self.lfs_threshold
        self.APART_args['passing_rate_upper_limit'] = passing_rate_upper_limit
        self.APART_args['sample_size'] = sample_size
        # 保存 APART_args 到 self.current_json_path
        self.WriteCurrentAPART_args(cover = True)
        
        if shrink_strategy:
            self.boarder_samples = np.array(self.boarder_samples)
            self.boarder_samples = self.boarder_samples[0:min(int(shrink_ratio * sample_size), len(self.boarder_samples))]
            np.save(f'./data/APART_data/Aboarder_samples_{self.circ}.npy', self.boarder_samples) 
        self.GenAPARTDataLogger.info(f"End The ASample Progress! The size of samples is {len(self.samples)}, the size of boarder_samples is None, cost {time.time() - t0:.2f}s")  
        return self.samples
    
        
    def GenDataFuture(self, samples:np.ndarray = None, start_sample_index = 0, cpu_process = None, ignore_error_path = None, save_path = "./data/APART_data/tmp", **kwargs):
        """
        使用 future 模块的 ProcessPoolExecutor 生成数据; 
        params:
            samples: np.ndarray, 用于生成数据的初始点集
            idt_cut_time_alpha: float, 计算 IDT 的 cut time 阈值系数
            lfs_error_tol: float, LFS 误差容忍度
            start_sample_index: int, 从 samples 的第几个点开始生成数据
            cpu_process: int, 使用的 cpu 核心数
            ignore_error_path: str, 保存 ignore_error 的路径
            save_path: str, 保存数据的路径
            LFS_exp_factor: float, LFS 的缩减指数因子
        """
        self.GenAPARTDataLogger.info(f"Start the GenAPARTData_multiprocess at circ {self.circ}")
        mkdirplus(save_path)
        
        cpu_process = self.APART_args['cpu_process'] if cpu_process is None else cpu_process
        expected_sample_size = self.APART_args['sample_size'] if isinstance(self.APART_args['sample_size'], int) else self.APART_args['sample_size'][self.circ]
        
        RES = []
        def callback(status):
            status = status.result()
            if not status is None:
                RES.append(status)

        sample_length = np.size(samples, 0)
        with ProcessPoolExecutor(max_workers = cpu_process) as exec:
            for index in range(sample_length):
                try:
                    future = exec.submit(
                            GenOneDataLFS, 
                            index = index + start_sample_index,
                            LFS_condition = self.LFS_condition,
                            Alist = samples[index],
                            eq_dict = self.eq_dict,
                            fuel = self.fuel,
                            oxidizer = self.oxidizer,
                            reduced_mech = self.reduced_mech,
                            my_logger = self.GenAPARTDataLogger,
                            **kwargs
                            ).add_done_callback(callback)
                except Exception as r:
                    self.GenAPARTDataLogger.info(f'Multiprocess error; error reason:{r}')
                finally:
                    # 若 sample_length 达到了预期的样本数, 则退出
                    if index - len(RES) >= expected_sample_size:
                        break
        if not ignore_error_path is None:
            np.save(ignore_error_path, np.array(RES))  
        return sample_length - len(RES)
    

    def _TrainDataProcess(self, load_file_name = None, concat_pre = False,  rm_tree = True, 
                              shrink_strategy = False, extract_strategy = False,rate = 0.8, device = 'cuda', 
                              one_data_save_path = "./data/APART_data/tmp", net_data_dirpath = "./data/APART_data/ANET_data", **kwargs) -> None:
        """
        基本上和 APART 中的对应函数功能相同，增加了将 0D 末态温度同时也作为训练网络数据集的一部分的功能
        """
        # gather_data 部分
        aboarder_data_path = os.path.dirname(self.apart_data_path) + f"/Aboarder_apart_data_circ={self.circ}.npz"
        aboarder_one_data_path = os.path.dirname(one_data_save_path) + f"/boarder_tmp"
        if not os.path.exists(self.apart_data_path) or os.path.getsize(self.apart_data_path) /1024 /1024 <= 1:
            self.gather_apart_data(save_path = one_data_save_path, rm_tree = rm_tree, LFS_mode = True, **kwargs)
        if (shrink_strategy and os.path.exists(aboarder_one_data_path)) or \
            (os.path.exists(aboarder_data_path) and os.path.getsize(aboarder_data_path) /1024 /1024 > 1):
            load_file_name = [
                self.apart_data_path,
                aboarder_data_path,
            ]
            if not os.path.exists(aboarder_data_path) or os.path.getsize(aboarder_data_path) /1024 /1024 <= 1:
                self.gather_apart_data(
                    save_path = aboarder_one_data_path, rm_tree = rm_tree, save_file_name = aboarder_data_path
                    , LFS_mode = True, **kwargs
                )
        elif concat_pre:
            data_dirname = os.path.dirname(load_file_name)
            load_file_name = read_all_files(data_dirname, 'apart_data')
        else:
            load_file_name = self.apart_data_path
        self.TrainAnetLogger.info('Generating DNN Data...')
        
        if isinstance(load_file_name, str) and not concat_pre:
            Data = np.load(load_file_name)
            Alist_data = Data['Alist']
            all_lfs_data = Data['all_lfs_data'] 
            # 判断是否加载之前的 data:
        else:
            Alist_data, all_lfs_data = [], []
            for file in load_file_name:   
                Data = np.load(file)
                Alist_data.extend(Data['Alist'])
                all_lfs_data.extend(Data['all_lfs_data'])
        
        # 进入抽取模式，抽取之前数据集中表现最优秀的前 5% 样本点
        if extract_strategy and self.circ != 0:
            extract_rate = kwargs.get('extract_rate', 0.05)
            self.TrainAnetLogger.info(f'Extracting DNN Data From Previous Best Sample..., extract_rate = {extract_rate}')
            for circ in range(self.circ):
                apart_data_path = os.path.dirname(self.apart_data_path) + f"/apart_data_circ={circ}.npz"
                if os.path.exists(apart_data_path) and os.path.getsize(apart_data_path) /1024 /1024 > 1:
                    Alist, LFS, LFS = self.SortALIST(
                        apart_data_path = apart_data_path,
                        experiment_time = extract_rate,
                        return_all = True,
                        logger = self.TrainAnetLogger
                    )
                    Alist_data.extend(Alist)
                    all_lfs_data.extend(LFS)
                else:
                    self.TrainAnetLogger.warning(f"Extracting: Can't find {apart_data_path} at circ = {circ}, skip")
        
        # 转化为 ndarray 加快转化 tensor 速度 + IDT 对数处理 + LFS_ log2 处理
        all_lfs_data, Alist_data  = np.array(all_lfs_data), np.array(Alist_data)
        
        # 如果 Alist_data 是 1 维数组，提示错误：检查 Alist 的长度是否一样
        if len(Alist_data.shape) == 1: 
            raise ValueError(f"Alist_data is 1D array, check the length of Alist_data is same or not")

        self.TrainAnetLogger.info(f"The size of all_lfs_data is {all_lfs_data.shape}, Alist_data is {Alist_data.shape}, all_lfs_data is {all_lfs_data.shape}")
        # 使用 numpy 剔除 all_lfs_data 中与 self.time 中相差大于 4 的数据点
        self.TrainAnetLogger.info(f"all_lfs_data is {all_lfs_data}, the first one is {all_lfs_data[0]}, with max {np.max(all_lfs_data)} and min {np.min(all_lfs_data)}")
        self.TrainAnetLogger.info(f"self.true_lfs_data is {self.true_lfs_data}, with max {np.max(self.true_lfs_data)} and min {np.min(self.true_lfs_data)}")
        
        # 如果样本点 LFS 与真实值相差过大，我们不愿意见到这样的现象，故而删除这些样本点
        tmp_APART_args = read_json_data(self.model_current_json)
        LFS_threshold = tmp_APART_args.get("lfs_threshold", 0.5)
        LFS_threshold = LFS_threshold[self.circ] if isinstance(LFS_threshold, Iterable) else LFS_threshold
        if_pass = np.all(np.abs(all_lfs_data - self.true_lfs_data) <= LFS_threshold + 2, axis = 1)
        
        self.TrainAnetLogger.info(f"DataProcessing: LFS_threshold: {LFS_threshold} Passing LFS: if_pass rate is {np.sum(if_pass) / if_pass.shape[0] * 100} %")
        passed_lfs_data, passed_A = all_lfs_data[if_pass], Alist_data[if_pass]

        # 只保留 sample size 个数据点
        expected_sample_size = self.APART_args['sample_size'] if isinstance(self.APART_args['sample_size'], int) else self.APART_args['sample_size'][self.circ]
        if passed_lfs_data.shape[0] > expected_sample_size:
            passed_lfs_data, passed_A = passed_lfs_data[:expected_sample_size], passed_A[:expected_sample_size]

        assert passed_lfs_data.shape[0] > 0 and passed_A.shape[0] > 0, f"Error! passed_lfs_data.shape = {passed_lfs_data.shape}, passed_A.shape = {passed_A.shape}"
        
        self.APART_args['input_dim'], self.APART_args['output_dim'] = Alist_data.shape[1], all_lfs_data.shape[1]
        self.APART_args['train_size'] = int(Alist_data.shape[0] * rate)
        self.APART_args['test_size'] = Alist_data.shape[0] - self.APART_args['train_size']

        self.TrainAnetLogger.info(f"The size of LFS  train set is {self.APART_args['train_size']}; test set is {self.APART_args['test_size']}")

        # LFS_ 数据集单独列出
        dataset = DATASET_SingleHead(
            data_A = torch.tensor(Alist_data, dtype = torch.float32),
            data_QoI = torch.tensor(all_lfs_data, dtype = torch.float32),
            device = device
        )
        train_data, test_data = random_split( dataset
            , lengths = [self.APART_args['train_size'], self.APART_args['test_size']])
        self.train_loader = DataLoader(train_data,
                                       shuffle = True,
                                       batch_size = self.APART_args['batch_size'],
                                       drop_last = True)   
        # 单独写出来 test 的数据
        test_loader = DataLoader(test_data,
                                shuffle = False,
                                batch_size = len(test_data))
        for x_test, lfs_test in test_loader:
            self.test_loader = (x_test.to(device), lfs_test.to(device))


        if not net_data_dirpath is None:
            np.savez(
                net_data_dirpath + f"/ANET_Dataset_train={self.APART_args['train_size']}_circ={self.circ}.npz",
                train_data_x = dataset[train_data.indices][0].cpu().detach().numpy(),
                train_data_y = dataset[train_data.indices][1].cpu().detach().numpy(),
                test_data_x = dataset[test_data.indices][0].cpu().detach().numpy(),
                test_data_y = dataset[test_data.indices][1].cpu().detach().numpy(),
            )


    def _DeePMO_train(self, device = None, LFS_train_outside_weight = 1, **kwargs):
        
        """
        在原有训练代码的基础上，增加了对 LFS_ 的训练。首先进行 IDT / LFS 的训练，
        紧接着再次单独训练一个 LFS  网络
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        model_pth_path  = self.model_path
        model_loss_path = self.model_loss_path

        t0 = time.time()
        input_dim, output_dim = self.APART_args['input_dim'], self.APART_args['output_dim']
        tmp_save_path = mkdirplus(f'{model_pth_path}/tmp')
        self.TrainAnetLogger.info(f'A -> LFS training has started...; circ = {self.circ}')
        input_dim, output_dim = self.APART_args['input_dim'], self.APART_args['output_dim']
        hidden_units = self.APART_args.get('hidden_units', self.APART_args['hidden_units'])
        ANET = Network_PlainSingleHead(input_dim, hidden_units, output_dim).to(device)
        optimizer = torch.optim.Adam(ANET.parameters(), lr = self.APART_args['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.APART_args['lr_decay_step'], gamma = self.APART_args['lr_decay_rate'])
        # 第一次保存初始模型
        state = {'model':ANET.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':0}
        torch.save(state, f'{tmp_save_path}/Network_lfs_epoch_{0}.pth')  # 保存DNN模型
        # 开始训练
        train_his2, test_his2 = [], []; criterion = nn.MSELoss()
        epoch_index = []
        for epoch in range(self.APART_args['epoch']):
            # 预测
            with torch.no_grad():
                x_dnn_test, lfs_test = self.test_loader
                lfs_pred = ANET.forward(x_dnn_test)
                loss = criterion(lfs_pred, lfs_test)
                test_loss = loss.item()
            # 训练
            train_loss = 0
            for _, (x_train_batch, lfs_train_batch) in enumerate(self.train_loader):  # 按照 batch 进行训练
                x_train_batch, lfs_train_batch = x_train_batch.to(device), lfs_train_batch.to(device)
                lfs_train_batch_pred = ANET.forward(x_train_batch)
                train_loss_batch = criterion(lfs_train_batch_pred, lfs_train_batch)
                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()
                train_loss += train_loss_batch.item()
            
            scheduler.step()
            batch_num = len(self.train_loader)
            train_loss /= batch_num
            train_his2.append(train_loss); test_his2.append(test_loss); epoch_index.append(epoch);           

            if epoch % 5 == 0:
                GPUtil.showUtilization()
                self.TrainAnetLogger.info(f"epoch: {epoch}\t train loss: {train_loss:.3e} "+
                                            f"test loss: {test_loss:.3e} ,"+
                                            f"time cost: {int(time.time()-t0)} s   lr:{optimizer.param_groups[0]['lr']:.2e}")
            if (epoch == 0) or ((epoch - 25) % 50 == 0):
                state = {'model':ANET.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}   
                torch.save(state, f'{tmp_save_path}/Network_lfs_epoch_{epoch}.pth')  # 保存DNN模型  

        # 构建 early stopping 必需的文件夹, 注意这里的 early stopping 是以 lfs 作为基准的，因为相比其他两个指标 lfs 更加重要
        early_stopping_file = mkdirplus(f'{model_pth_path}/early_stopping')
        train_his2, test_his2 = np.array(train_his2), np.array(test_his2)

        # early stopping 平均 earlystopping_step 的误差求最小
        earlystopping_step = min(self.APART_args['epoch'], 50)

        test_loss_sum = np.sum(test_his2.reshape(-1, earlystopping_step, 1)[...,0], axis = 1)
        stop_index = int(earlystopping_step * np.argmin(test_loss_sum) + earlystopping_step / 2)
        best_pth_path = f'{early_stopping_file}/best_stopat_{stop_index}_circ={self.circ}.pth'
        self.APART_args['best_ppth'] = best_pth_path
        try:
            shutil.copy(f'{tmp_save_path}/Network_lfs_epoch_{stop_index}.pth', best_pth_path)
            shutil.rmtree(tmp_save_path, ignore_errors=True); mkdirplus(tmp_save_path) # 删除中间文件 
        except:
            self.TrainAnetLogger.warning(f"copy file error, {tmp_save_path}/Network_lfs_epoch_{stop_index}.pth not exist")

        # 保存实验结果
        np.savez(f'{model_loss_path}/Network_lfs_circ={self.circ}.npz', # 保存DNN的loss
                epoch_index = epoch_index, 
                train_his = train_his2, 
                test_his = test_his2,
                stop_index = stop_index,
                circ = self.circ)
        self.APART_args['stop_epoch'] = int(stop_index)

        return best_pth_path, best_pth_path


    def DeePMO_train(self, concat_pre = False,  rm_tree = True, shrink_strategy = None, extract_strategy = None, 
                           rate = 0.8, device = None, LFS_train_outside_weight = 1,
                           load_file_name = None, one_data_save_path = "./data/APART_data/tmp", 
                           net_data_dirpath = "./data/APART_data/ANET_data", lock_json = False, 
                           **kwargs):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                
            else:
                device = torch.device("cpu")
        self.TrainAnetLogger.info(f"ANET_train_LFS: Device is {device}")
        assert self.LFS_mode is True
        t0 = time.time()  

        concat_pre = self.APART_args.get('concat_pre', False) if concat_pre is None else concat_pre
        shrink_strategy = self.APART_args.get('shrink_strategy', False) if shrink_strategy is None else shrink_strategy
        extract_strategy = self.APART_args.get('extract_strategy', False) if extract_strategy is None else extract_strategy
        extract_rate = self.APART_args.get('extract_rate', 0.05) if 'extract_rate' not in kwargs else kwargs.pop('extract_rate')
        rate = self.APART_args.get('rate', 0.8) if rate is None else rate

        self._TrainDataProcess(
            load_file_name = load_file_name,
            concat_pre = concat_pre,
            shrink_strategy = shrink_strategy,
            extract_strategy = extract_strategy,
            rm_tree = rm_tree,
            rate = rate,
            device = device,
            one_data_save_path = one_data_save_path,
            net_data_dirpath = net_data_dirpath,
            extract_rate = extract_rate,
            **kwargs
        )
        best_ppth, LFS_best_ppth = self._DeePMO_train(device = device, LFS_train_outside_weight = LFS_train_outside_weight)
        
        # 保存DNN的超参数到JSON文件中
        self.WriteCurrentAPART_args(cover = True)
        if lock_json: 
            subprocess.run(f"chmod 444 {self.model_current_json}", shell = True)
        self.TrainAnetLogger.info("="*100)  
        self.TrainAnetLogger.info(f"{best_ppth=}")  
        self.TrainAnetLogger.info(f"{LFS_best_ppth=}")
        tmp_data = np.load(f"{self.model_loss_path}/Network_lfs_circ={self.circ}.npz")
        train_his1 = tmp_data['train_his']; test_his1 = tmp_data['test_his']; stop_index = tmp_data['stop_index']
        
        fig, ax = plt.subplots(1, 1, figsize = (16, 4))
        ax.semilogy(train_his1, lw=1, label='LFStinction_train')
        ax.semilogy(test_his1, 'r', lw=1.2, label='LFStinction_LFStest')
        ax.axvline(stop_index, label = 'early stopping', color = 'green')
        ax.set_xlabel('epoch');
        ax.set_ylabel('loss (log scale)')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/loss_his_circ={self.circ}.png')
        plt.close(fig)

        self.TrainAnetLogger.info(f"Finished Train DNN! Total cost {time.time() - t0:.2f} s")


    def SkipSolveInverse(self, father_sample:str = None, save_dirpath = f'./inverse_skip', device = 'cpu', experiment_time = 15, 
                         ref_true_LFS_data = None, ref_LFS_condition = None, **kwargs):
        """
        自动跳过 Inverse 部分直接使用最优样本就可以拿到结果，可以直接放在训练步骤之后使用
        """
        np.set_printoptions(suppress=True, precision=3)
        save_folder = mkdirplus(save_dirpath)
        # 读取本次 json 文件中的 lfs_mean 与 lfs_std
        # 加载最优的样本
        if not father_sample is None and os.path.exists(father_sample):
            tmp_father_sample = np.load(father_sample)
            inverse_alist = tmp_father_sample['Alist']
        else:
            inverse_alist = self.SortALIST(self.apart_data_path, experiment_time = experiment_time,)
        # 加载网络
        optim_lfs_net = load_best_dnn(Network_PlainSingleHead, self.model_current_json, device = device,)
        for index in range(experiment_time):
            self.InverseLogger.info(f'experiment_index: {index}')
            inverse_path = mkdirplus(save_folder + f'/{index}')
            try:  
                # IDT图像部分
                t0 = time.time()
                # 生成初值机理
                A_init = np.array(inverse_alist[index], dtype = np.float64)
                Adict2yaml(eq_dict = self.eq_dict, original_chem_path = self.reduced_mech, chem_path = inverse_path +'/optim_chem.yaml', Alist = A_init)
                t1 = time.time(); self.InverseLogger.info('solve inverse done! time cost:%s seconds' % (t1 - t0))
                
                # LFS part
                cantera_lfs_data = yaml2FS_Mcondition(inverse_path + '/optim_chem.yaml', FS_condition = self.LFS_condition,  fuel = self.fuel, oxidizer = self.oxidizer,
                                                      cpu_process = os.cpu_count() - 1)
                relative_error = np.mean(np.abs((cantera_lfs_data - self.true_lfs_data) / self.true_lfs_data)) * 100
                self.InverseLogger.info(f"Relative Error is {relative_error * 100} %")  
                self.InverseLogger.info("-" * 90)                
                CompareDRO_LFS_lineplot(
                    detail_lfs = self.true_lfs_data,
                    reduced_lfs = self.reduced_lfs_data,
                    optimal_lfs = cantera_lfs_data,
                    FS_condition = self.LFS_condition,
                    save_path = inverse_path + '/compare_nn_LFS.png',
                    n_col = 6,
                )
                
                self.InverseLogger.info(f"log2 Relative Error is {relative_error * 100} %")
                # log2 scale
                final_lfs = optim_lfs_net(torch.tensor(A_init, dtype = torch.float32)).detach().numpy()
                self.InverseLogger.info("Compare First LFS:" + "\n" + f"True:{self.true_lfs_data}; " + "\n" +
                                        f"Cantera:{cantera_lfs_data};" + f"Final:{2 ** final_lfs};")
                self.InverseLogger.info("-" * 90)

                compare_nn_train3(
                        cantera_lfs_data,
                        self.true_lfs_data,
                        self.reduced_lfs_data,
                        final_lfs, 
                        labels = [r'$Optimal$', r'$Reduced$', r'$DNN\_Optimal$'],
                        markers = ['+', '+', 'o'],
                        colors = ['blue', 'red', 'blue'],
                        title = f'LFS  Relative Error: {relative_error:.2f} %',
                        save_path = inverse_path + '/compare_nn_LFS.png',
                        wc = self.LFS_condition
                    )
                    
                # 保存 lfs 的相关数据
                np.savez(
                        inverse_path + "/result_data.npz",
                        true_lfs_data = self.true_lfs_data,
                        reduced_lfs_data = self.reduced_lfs_data,
                        cantera_lfs_data = cantera_lfs_data,
                        dnn_lfs_data = np.array(final_lfs),
                        Alist = A_init
                        )
                if not ref_true_LFS_data is None and not ref_LFS_condition is None:
                    ref_optim_LFS_data = yaml2FS_Mcondition(inverse_path + '/optim_chem.yaml', FS_condition = ref_LFS_condition,  fuel = self.fuel, oxidizer = self.oxidizer,
                                                      cpu_process = os.cpu_count() - 1)
                    ref_reduced_LFS_data_path = f"{save_folder}/ref_reduced_LFS_data.npz"
                    if not os.path.exists(ref_reduced_LFS_data_path):
                        ref_reduced_LFS_data = yaml2FS_Mcondition(self.reduced_mech, FS_condition = ref_LFS_condition,  fuel = self.fuel, oxidizer = self.oxidizer,
                                                        cpu_process = os.cpu_count() - 1)
                        np.savez(ref_reduced_LFS_data_path, ref_reduced_LFS_data = ref_reduced_LFS_data)
                    else:
                        ref_reduced_LFS_data = np.load(ref_reduced_LFS_data_path)['ref_reduced_LFS_data']
                    CompareDRO_LFS_lineplot(
                        detail_lfs = ref_true_LFS_data,
                        reduced_lfs = ref_reduced_LFS_data,
                        optimal_lfs = ref_optim_LFS_data,
                        FS_condition = ref_LFS_condition,
                        save_path = inverse_path + '/compare_ref_LFS.png',
                        n_col = 6,
                    )
                self.InverseLogger.info('plot compare picture done! time cost:%s seconds' % (time.time() - t1))
            except Exception:
                exstr = traceback.format_exc()
                self.InverseLogger.info(f'!!ERROR:{exstr}')


    def gather_apart_data(self, save_path = "./data/APART_data/tmp", save_file_name = None,  rm_tree = True, 
                            cover = True, logger = None, **kwargs):
            """
            重写特化于当前脚本的 gather apart data; 不继承原有的函数
            """
            t0 = time.time()
            save_file_name = self.apart_data_path if save_file_name is None else save_file_name
            files = [file for file in os.listdir(save_path) if file.find('.npz') != -1]
            filenums = len(files); logger = self.TrainAnetLogger if logger is None else logger
            if cover or not os.path.exists(save_file_name) or os.path.getsize(save_file_name) /1024/1024 <= 1:
                all_lfs_data, Alist_data = [], []
                for target_file in files:
                    target_file = os.path.join(save_path, target_file)
                    if os.path.getsize(target_file) >= 5:
                        tmp = np.load(target_file)
                        if len(tmp['Alist']) == 0: break
                        all_lfs_data.append(tmp['LFS'].tolist());   Alist_data.append(tmp['Alist'].tolist()); 
                        if len(Alist_data) % 100 == 0:
                            logger.info(f"Cost {time.time() - t0:.1f}, Gather Data Process has finished {len(Alist_data)/filenums * 100:.2f} %")
                np.savez(save_file_name,
                                all_lfs_data = all_lfs_data, Alist = Alist_data,)
                logger.info("apart_data Saved!")
                if rm_tree:
                    # 不使用 rm -rf 的原因是因为 rm -rf 删除大批量文件的效率太低
                    # 使用 rsync 命令删除文件夹
                    logger.info(f"removing the tmp files in {save_path}")
                    mkdirplus("./data/APART_data/blank_dir")
                    subprocess.run(f"rsync --delete-before -d -a ./data/APART_data/blank_dir/ {save_path}/", shell = True)
                return all_lfs_data, Alist_data
            else:
                self.GenAPARTDataLogger.warning("gather_apart_data function is out-of-commision")



    def SortALISTStat(self, apart_data_path = None, father_sample_path = None):
        """
        用于分析每步采样后得到的 apart_data 中数据的分布情况，分为 

        1.  Alist 与 A0 之间的距离，Alist 之间的距离，Alist 在每个维度上的方差，Alist 在每个维度上分布的图像
        3.  LFS 在每个工况和每个特定的 res_time 上的分布情况; LFS 与 IDT 的相关性
        4.  Mole 在每个工况上的分布情况; Mole 与 IDT 的相关性
        5.  LFS 在每个工况上的分布情况; LFS 与 IDT 的相关性
        """
        np.set_printoptions(precision = 2, suppress = True)
        apart_data_path = self.apart_data_path if apart_data_path is None else apart_data_path
        logger = Log(f"./log/SortALISTStat_circ={self.circ}.log"); mkdirplus("./analysis")
        # 读取数据
        apart_data = np.load(apart_data_path)
        Alist = apart_data['Alist']; LFS = apart_data['all_lfs_data']
        dist_A0 = np.mean(Alist - self.reduced_mech_A0, axis = 0)
        experiment_time = dist_A0.shape[0]
        # 样本点之间的平均距离
        dist_center = np.zeros(experiment_time)
        for i in range(experiment_time):
            for j in range(i+1, experiment_time):
                dist_center += np.abs(Alist[i, :] - Alist[j, :])
        dist_center /= (experiment_time ** 2 - experiment_time) / 2
        # 样本点在每个维度上的方差值
        std_sample = np.std(Alist, axis = 0)

        logger.info("SortALISTStat Result:")
        logger.info("="*50)
        logger.info(f"样本点与原点 A0 的平均距离:{dist_A0}")
        logger.info(f"样本点之间的平均距离:{dist_center}")
        logger.info(f"样本点在每个维度上的方差值:{std_sample}")

        # 加载 father_sample 与 A0 的差值
        if father_sample_path is None: father_sample_path = os.path.dirname(apart_data_path) + f"/best_sample_circ={self.circ}.npy"
        father_sample = np.load(father_sample_path) - self.A0
        ## 统计 father_sample 在每个列上正值和负值个数
        father_sample_positive = np.sum(father_sample >= 0, axis = 0)
        father_sample_negative = np.sum(father_sample < 0, axis = 0)
        ## logger 记录; 首先使用 Alist2eq_dict 形成字典: {equation: [father_sample_positive, father_sample_negative]}; 之后使用 logger 记录
        father_sample_positive_dict = Alist2eq_dict(
            father_sample_positive, self.eq_dict
        )
        father_sample_negative_dict = Alist2eq_dict(
            father_sample_negative, self.eq_dict
        )
        logger.info(f"father_sample 在每个列上正值个数:{father_sample_positive_dict}")
        logger.info(f"father_sample 在每个列上负值个数:{father_sample_negative_dict}")

        # 使用小提琴图绘制 Alist - self.reduced_mech_A0 的分布情况; 使用 seaborn 的小提琴图
        fig, ax = plt.subplots(1, 2,figsize = (14, 5))
        sns.violinplot(data = Alist - self.reduced_mech_A0, ax = ax[0])
        sns.violinplot(data = father_sample, ax = ax[1])
        ax[0].set_title("Alist - reduced_mech_A0 Distribution")
        ax[0].set_xticks(np.arange(1, len(self.reduced_mech_A0) + 1))
        ax[0].set_xticklabels([f"{i}" for i in range(1, len(self.reduced_mech_A0) + 1)])
        ax[0].set_xlabel("Dimension")
        ax[0].set_ylabel("Distribution of Sample in A")
        ax[1].set_title("Father Sample Distribution")
        ax[1].set_xticks(np.arange(1, len(self.reduced_mech_A0) + 1))
        ax[1].set_xticklabels([f"{i}" for i in range(1, len(self.reduced_mech_A0) + 1)])
        ax[1].set_xlabel("Dimension")
        fig.savefig(f"./analysis/Alist-A0_circ={self.circ}.png")
        plt.close(fig)
        logger.info("Alist - A0 的分布情况已保存至 ./analysis/Alist-A0.png")


        # 调用 sample_distribution 函数绘制 lfs 与 LFS 的分布情况
        ## 加载上一步中的最优样本点
        previous_best_chem = f"./data/APART_data/reduced_data/previous_best_chem_circ={self.circ}.yaml"
        previous_best_chem_LFS = yaml2FS_Mcondition(
            previous_best_chem, FS_condition = self.LFS_condition, fuel = self.fuel, oxidizer = self.oxidizer,
        )
        ## 调用 sample_distribution 函数绘制 lfs 与 LFS 的分布情况
        from Apart_Package.APART_plot.APART_plot import sample_distribution
        asamples = np.load(os.path.dirname(apart_data_path) + f"/Asamples_{self.circ}.npy")
        network = load_best_dnn(Network_PlainSingleHead, self.model_current_json, device = 'cpu')
        LFS_func = lambda x: network(torch.tensor(x, dtype = torch.float32)).detach().numpy()
        sample_distribution_IDT(
                                apart_data['all_lfs_data'],
                                self.true_lfs_data,
                                self.reduced_lfs_data,
                                marker_lfs=previous_best_chem_LFS,
                                LFS_func = LFS_func, asamples = asamples,
                                FS_condition = self.LFS_condition,
                                save_path = f"./analysis/SampleDistribution_LFS_circ={self.circ}.png"
        )

        return dist_A0, dist_center, std_sample


def GenOneDataLFS(LFS_condition: np.ndarray, Alist:list, eq_dict:dict, reduced_mech:str, 
                  index:int,  my_logger:Log, fuel:str = None, oxidizer: str = None, 
                  tmp_chem_path:str = None, remove_chem = True,  save_path = 'data/APART_data/tmp', 
                  LFS_fuel:str = None, LFS_oxidizer: str = None, **kwargs):
    """
    将 IDT 和 LFS 结合起来
    相比 IDT 的函数增加了 LFS_condition 和 RES_TIME_LIST, lfs_error_tol 3个参数
    """
    tmp_chem_path = save_path + f'/{index}th.yaml' if tmp_chem_path is None else tmp_chem_path
    save_path = save_path + f'/{index}th.npz'
    Adict2yaml(reduced_mech, tmp_chem_path, eq_dict = eq_dict, Alist = Alist)
    t0 = time.time()
    if LFS_fuel is None: LFS_fuel = fuel
    if LFS_oxidizer is None: LFS_oxidizer = oxidizer
    # 将 LFS_condition RES_TIME_LIST 等转化为 array
    LFS_condition = np.array(LFS_condition)
    LFS = _GenOneLFS(LFS_condition, LFS_fuel, LFS_oxidizer, tmp_chem_path, index, my_logger, **kwargs)
    if remove_chem: os.remove(tmp_chem_path)
    if isinstance(LFS, int):
        return LFS
    else:
        my_logger.info(f"Mechanism {index} : cost {time.time()-t0:.2f} s, the first LFS element is {LFS[0:3]}, " + 
                       f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024:.2e} GB")
        np.savez(save_path, LFS = LFS, Alist = Alist,)
        return None
