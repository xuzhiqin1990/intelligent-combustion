# -*- coding:utf-8 -*-

import os, sys, time, shutil, psutil, GPUtil, traceback, subprocess
# 在程序运行之前且导入 torch 之前先确定是否使用 GPU
try:
    device = GPUtil.getFirstAvailable(maxMemory=0.5, maxLoad=0.5)
    print("Avaliable GPU is ", device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
except:
    pass

import numpy as np, pandas as pd, seaborn as sns, cantera as ct, torch.nn as nn
import matplotlib.pyplot as plt
from typing import Iterable
from func_timeout import func_set_timeout
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial

from .basic_set import _DeePMO
from APART_base import _GenOneIDT_HRR
from APART_plot.APART_plot import compare_nn_train3, sample_distribution_IDT
from utils.cantera_utils import *
from utils.setting_utils import *
from utils.yamlfiles_utils import * 
from .DeePMO_V1_Network import Network_PlainDoubleHead, DATASET_DoubleHead


class DeePMO_IDT_HRR(_DeePMO):
    """
    在 DeePMO_IDT 的基础上，增加了 HRR 的计算并全面取消了 PSR 的计算，使用 HRR 替换 PSR 的网络计算地位和预测地位
    """
    def __init__(self, circ = 0, basic_set = True, setup_file: str = './settings/setup.yaml', 
                IDT_mode: int = None, SetAdjustableReactions_mode:int = None,
                 previous_best_chem:str = None, GenASampleRange = None, GenASampleRange_mode = None, **kwargs) -> None:
        
        basic_set = kwargs.pop('basic_set', basic_set)
        super().__init__(circ, basic_set, setup_file, SetAdjustableReactions_mode = SetAdjustableReactions_mode, **kwargs)
        GenASampleRange_mode = self.APART_args.get('GenASampleRange_mode', None) if GenASampleRange_mode is None else GenASampleRange_mode
        if GenASampleRange is None:
            GenASampleRange = True if GenASampleRange_mode is not None else False
        # 计算最大热释放率
        self.cal_true_HRR()
        if GenASampleRange:
            if self.circ == 0: 
                previous_best_chem = self.reduced_mech
                previous_best_chem_IDT = self.reduced_idt_data
                previous_best_chem_HRR = self.reduced_hrr_data
            else:
                previous_best_chem, previous_best_chem_IDT, previous_best_chem_HRR, _ = self.SortALIST(
                    apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                    experiment_time = 1,
                    T_threshold_ratio = self.APART_args.get('T_threshold_ratio', None),
                    return_all=True,
                )
                previous_best_chem = np.squeeze(previous_best_chem)
                previous_eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ - 1}.json")
                previous_best_chem = Adict2yaml(self.reduced_mech, f"./data/APART_data/reduced_data/previous_best_chem_circ={self.circ}.yaml", previous_eq_dict, previous_best_chem)
            # previous_best_chem = f"./inverse_skip/circ={self.circ}/0/optim_chem.yaml" if previous_best_chem is None else previous_best_chem
            self.GenASampleRange(mode = GenASampleRange_mode, target_chem = previous_best_chem, **kwargs)
            # !! 暂时不使用 GenASampleThreshold
            # self.GenASampleThreshold( 
            #     best_chem_IDT = previous_best_chem_IDT, 
            #     best_chem_HRR = previous_best_chem_HRR,
            #     **kwargs
            # )
        else:
            # 在不调用 GenAsampleRange 的情况下, 需要根据之前生成的 eq_dict 更新 self.eq_dict 和 self.A0
            self.eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json")
            self.A0 = eq_dict2Alist(self.eq_dict)
            
        self.WriteCurrentAPART_args(
            IDT_weight = self.IDT_weight,
            HRR_weight = self.HRR_weight,
            idt_defined_T_diff = self.idt_defined_T_diff, 
            idt_defined_time_multiple = self.idt_defined_time_multiple, 
            GenASampleRange_mode = GenASampleRange_mode,
            GenASampleRange = GenASampleRange,
            # **currentAPART_args
        )


    def cal_true_HRR(self, save_path = "./data/true_data/true_HRR.npz"):
        """
        计算真实机理的 HRR
        add:
            self.true_hrr_data, self.reduced_hrr_data
        """
        if save_path is not None and os.path.exists(save_path):
            data = np.load(save_path)
            self.true_hrr_data = data['true_hrr_data']; self.reduced_hrr_data = data['reduced_hrr_data']
        else:
            true_idt_data, self.true_hrr_data, true_T_data = yaml2idt_hrr(
                self.detail_mech, self.IDT_condition, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer, 
                cut_time = self.idt_cut_time, idt_defined_T_diff=self.idt_defined_T_diff, time_multiple = self.idt_defined_time_multiple,
                IDT_mode = self.IDT_mode
            )
            reduced_idt_data, self.reduced_hrr_data, reduced_T_data = yaml2idt_hrr(
                self.reduced_mech, self.IDT_condition, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                cut_time = self.idt_cut_time, idt_defined_T_diff=self.idt_defined_T_diff, time_multiple = self.idt_defined_time_multiple,
                IDT_mode = self.IDT_mode
            )
            if save_path is not None:
                np.savez(save_path, true_hrr_data = self.true_hrr_data, reduced_hrr_data = self.reduced_hrr_data, 
                         true_idt_data = true_idt_data, reduced_idt_data = reduced_idt_data,
                         true_T_data = true_T_data, reduced_T_data = reduced_T_data, condition = self.IDT_condition)


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
                    requirement: self.APART_args 需要以下 keys: IDT_{first/midst/miden/last}_{l/r}_alpha
        """
        adjustable_reactions = adjustable_reactions if adjustable_reactions is not None else list(self.eq_dict.keys())
        adjustable_reactions_eq_dict = {key: self.eq_dict[key] for key in adjustable_reactions}
        
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
    
            case 'locally_IDT_sensitivity':  
                self.l_alpha, self.r_alpha = [], []; self.alpha_dict = {}; self.eq_dict = {}
                IDTsensitivity_json_path = f"./data/APART_data/reduced_data/IDT_sensitivity_circ={self.circ}.json"  

                # 计算 original_eq_dict 中所有反应关于 IDT 的敏感度
                if not os.path.exists(IDTsensitivity_json_path):
                    IDT_sensitivity = yaml2idt_sensitivity(
                                target_chem,
                                IDT_condition = self.IDT_condition,
                                fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                                mode = self.IDT_mode,
                                specific_reactions = adjustable_reactions,
                            )          
                    # IDT_sensitivity 内所有 value 取绝对值后求平均值，替换原来的位置
                    IDT_sensitivity = {k: np.mean(np.abs(v)) for k, v in IDT_sensitivity.items()}
                    # 所有的 value 标准化: value - min(value) / (max(value) - min(value))
                    IDT_sensitivity = {k: (v - min(IDT_sensitivity.values())) / (max(IDT_sensitivity.values()) - min(IDT_sensitivity.values())) for k, v in IDT_sensitivity.items()}                      
                    # IDT_sensitivity 按照 value 重排序并保存
                    IDT_sensitivity = {k: v for k, v in sorted(IDT_sensitivity.items(),
                                                                key=lambda item: item[1], reverse=True)}
                    write_json_data(f"./data/APART_data/reduced_data/IDT_sensitivity_circ={self.circ}.json", 
                                    IDT_sensitivity)
                else:
                    IDT_sensitivity = read_json_data(IDTsensitivity_json_path)
                # 根据 IDT_sensitivity 中的 value 对 original_eq_dict 进行分组: value < 1e-2, 1e-2 < value < 1e-1, 1e-1 < value < 1
                reactions_group = [
                        [k for k, v in IDT_sensitivity.items() if 1e-1 <= v < 1 + 1e-3],
                        [k for k, v in IDT_sensitivity.items() if 1e-2 <= v < 1e-1],
                        [k for k, v in IDT_sensitivity.items() if v <= 1e-2],
                    ]
                # 读取 APART_args 中三组 l_alpha 与 r_alpha 数值: 
                # IDTse_last_l_alpha, IDTse_last_r_alpha 用于统一赋值给第一组 reactions_group[-1]
                # IDTse_midst_l_alpha, IDTse_midst_r_alpha 用于给第2组 reactions_group[-2] 作为最小调整范围
                # IDTse_first_l_alpha, IDTse_first_r_alpha 用于统一赋值给第3组 reactions_group[-3]
                # 第四组 reactions_group[-4] 不做调整 (采样区间为 0)
                for i, sind in enumerate(['first', 'midst', 'last']):
                    tmp_A0, tmp_eq_dict = yaml_eq2A(target_chem, *reactions_group[i], )
                    # 检测 tmp_eq_dict 是否为空
                    if len(reactions_group[i]) == 0:
                        self.GenAPARTDataLogger.info(f"reactions_group[{i}] is empty, break from the GenASampleRange Process!")
                        continue
                    if i != 0:
                        tmp_l_alpha = self.APART_args[f'IDTse_{sind}_l_alpha']
                        tmp_r_alpha = self.APART_args[f'IDTse_{sind}_r_alpha']
                        tmp_l_alpha = tmp_l_alpha[self.circ] if isinstance(tmp_l_alpha, Iterable) else tmp_l_alpha
                        tmp_r_alpha = tmp_r_alpha[self.circ] if isinstance(tmp_r_alpha, Iterable) else tmp_r_alpha
                        # if tmp_l_alpha == 0 and tmp_r_alpha == 0:
                        #     break # 如果 l_alpha 和 r_alpha 都为 0，则不做调整，因此不写如 eq_dict 中
                        self.alpha_dict.update(
                            {key: (tmp_l_alpha, tmp_r_alpha) for key in tmp_eq_dict.keys()}
                        )       
                        self.l_alpha.extend([tmp_l_alpha] * len(tmp_A0))
                        self.r_alpha.extend([tmp_r_alpha] * len(tmp_A0))                 

                    # 第 0 类推荐完全不调整; 但是依然可以设置 overse_l_alpha 和 overse_r_alpha 来调整
                    elif i == 0:
                        # 获取 overse_l_alpha 和 overse_r_alpha
                        overse_l_alpha = self.APART_args.get('IDTse_overse_l_alpha', 0)
                        overse_r_alpha = self.APART_args.get('IDTse_overse_r_alpha', 0)
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
                self.eq_dict = {key: self.eq_dict[key] for key in adjustable_reactions_eq_dict.keys() if key in self.eq_dict.keys()}
                l_alpha_dict = {key: self.alpha_dict[key][0] for key in adjustable_reactions_eq_dict.keys()}
                r_alpha_dict = {key: self.alpha_dict[key][1] for key in adjustable_reactions_eq_dict.keys()}
                self.l_alpha = eq_dict_broadcast2Alist(l_alpha_dict, self.eq_dict)
                self.r_alpha = eq_dict_broadcast2Alist(r_alpha_dict, self.eq_dict)
            case 'Scale_Sensitivity':
                M = np.amax(np.abs(self.true_idt_data - self.reduced_idt_data))
                Delta = self.APART_args.get('sensitivity_scale_coeff', 0.2)
                original_eq_dict = copy.deepcopy(self.eq_dict)
                self.l_alpha, self.r_alpha = [], []; self.alpha_dict = {}; self.eq_dict = {}
                IDTsensitivity_json_path = f"./data/APART_data/reduced_data/IDT_sensitivity_circ={self.circ}.json"  

                # 计算 original_eq_dict 中所有反应关于 IDT 的敏感度
                if not os.path.exists(IDTsensitivity_json_path):
                    IDT_sensitivity = yaml2idt_sensitivity(
                                target_chem,
                                IDT_condition = self.IDT_condition,
                                fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                                mode = self.IDT_mode,
                                specific_reactions = list(original_eq_dict.keys())
                            )          
                    # IDT_sensitivity 内所有 value 取绝对值后求平均值，替换原来的位置
                    IDT_sensitivity = {k: np.mean(np.abs(v)) for k, v in IDT_sensitivity.items()}
                    # IDT_sensitivity 按照 value 重排序并保存
                    IDT_sensitivity = {k: v for k, v in sorted(IDT_sensitivity.items(),
                                                                key=lambda item: item[1], reverse=True)}
                    write_json_data(IDTsensitivity_json_path, IDT_sensitivity)
                else:
                    IDT_sensitivity = read_json_data(IDTsensitivity_json_path)
                # value1 = {k:  np.log10(M / v + 1) for k, v in sensitivity.items()}
                distinctive_func = lambda x: np.min([np.log10(x / M**3 + 1), np.log10(M / x + 1)]) * Delta
                IDT_sensitivity = {k: distinctive_func(v) for k, v in IDT_sensitivity.items()}
                IDT_sensitivity = dict(sorted(IDT_sensitivity.items(), key=lambda x: x[1], reverse=True))
                # 去除 self.eq_dict 中不存在的 key
                IDT_sensitivity = {k: v for k, v in IDT_sensitivity.items() if k in original_eq_dict.keys()}
                self.alpha_dict = {
                    key: [IDT_sensitivity[key] * -1, IDT_sensitivity[key] * 1] for key in IDT_sensitivity.keys()
                }
                l_alpha_dict = {key: self.alpha_dict[key][0] for key in original_eq_dict.keys()}
                r_alpha_dict = {key: self.alpha_dict[key][1] for key in original_eq_dict.keys()}
                self.l_alpha = eq_dict_broadcast2Alist(l_alpha_dict, self.eq_dict)
                self.r_alpha = eq_dict_broadcast2Alist(r_alpha_dict, self.eq_dict)
        self.A0 = eq_dict2Alist(self.eq_dict)  
        self.gen_yaml:Callable = partial(Adict2yaml, eq_dict = self.eq_dict)
        self.APART_args['eq_dict'] = self.eq_dict
        self.l_alpha = np.array(self.l_alpha); self.r_alpha = np.array(self.r_alpha)
        self.GenAPARTDataLogger.info(f"Current CIRC alphas are {np.unique(self.l_alpha)} ~ {np.unique(self.r_alpha)}")
        write_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json", self.eq_dict, cover = True)
        write_json_data(f"./data/APART_data/reduced_data/alpha_dict_circ={self.circ}.json", self.alpha_dict, cover = True)
        self.WriteCurrentAPART_args(eq_dict = self.eq_dict, l_alpha = self.l_alpha, r_alpha = self.r_alpha)


    def GenASampleThreshold(self, best_chem_IDT, best_chem_HRR, threshold_expand_factor = None, **kwargs):
        """
        生成采样的阈值; 根据 best_chem_IDT, best_chem_HRR 生成采样阈值. 需要在 init 中就启动
        生成的具体策略为: 
            根据 best_chem_IDT/HRR 中的每一项，计算其与真实值的误差 Rlos，乘以系数 threshold_expand_factor 后，
            以此作为 IDT/HRR 采样阈值；同时，如果 thresholds 是一个列表，将阈值设置为
            Rlos * threshold_expand_factor * thresholds[i] / thresholds[i - 1]
        add
            self.idt_threshold, self.hrr_threshold
        """
        self.GenAPARTDataLogger.info(f"Start GenASampleThreshold with best_chem_IDT: {best_chem_IDT}, best_chem_HRR: {best_chem_HRR}")
        idt_threshold = self.APART_args['idt_threshold']
        hrr_threshold = self.APART_args['hrr_threshold']
        idt_threshold = np.array(idt_threshold)[self.circ - 1] if isinstance(idt_threshold, Iterable) else idt_threshold
        hrr_threshold = np.array(hrr_threshold)[self.circ - 1] if isinstance(hrr_threshold, Iterable) else hrr_threshold
        threshold_expand_factor = self.APART_args.get('threshold_expand_factor', 1.5) if threshold_expand_factor is None else threshold_expand_factor

        idt_Rlos = 10 ** np.amax(np.abs(np.log10(best_chem_IDT) - np.log10(self.true_idt_data)))  
        hrr_Rlos = 10 ** np.amax(np.abs(np.log10(best_chem_HRR) - np.log10(self.true_hrr_data)))
        self.GenAPARTDataLogger.info(f"Current CIRC Rlos are IDT:{idt_Rlos} ~ HRR:{hrr_Rlos}")
        if self.circ >= 1:
            if isinstance(idt_threshold, Iterable):
                self.idt_threshold = min(idt_Rlos * threshold_expand_factor * idt_threshold[self.circ] / idt_threshold[self.circ - 1], idt_threshold)
            else:
                self.idt_threshold = min(idt_Rlos * threshold_expand_factor, idt_threshold)
            
            if isinstance(hrr_threshold, Iterable):
                self.hrr_threshold = min(hrr_Rlos * threshold_expand_factor * hrr_threshold[self.circ] / hrr_threshold[self.circ - 1], hrr_threshold)
            else:
                self.hrr_threshold = min(hrr_Rlos * threshold_expand_factor, hrr_threshold)

        else:
            idt_threshold = np.array(idt_threshold)[0] if isinstance(idt_threshold, Iterable) else idt_threshold
            hrr_threshold = np.array(hrr_threshold)[0] if isinstance(hrr_threshold, Iterable) else hrr_threshold
            self.idt_threshold = min(idt_Rlos * threshold_expand_factor, idt_threshold)
            self.hrr_threshold = min(hrr_Rlos * threshold_expand_factor, hrr_threshold)

        self.GenAPARTDataLogger.info(f"Current CIRC thresholds are fixed as {np.log10(self.idt_threshold)}(np.log10) ~ {np.log10(self.hrr_threshold)}(np.log10)")
        self.GenAPARTDataLogger.info("=" * 50)
    

    def SortALIST(self, w1 = None, w3 = None, apart_data_path = None, experiment_time = 50, cluster_ratio = False, 
                  IDT_reduced_threshold = None, SortALIST_T_threshold:float = None, SortALIST_T_threshold_ratio:list|float = None,
                  need_idt = False, return_all = False, father_sample_save_path = None,
                  logger = None, ord = np.inf, **kwargs) -> np.ndarray:
        """
        从 apart_data.npz 中筛选出来最接近真实 IDT 的采样点，以反问题权重作为 IDT 和 HRR 的筛选权重
        增加一个限制： 筛选出的结果到真实值的距离不能比 Reduced 结果差大于 IDT_reduced_threshold 和 PSR_reduced_threshold 倍
        1. 以此作为反问题初值; 2. 用于 ASample 筛选
        params:
            w1, w2: IDT/HRR 的权重。
            apart_data_path: 输入 apart_data.npz
            experiment_time: 最后返回的列表大小
            cluster_ratio: 是否使用聚类模式; 如果为 int 类型，则表示聚类中初始点的数量
            need_idt: 如果需要筛选出的 Alist 对应的 apart_data， 将此键值改为 True
            father_sample_save_path: 保存 father_sample 的路径
            SortALIST_T_threshold: 筛选出的结果中，温度误差不能高于 SortALIST_T_threshold
            SortALIST_T_threshold_ratio: 筛选出的结果中，
                        温度误差不能高于 SortALIST_T_threshold_ratio * self.Temperature_Diff; 优先级高于 SortALIST_T_threshold
            need_idt: 如果需要筛选出的 Alist 对应的 apart_data， 将此键值改为 True
        """
        if w1 is None: w1 = self.IDT_weight
        if w3 is None: w3 = self.HRR_weight
        if apart_data_path is None: apart_data_path = self.apart_data_path
        if logger is None: logger = self.GenAPARTDataLogger
        apart_data = np.load(apart_data_path)
        apart_data_idt = apart_data['all_idt_data']; Alist = apart_data['Alist']; 
        apart_data_T = apart_data['all_T_data']; apart_data_hrr = apart_data['all_hrr_data']

        # experiment_time 可以存放比例值
        if experiment_time < 1: experiment_time = int(experiment_time * apart_data_idt.shape[0])

        assert Alist.shape[0] == apart_data_idt.shape[0] != 0, "Alist or apart_data_idt is empty!"
        SortALIST_T_threshold = self.APART_args.get('SortALIST_T_threshold', SortALIST_T_threshold); SortALIST_T_threshold_ratio = self.APART_args.get('SortALIST_T_threshold_ratio', SortALIST_T_threshold_ratio)
        # 若 SortALIST_T_threshold_ratio 不为 None， 则将 SortALIST_T_threshold 设置为 SortALIST_T_threshold_ratio * self.Temperature_Diff
        if not SortALIST_T_threshold_ratio is None: 
            if isinstance(SortALIST_T_threshold_ratio, Iterable): SortALIST_T_threshold_ratio = SortALIST_T_threshold_ratio[self.circ - 1]
            SortALIST_T_threshold = SortALIST_T_threshold_ratio * self.Temperature_Diff
        true_idt = np.log10(self.true_idt_data); reduced_idt = np.log10(self.reduced_idt_data)

        # 初始化筛选向量 if_idt_pass
        if_idt_pass = np.ones(apart_data_idt.shape[0], dtype = bool)

        # 筛选出的结果温度误差不能高于 SortALIST_T_threshold
        if not SortALIST_T_threshold is None:
            tmp_if_idt_pass = if_idt_pass * np.all(np.abs(self.true_T_data - apart_data_T) <= SortALIST_T_threshold, axis = 1)
            # 若 tmp_if_idt_pass 全为 False， 则跳过这一步骤
            if np.any(tmp_if_idt_pass):
                if_idt_pass = tmp_if_idt_pass
            else:
                logger.warning(f'All data is filtered out by SortALIST_T_threshold ratio: {SortALIST_T_threshold_ratio} and SortALIST_T_threshold: {SortALIST_T_threshold},'
                                        + f'But the np.abs(self.true_T_data - apart_data_T) is {np.abs(self.true_T_data - apart_data_T)} so skip this step!')
                
        # 筛选出的结果不能比 Reduced 结果差大于 IDT_reduced_threshold 倍
        if not IDT_reduced_threshold is None:
            if_idt_pass *= np.all(np.abs(true_idt - np.log10(apart_data_idt)) <= IDT_reduced_threshold * np.abs(reduced_idt - true_idt), axis = 1)
            if if_idt_pass.sum() == 0: logger.warning(
                f"No sample pass the IDT_reduced_threshold filter! apart_data - true = {np.mean(np.abs(true_idt - np.log10(apart_data_idt)))} and true - reduce = {np.abs(reduced_idt - true_idt)}"
                )

        # 计算 DIFF_idt 和 DIFF_hrr
        diff_idt = np.linalg.norm(
            w1 * (np.log10(apart_data_idt) - true_idt), 
            axis = 1, ord = ord)
        diff_hrr = np.linalg.norm(
            w3 * (np.log10(apart_data_hrr) - np.log10(self.true_hrr_data)), 
            axis = 1, ord = ord)
        
        diff_idt = diff_idt[if_idt_pass]; diff_hrr = diff_hrr[if_idt_pass]
        Alist = Alist[if_idt_pass, :]; apart_data_idt = apart_data_idt[if_idt_pass, :]
        apart_data_hrr = apart_data_hrr[if_idt_pass, :]; apart_data_T = apart_data_T[if_idt_pass, :]

        diff = diff_idt + diff_hrr
        index = np.argsort(diff); Alist = Alist[index,:]
        apart_data_idt = apart_data_idt[index,:]; apart_data_T = apart_data_T[index,:]; apart_data_hrr = apart_data_hrr[index,:]

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
           np.savez(father_sample_save_path, Alist = Alist[0:experiment_time, :], IDT = apart_data_idt[0:experiment_time, :],) 
        if return_all:
            return Alist[0:experiment_time, :], apart_data_idt[0:experiment_time, :], apart_data_hrr[0:experiment_time, :], apart_data_T[0:experiment_time, :]
        elif need_idt:
            return Alist[0:experiment_time, :], apart_data_idt[0:experiment_time, :]
        else:
            return Alist[0:experiment_time, :]
        

    def ASample(self, sample_size = None, coreAlist = None, IDT_weight = None, HRR_weight = None,
                passing_rate_upper_limit = None, if_HRR_filter:bool = True,
                IDT_reduced_threshold = None, shrink_delta = None,
                cluster_ratio = False, father_sample_save_path = None, start_circ = 0, 
                shrink_ratio = None, average_timeout_time = 0.072, sampling_expand_factor = None, **kwargs):
        """
        重写了 APART.ASample_IDT 函数，主要是为了增加针对不同反应调节采样范围的函数
        202230730: 增加了通过率上限 passing_rate_upper_limit, 是否启用 PSR 和 PSRex 训练的 if_HRR_filter 与 if_PSRex_filter
        """
        np.set_printoptions(precision = 2, suppress = True); t0 = time.time()

        # 预设置
        self.GenAPARTDataLogger.info(f"Start The ASample Process; Here we apply three aspect into consideration: IDT: True, HRR: {if_HRR_filter}")
        # 提取采样的左右界限 + 采样阈值
        if IDT_weight is None: IDT_weight = self.IDT_weight
        if HRR_weight is None: HRR_weight = self.HRR_weight

        # 检测类中是否存在 self.idt_threshold, self.hrr_threshold
        if not hasattr(self, 'idt_threshold') or not hasattr(self, 'hrr_threshold'):
            idt_threshold = self.APART_args['idt_threshold']
            hrr_threshold = self.APART_args['hrr_threshold']
            self.idt_threshold = np.array(idt_threshold)[self.circ - 1] if isinstance(idt_threshold, Iterable) else idt_threshold
            self.hrr_threshold = np.array(hrr_threshold)[self.circ - 1] if isinstance(hrr_threshold, Iterable) else hrr_threshold  

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
        sampling_expand_factor = self.APART_args.get('sampling_expand_factor',3 / threshold_expand_factor) if sampling_expand_factor is None else sampling_expand_factor

        # 将YAML文件的A值提取并构建均匀采样点
        if self.circ == 0 or self.circ == start_circ:
            self.samples = sample_constant_A(sample_size, self.reduced_mech_A0, self.l_alpha, self.r_alpha)
            if shrink_strategy:
                self.boarder_samples = sample_constant_A(int(sample_size * shrink_ratio), self.reduced_mech_A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha),)
        else:
            net = load_best_dnn(Network_PlainDoubleHead, self.model_previous_json, device = 'cpu')
            self.GenAPARTDataLogger.info(f"idt_threshold: log10({self.idt_threshold}) = {np.log10(self.idt_threshold)}; hrr_threshold: {np.log10(self.hrr_threshold)}; sample_size: {sample_size}; father sample size: {core_size}")
            self.GenAPARTDataLogger.info(f"shrink_delta: {shrink_delta}; shrink_strategy: {shrink_strategy}")
            self.GenAPARTDataLogger.info(f"IDT_reduced_threshold: {IDT_reduced_threshold}; passing_rate_upper_limit: {passing_rate_upper_limit}")
            self.GenAPARTDataLogger.info("="*100)

            # 读取之前的 apart_data.npz 从中选择前 1% 的最优采样点作为核心; 依然选择 IDT 作为指标

            if coreAlist is None:
                cluster_weight = kwargs.get('cluster_weight', 0.1)
                previous_coreAlist = self.SortALIST(IDT_weight, HRR_weight, 
                        apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                        experiment_time = core_size, IDT_reduced_threshold = IDT_reduced_threshold, 
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
                
            t0 = time.time(); idt_zero_father_sample_times = 0; hrr_zero_father_sample_times = 0
            self.samples = []; self.boarder_samples = []; tmp_sample = []  
            while len(self.samples) < sample_size:
                # 每次采样的样本点不能太少，也不能太多；因此创建自适应调节机制
                if len(tmp_sample) >= sample_size * 0.02:
                    tmp_sample_size = int(sampling_expand_factor * (sample_size - len(self.samples)) // core_size)
                else:
                    tmp_sample_size = int(sampling_expand_factor * sample_size // core_size)
                for A0 in coreAlist:
                    self.GenAPARTDataLogger.info(f"tmp_sample_size in this circle is {tmp_sample_size}")
                    try:  
                        tmp_sample, idt_pred_data = SampleAWithNet(net.forward_Net1, np.log10(self.true_idt_data), threshold = np.log10(self.idt_threshold), 
                                                    size = tmp_sample_size, A0 = A0, l_alpha = self.l_alpha, passing_rate_upper_limit = np.sqrt(passing_rate_upper_limit),
                                                    r_alpha = self.r_alpha, save_path = None, debug = True, reduced_data = np.log10(self.reduced_idt_data), reduced_threshold = None)
                        if shrink_strategy:
                            tmp_boarder_sample = SampleAWithNet(net.forward_Net1, np.log10(self.true_idt_data), threshold = np.log10(self.idt_threshold) + 1, 
                                                    size = int(tmp_sample_size * shrink_ratio), A0 = A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                    r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha), 
                                                    reduced_data = np.log10(self.reduced_idt_data), reduced_threshold = None, debug = False)
                        
                        idt_Rlos = np.abs(np.mean(idt_pred_data - np.log10(self.true_idt_data), axis = 0))
                        self.GenAPARTDataLogger.info(f"IDT: On Average, Working Condition Index which not satisfy threshold IS {np.where(idt_Rlos > np.log10(self.idt_threshold))[0]}")
                        self.GenAPARTDataLogger.info(f"IDT: On Average, Working Condition which not satisfy threshold IS {self.IDT_condition[np.where(idt_Rlos > np.log10(self.idt_threshold))[0],:]}")
                        self.GenAPARTDataLogger.info(f"true IDT is {np.log10(self.true_idt_data)}")
                        self.GenAPARTDataLogger.info(f"First element of sample prediction IDT is {idt_pred_data[0]}")
                        self.GenAPARTDataLogger.info(f"Abs Difference between true IDT and sample prediction is {idt_Rlos}")
                        self.GenAPARTDataLogger.info(f"IDT: Remain sample size is {len(tmp_sample)}; pass rate is {len(tmp_sample) / tmp_sample_size * 100} %, total size is {len(self.samples) + len(tmp_sample)}! cost {time.time() - t0:.2f}s; Memory usage: {psutil.Process(os.getpid()).memory_info().rss:.2e} B")
                        self.GenAPARTDataLogger.info("-·"*100)
                        if len(tmp_sample) == 0:
                            raise ValueError("tmp_sample is Empty!")
                    except ValueError or IndexError:
                        traceback.print_exc()
                        self.GenAPARTDataLogger.warning(f"IDT: tmp_sample is Empty in This circle! idt_zero_father_sample_times = {idt_zero_father_sample_times}")
                        self.GenAPARTDataLogger.info("-·"*100)
                        idt_zero_father_sample_times += 1
                        continue
                    if if_HRR_filter:
                        try:
                            tmp_sample, hrr_pred_data = SampleAWithNet(net.forward_Net2, np.log10(self.true_hrr_data), threshold = np.log10(self.hrr_threshold), 
                                                        father_samples = tmp_sample, debug = True, passing_rate_upper_limit = np.sqrt(passing_rate_upper_limit),
                                                        # scalar_input = hrr_std, mean_input = hrr_mean
                                                        )
                            if shrink_strategy:
                                tmp_boarder_sample = SampleAWithNet(net.forward_Net2, np.log10(self.true_hrr_data), threshold = np.log10(self.hrr_threshold) + 1, 
                                                        father_samples = tmp_boarder_sample, debug = False,
                                                        #   scalar_input = hrr_std, mean_input = hrr_mean
                                                          )    
                                self.boarder_samples.extend(tmp_boarder_sample.tolist())                  
                            hrr_Rlos = np.abs(np.mean(hrr_pred_data - np.log10(self.true_hrr_data), axis = 0))
                            self.GenAPARTDataLogger.info(f"hrr: On Average, Working Condition Index which not satisfy threshold IS {np.where(hrr_Rlos > np.log10(self.true_hrr_data),)[0]}")
                            # self.GenAPARTDataLogger.info(f"hrr: On Average, Working Condition which not satisfy threshold IS {np.unique(np.tile(self.hrr_condition, (1,3))[np.where(hrr_Rlos > true_hrr)[0],:], axis = 0)}")
                            self.GenAPARTDataLogger.info(f"true hrr is {np.log10(self.true_hrr_data),}")
                            self.GenAPARTDataLogger.info(f"First element of sample prediction hrr is {hrr_pred_data[0]}")
                            self.GenAPARTDataLogger.info(f"Abs Difference between true hrr and sample prediction is {hrr_Rlos}")
                            self.GenAPARTDataLogger.info(f"IDT + HRR: Remained {len(tmp_sample)} in this iter, pass Rate is {len(tmp_sample) / tmp_sample_size * 100} %, total size is {len(self.samples)}! cost {time.time() - t0:.2f}s; Memory usage: {psutil.Process(os.getpid()).memory_info().rss:.2e} B")  
                            self.GenAPARTDataLogger.info(f"IDT + HRR: Sampled {len(tmp_boarder_sample)} Boarder Data in this iter, total size is None! Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024:.2e} GB")  
                            self.GenAPARTDataLogger.info("-·"*100)
                            if len(tmp_sample) == 0:
                                raise ValueError("tmp_sample is Empty!")

                        except ValueError or IndexError:
                            self.GenAPARTDataLogger.warning(f"hrr: tmp_sample is Empty in This circle! hrr_zero_father_sample_times = {hrr_zero_father_sample_times}")
                            self.GenAPARTDataLogger.info("-·"*100)
                            hrr_zero_father_sample_times += 1
                            continue
                    self.samples.extend(tmp_sample.tolist())
                    pass_rate = len(tmp_sample) / tmp_sample_size
                    # pass_rate 太高调小 idt_threshold
                    if pass_rate ** 1/2 > passing_rate_upper_limit * 0.9 and pass_rate  ** 1/2 > 0.5:
                        self.idt_threshold = self.idt_threshold * 0.8
                        self.hrr_threshold = self.hrr_threshold * 0.8
                        self.GenAPARTDataLogger.warning(f"IDT + HRR: pass_rate is {pass_rate * 100} %, which is too high! idt_threshold have changed to log10({self.idt_threshold}) = {np.log10(self.idt_threshold)}")
                    
                    # 样本太多立即退出
                    if len(self.samples) >= sample_size * 1.2:
                        self.GenAPARTDataLogger.warning(f"IDT + HRR: Stop the Sampling Process, Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                        self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                        break

                # 时间过长退出机制
                if time.time() - t0 > sample_size * average_timeout_time:
                    self.GenAPARTDataLogger.warning("Error: Function Asample has timed out!")
                    break
                self.GenAPARTDataLogger.info(f"In this Iteration, IDT + HRR: Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                # 如果过长时间无法筛选到合适的点，稍微拓宽采样的 threshold
                if idt_zero_father_sample_times > 10 * len(coreAlist):
                    self.idt_threshold = 10 ** self.idt_threshold
                    self.GenAPARTDataLogger.warning(f"IDT: idt_threshold have changed to log10{self.idt_threshold} = {np.log10(self.idt_threshold)}")
                    idt_zero_father_sample_times = 0
                if hrr_zero_father_sample_times > 10* len(coreAlist):
                    self.hrr_threshold = 10 ** np.amax(hrr_Rlos)
                    self.GenAPARTDataLogger.warning(f"HRR: hrr_threshold: {np.amin(self.hrr_threshold)} ~ {np.amax(self.hrr_threshold)};")
                    hrr_zero_father_sample_times = 0 
        
        self.samples = np.array(self.samples)
        self.father_samples = coreAlist
        if len(self.samples) == 0:
            raise ValueError("self.samples is Empty! End the whole process! Please check the threshold!")
        np.save(f'./data/APART_data/Asamples_{self.circ}.npy', self.samples)  
        # 更新 idt_threshold 和 hrr_threshold 
        self.APART_args['idt_threshold'] = self.idt_threshold
        self.APART_args['hrr_threshold'] = self.hrr_threshold
        self.APART_args['passing_rate_upper_limit'] = passing_rate_upper_limit
        self.APART_args['sample_size'] = sample_size
        # 保存 APART_args 到 self.current_json_path
        self.WriteCurrentAPART_args(cover = True)
        
        if shrink_strategy:
            self.boarder_samples = np.array(self.boarder_samples)
            # 截取 boarder_samples 前 shrink_ratio * sample_size 个点
            self.boarder_samples = self.boarder_samples[0:min(int(shrink_ratio * sample_size), len(self.boarder_samples))]
            np.save(f'./data/APART_data/Aboarder_samples_{self.circ}.npy', self.boarder_samples) 
        self.GenAPARTDataLogger.info(f"End The ASample Progress! The size of samples is {len(self.samples)}, the size of boarder_samples is None, cost {time.time() - t0:.2f}s")  
        return self.samples
    
        
    def GenDataFuture(self, samples:np.ndarray = None, idt_cut_time_alpha = 1.5, start_sample_index = 0, cpu_process = None, ignore_error_path = None, 
                          save_path = "./data/APART_data/tmp", **kwargs):
        """
        使用 future 模块的 ProcessPoolExecutor 生成数据; 
        params:
            samples: np.ndarray, 用于生成数据的初始点集
            idt_cut_time_alpha: float, 计算 IDT 的 cut time 阈值系数
            start_sample_index: int, 从 samples 的第几个点开始生成数据
            cpu_process: int, 使用的 cpu 核心数
            ignore_error_path: str, 保存 ignore_error 的路径
            save_path: str, 保存数据的路径
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
                            GenOneDataIDT_HRR, 
                            index = index + start_sample_index,
                            IDT_condition = self.IDT_condition,
                            Alist = samples[index],
                            eq_dict = self.eq_dict,
                            fuel = self.IDT_fuel,
                            oxidizer = self.IDT_oxidizer,
                            reduced_mech = self.reduced_mech,
                            my_logger = self.GenAPARTDataLogger,
                            IDT_mode = self.IDT_mode,
                            # 将 idt_arrays 采取如下设置: 其每个分量是 true_idt_data 和 reduced_idt_data 对应分量中大的那个
                            idt_arrays = np.maximum(self.true_idt_data, self.reduced_idt_data),
                            cut_time_alpha = idt_cut_time_alpha,
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
                            shrink_strategy = False, extract_strategy = False,
                            rate = 0.8, device = 'cuda', 
                            one_data_save_path = "./data/APART_data/tmp", 
                            net_data_dirpath = "./data/APART_data/ANET_data", **kwargs) -> None:
        """
        基本上和 APART 中的对应函数功能相同，增加了将 0D 末态温度同时也作为训练网络数据集的一部分的功能
        """
        # gather_data 部分
        aboarder_data_path = os.path.dirname(self.apart_data_path) + f"/Aboarder_apart_data_circ={self.circ}.npz"
        aboarder_one_data_path = os.path.dirname(one_data_save_path) + f"/boarder_tmp"
        if not os.path.exists(self.apart_data_path) or os.path.getsize(self.apart_data_path) /1024 /1024 <= 1:
            self.gather_apart_data(save_path = one_data_save_path, rm_tree = rm_tree, **kwargs)
        if (shrink_strategy and os.path.exists(aboarder_one_data_path)) or \
            (os.path.exists(aboarder_data_path) and os.path.getsize(aboarder_data_path) /1024 /1024 > 1):
            load_file_name = [
                self.apart_data_path,
                aboarder_data_path,
            ]
            if not os.path.exists(aboarder_data_path) or os.path.getsize(aboarder_data_path) /1024 /1024 <= 1:
                self.gather_apart_data(
                    save_path = aboarder_one_data_path, rm_tree = rm_tree, save_file_name = aboarder_data_path
                    , **kwargs
                )
        elif concat_pre:
            data_dirname = os.path.dirname(load_file_name)
            load_file_name = read_all_files(data_dirname, 'apart_data')
        else:
            load_file_name = self.apart_data_path
        self.TrainAnetLogger.info('Generating DNN Data...')
        
        if isinstance(load_file_name, str) and not concat_pre:
            Data = np.load(load_file_name)
            Alist_data = Data['Alist']; all_idt_data =  Data['all_idt_data']; 
            all_T_data = Data['all_T_data']; all_hrr_data = Data['all_hrr_data']
            # 判断是否加载之前的 data:
        else:
            all_idt_data, all_T_data, Alist_data, all_hrr_data = [], [], [],[]
            for file in load_file_name:   
                Data = np.load(file)
                Alist_data.extend(Data['Alist']); all_idt_data.extend(Data['all_idt_data']); 
                all_T_data.extend(Data['all_T_data']); all_hrr_data.extend(Data['all_hrr_data'])
                
        
        # 进入抽取模式，抽取之前数据集中表现最优秀的前 5% 样本点
        if extract_strategy and self.circ != 0:
            extract_rate = kwargs.get('extract_rate', 0.05)
            self.TrainAnetLogger.info(f'Extracting DNN Data From Previous Best Sample..., extract_rate = {extract_rate}')
            for circ in range(self.circ):
                apart_data_path = os.path.dirname(self.apart_data_path) + f"/apart_data_circ={circ}.npz"
                if os.path.exists(apart_data_path) and os.path.getsize(apart_data_path) /1024 /1024 > 1:
                    Alist, IDT, HRR, T = self.SortALIST(
                        apart_data_path = apart_data_path,
                        experiment_time = extract_rate,
                        return_all = True,
                        logger = self.TrainAnetLogger
                    )
                    all_idt_data.extend(IDT); all_T_data.extend(T); Alist_data.extend(Alist)
                    all_hrr_data.extend(HRR)
                else:
                    self.TrainAnetLogger.warning(f"Extracting: Can't find {apart_data_path} at circ = {circ}, skip")
        
        # 转化为 ndarray 加快转化 tensor 速度 + IDT HRR 对数处理 处理
        all_idt_data, all_T_data, Alist_data, all_hrr_data  = \
             np.array(all_idt_data), np.array(all_T_data), np.array(Alist_data), np.array(all_hrr_data)
        all_idt_data = np.log10(all_idt_data); all_hrr_data = np.log10(all_hrr_data)
        # 如果 Alist_data 是 1 维数组，提示错误：检查 Alist 的长度是否一样
        if len(Alist_data.shape) == 1: 
            raise ValueError(f"Alist_data is 1D array, check the length of Alist_data is same or not")

        self.TrainAnetLogger.info(f"The size of all_idt_data is {all_idt_data.shape}, all_T_data is {all_T_data.shape}, Alist_data is {Alist_data.shape}, all_hrr_data is {all_hrr_data.shape}")

        # 只保留 sample size 个数据点
        expected_sample_size = self.APART_args['sample_size'] if isinstance(self.APART_args['sample_size'], int) else self.APART_args['sample_size'][self.circ]
        if all_idt_data.shape[0] > expected_sample_size:
            all_idt_data, all_T_data, Alist_data, all_hrr_data = \
                all_idt_data[:expected_sample_size], all_T_data[:expected_sample_size], \
                Alist_data[:expected_sample_size], all_hrr_data[:expected_sample_size]

        assert all_idt_data.shape[0] > 0 and all_T_data.shape[0] > 0 and Alist_data.shape[0] > 0, \
            f"Error! all_idt_data.shape = {all_idt_data.shape}, all_T_data.shape = {all_T_data.shape}, Alist_data.shape = {Alist_data.shape}"
        
        assert all_hrr_data.shape[0] == all_idt_data.shape[0] == all_T_data.shape[0] == Alist_data.shape[0], \
            f"Error The shape at axis 0 is not same! all_hrr_data.shape = {all_hrr_data.shape}, all_idt_data.shape = {all_idt_data.shape}, all_T_data.shape = {all_T_data.shape}, Alist_data.shape = {Alist_data.shape}"
        # T data 做 zscore 标准化
        all_T_data, T_mean, T_std = zscore(all_T_data)
        self.APART_args.update({'T_mean': T_mean.tolist(), "T_std": T_std.tolist()})

        self.APART_args['input_dim'], self.APART_args['output_dim'] = Alist_data.shape[1], [all_idt_data.shape[1], all_hrr_data.shape[1]]
        self.APART_args['train_size']  = int(Alist_data.shape[0] * rate)
        self.APART_args['test_size'] = Alist_data.shape[0] - self.APART_args['train_size']

        self.TrainAnetLogger.info(f"The size of IDT & HRR train set is {self.APART_args['train_size']}; test set is {self.APART_args['test_size']}")
        dataset = DATASET_DoubleHead(
            data_A = torch.tensor(Alist_data, dtype = torch.float32),
            data_QoI1 = torch.tensor(all_idt_data, dtype = torch.float32),
            data_QoI2 = torch.tensor(all_hrr_data, dtype = torch.float32),
            device = device
        )
        train_data, test_data = random_split(dataset
            , lengths = [self.APART_args['train_size'], self.APART_args['test_size']])

        self.train_loader = DataLoader(train_data, 
                                       shuffle = True, 
                                       batch_size = self.APART_args['batch_size'],
                                       drop_last = True,)
        
        # 单独写出来 test 的数据
        test_loader = DataLoader(test_data, 
                                shuffle = False, 
                                batch_size = len(test_data),)

        for x_test, idt_test, hrr_test in test_loader:
            self.test_loader = (x_test.to(device), idt_test.to(device), hrr_test.to(device))
 

        if not net_data_dirpath is None:
            np.savez(
                net_data_dirpath + f"/ANET_Dataset_train={self.APART_args['train_size']}_circ={self.circ}.npz",
                train_data_x = dataset[train_data.indices][0].cpu().numpy(),
                train_data_y = dataset[train_data.indices][1].cpu().numpy(),
                test_data_x =  dataset[test_data.indices][0].cpu().numpy(),
                test_data_y =  dataset[test_data.indices][1].cpu().numpy(),
            )


    def _DeePMO_train(self, device = None, HRR_train_outside_weight = 1, **kwargs):
        
        """
        在原有训练代码的基础上，增加了对 HRR_extinction 的训练。首先进行 IDT / HRR 的训练，
        紧接着再次单独训练一个 HRR extinction 网络
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        model_pth_path  = self.model_path
        model_loss_path = self.model_loss_path

        t0 = time.time()
        input_dim, output_dim = self.APART_args['input_dim'], self.APART_args['output_dim']
        tmp_save_path = mkdirplus(f'{model_pth_path}/tmp')
        self.TrainAnetLogger.info(f'A -> IDT & HRR training has started...; circ = {self.circ}')
        ANET = Network_PlainDoubleHead(input_dim, self.APART_args['hidden_units'], output_dim).to(device)
        if self.APART_args['optimizer'] == 'Adam': optimizer = optim.Adam(ANET.parameters(), lr = self.APART_args['learning_rate'])
        if self.APART_args['optimizer'] == 'SGD':  optimizer = optim.SGD(ANET.parameters(), lr = self.APART_args['learning_rate'], momentum = 0.1)
        # 第一次保存初始模型
        state = {'model':ANET.state_dict()}   
        torch.save(state, f'{model_pth_path}/Network_PlainDoubleHead_initialized_circ={self.circ}.pth')  # 保存DNN模型
        
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.APART_args['lr_decay_step'], gamma = self.APART_args['lr_decay_rate'])

        epoch_index, train_his1, test_his1= [], [], []
        for epoch in range(self.APART_args['epoch']):
            # 预测
            with torch.no_grad():
                x_dnn_test, idt_test, hrr_test = self.test_loader
                idt_pred, hrr_pred = ANET.forward_Net1(x_dnn_test), ANET.forward_Net2(x_dnn_test)
                loss1 = criterion(idt_pred, idt_test); loss2 = criterion(hrr_pred, hrr_test)
                test_loss = [float(loss1.cpu().detach()), float(loss2.cpu().detach())]
            # 训练
            idt_loss, hrr_loss = 0, 0
            for _, (x_train_batch, idt_train_batch, hrr_train_batch) in enumerate(self.train_loader):  # 按照 batch 进行训练
                idt_pred, hrr_pred = ANET.forward_Net1(x_train_batch), ANET.forward_Net2(x_train_batch)
                loss1 = criterion(idt_pred, idt_train_batch); loss2 = criterion(hrr_pred, hrr_train_batch)

                train_loss_batch = loss1 / loss1.detach() + loss2 / loss2.detach() * HRR_train_outside_weight
                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()
                idt_loss += float(loss1.detach()); hrr_loss += float(loss2.detach())
            
            scheduler.step()
            batch_num = len(self.train_loader)
            idt_loss /=  batch_num; hrr_loss /=  batch_num
            train_his1.append([idt_loss, hrr_loss]); test_his1.append(test_loss)
            epoch_index.append(epoch);           

            if epoch % 5 == 0:
                GPUtil.showUtilization()
                self.TrainAnetLogger.info(f"epoch: {epoch}\t train loss: {idt_loss:.3e} + {hrr_loss:.3e} "+
                                            f"test loss: {test_his1[-1][0]:.3e} + {test_his1[-1][1]:.3e},"+
                                            f"time cost: {int(time.time()-t0)} s   lr:{optimizer.param_groups[0]['lr']:.2e}")
            if (epoch == 0) or ((epoch - 25) % 50 == 0):
                state = {'model':ANET.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}   
                torch.save(state, f'{model_pth_path}/tmp/ANET_epoch_{epoch}.pth')  # 保存DNN模型   

        
        # 构建 early stopping 必需的文件夹, 注意这里的 early stopping 是以 IDT 作为基准的，因为相比其他两个指标 IDT 更加重要
        if not os.path.exists(f'{model_pth_path}/early_stopping'): # 创建临时保存网络参数的文件夹
            os.mkdir(f'{model_pth_path}/early_stopping')
        train_his1, test_his1 = np.array(train_his1), np.array(test_his1)

        # early stopping 平均 earlystopping_step 的误差求最小
        earlystopping_step = min(self.APART_args['epoch'], 50)

        test_loss_sum = np.sum(test_his1.reshape(-1, earlystopping_step, 2)[...,0], axis = 1)
        stop_index = int(earlystopping_step * np.argmin(test_loss_sum) + earlystopping_step / 2)
        best_pth_path = f'{model_pth_path}/early_stopping/model_best_stopat_{stop_index}_circ={self.circ}.pth'
        self.APART_args['best_ppth'] = best_pth_path
        try:
            shutil.copy(f'{model_pth_path}/tmp/ANET_epoch_{stop_index}.pth', best_pth_path)
            shutil.rmtree(f'{model_pth_path}/tmp', ignore_errors=True); mkdirplus(f'{model_pth_path}/tmp') # 删除中间文件 
        except:
            self.TrainAnetLogger.warning(f"copy file error, {model_pth_path}/tmp/ANET_epoch_{stop_index}.pth not exist")

        # 保存实验结果
        np.savez(f'{model_loss_path}/APART_loss_his_circ={self.circ}.npz', # 保存DNN的loss
                epoch_index = epoch_index, 
                train_his = train_his1, 
                test_his = test_his1,
                stop_index = stop_index,
                circ = self.circ)
        self.APART_args['stop_epoch'] = int(stop_index)
        return best_pth_path


    def DeePMO_train(self, concat_pre = False, rm_tree = True, shrink_strategy = None, extract_strategy = None, 
                           rate = 0.8, device = None,
                           load_file_name = None, one_data_save_path = "./data/APART_data/tmp", 
                           net_data_dirpath = "./data/APART_data/ANET_data", lock_json = False, 
                           **kwargs):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.TrainAnetLogger.info(f"DeePMO_train: Device on {device}")
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
        best_ppth = self._DeePMO_train(device = device)
        
        # 保存DNN的超参数到JSON文件中
        self.WriteCurrentAPART_args(cover = True)
        if lock_json: 
            subprocess.run(f"chmod 444 {self.model_current_json}", shell = True)
        self.TrainAnetLogger.info("="*100)  
        self.TrainAnetLogger.info(f"{best_ppth=}")  
        tmp_data = np.load(f"{self.model_loss_path}/APART_loss_his_circ={self.circ}.npz")
        train_his1 = tmp_data['train_his']; test_his1 = tmp_data['test_his']; stop_index = tmp_data['stop_index']
       
        fig, ax = plt.subplots(1, 2, figsize = (16, 4))
        ax[0].semilogy(train_his1[:,0], lw=1, label='IDTtrain')
        ax[0].semilogy(test_his1[:,0], 'r', lw=1.2, label='IDTtest')
        ax[0].axvline(stop_index, label = 'early stopping', color = 'green', )
        ax[0].set_xlabel('epoch'); 
        ax[0].legend(loc='upper right')
        ax[1].semilogy(train_his1[:,1], lw=1, label='HRRtrain')
        ax[1].semilogy(test_his1[:,1], 'r', lw=1.2, label='HRRtest')
        ax[1].axvline(stop_index, label = 'early stopping', color = 'green')
        ax[1].set_xlabel('epoch'); 
        ax[1].set_ylabel('loss (log scale)')
        ax[1].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/loss_his_circ={self.circ}.png')
        plt.close(fig)

        self.TrainAnetLogger.info(f"Finished Train DNN! Total cost {time.time() - t0:.2f} s")


    def SkipSolveInverse(self, w1 = None, w3 = None, father_sample:str = None, save_dirpath = f'./inverse_skip', 
                             csv_path = None, device = 'cpu', IDT_reduced_threshold = None,
                             raw_data = False, experiment_time = 15, **kwargs):
        """
        自动跳过 Inverse 部分直接使用最优样本就可以拿到结果，可以直接放在训练步骤之后使用
        """
        np.set_printoptions(suppress=True, precision=3)
        save_folder = mkdirplus(save_dirpath)
        if w1 is None: w1 = self.IDT_weight
        if w3 is None: w3 = self.HRR_weight

        # 加载最优的样本
        if not father_sample is None and os.path.exists(father_sample):
            tmp_father_sample = np.load(father_sample)
            inverse_alist = tmp_father_sample['Alist']
        else:
            inverse_alist = self.SortALIST(w1, w3, self.apart_data_path, experiment_time = experiment_time,
                    IDT_reduced_threshold = IDT_reduced_threshold, logger = self.InverseLogger)
        # 加载网络
        optim_net = load_best_dnn(Network_PlainDoubleHead, self.model_current_json, device = device)
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
                
                # 简化机理指标 vs 真实机理指标的绘图
                # IDT part
                cantera_idt_data, cantera_hrr_data, _ = yaml2idt_hrr(inverse_path + '/optim_chem.yaml', IDT_mode = self.IDT_mode, IDT_condition = self.IDT_condition, save = False, 
                                               fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer, cut_time = self.idt_cut_time, 
                                               idt_defined_T_diff = self.idt_defined_T_diff, time_multiple = self.idt_defined_time_multiple)
                relative_error = np.mean(np.abs((cantera_idt_data - self.true_idt_data) / self.true_idt_data)) * 100
                log_abs_error = np.mean(np.abs(np.log10(cantera_idt_data) - np.log10(self.true_idt_data)))
                original_log_abs_error = np.mean(np.abs(np.log10(self.true_idt_data) - np.log10(self.reduced_idt_data)))
                self.InverseLogger.info(f"Relative Error is {relative_error} %, Log Abs Error is {log_abs_error:.2f}, Original Log Abs Error is {original_log_abs_error:.2f}")
                # log scale
                true_idt_data = np.log10(self.true_idt_data); cantera_idt_data = np.log10(cantera_idt_data); 
                reduced_idt_data = np.log10(self.reduced_idt_data)
                final_idt = optim_net.forward_Net1(torch.tensor(A_init, dtype = torch.float32)).detach().numpy()

                self.InverseLogger.info(f"Average Diff between final_A and A0 is {np.mean(np.abs(A_init - self.reduced_mech_A0))}; While the min and max is {np.min(np.abs(A_init - self.A0))} and {np.max(np.abs(A_init - self.A0))}")
                self.InverseLogger.info("Compare First IDT:" + "\n" + f"True:{true_idt_data}; " + "\n" + f"Reduced:{reduced_idt_data};\n" + 
                                        f"Cantera:{cantera_idt_data};" + f"Final:{final_idt};")
                self.InverseLogger.info("-" * 90)
                compare_nn_train3(
                        true_idt_data,
                        cantera_idt_data,
                        reduced_idt_data,
                        final_idt, 
                        labels = [r'$Optimal$', r'$Reduced$', r'$DNN\_Optimal$'],
                        markers = ['+', '+', 'o'],
                        colors = ['blue', 'red', 'blue'],
                        title = f'IDT  Relative Error: {relative_error:.2f} %',
                        save_path = inverse_path + '/compare_nn_IDT.png',
                        wc = self.IDT_condition
                    )
                
                # HRR part
                relative_error = np.mean(np.abs((cantera_hrr_data - self.true_hrr_data) / self.true_hrr_data)) * 100
                self.InverseLogger.info(f"HRR Relative Error is {relative_error} %")  
                # log scale
                true_hrr_data = np.log10(self.true_hrr_data); reduced_hrr_data = np.log10(self.reduced_hrr_data); cantera_hrr_data = np.log10(cantera_hrr_data)
                final_hrr = optim_net.forward_Net2(torch.tensor(A_init, dtype = torch.float32)).detach().numpy()
                # self.InverseLogger.info(f"Average Diff between final_A and A0 is {np.mean(np.abs(A_init - self.reduced_mech_A0))}; While the min and max is {np.min(np.abs(A_init - self.A0))} and {np.max(np.abs(A_init - self.A0))}")
                self.InverseLogger.info("Compare First HRR:" + "\n" + f"True:{true_hrr_data}; " + "\n" + f"Reduced:{reduced_hrr_data};\n" + 
                                        f"Cantera:{cantera_hrr_data};" + f"Final:{final_hrr};")
                self.InverseLogger.info("-" * 90)
                compare_nn_train3(
                        true_hrr_data,
                        cantera_hrr_data,
                        reduced_hrr_data,
                        final_hrr, 
                        labels = [r'$Optimal$', r'$Reduced$', r'$DNN\_Optimal$'],
                        markers = ['+', '+', 'o'],
                        colors = ['blue', 'red', 'blue'],
                        title = f'HRR Relative Error: {relative_error:.2f} %',
                        save_path = inverse_path + '/compare_nn_HRR.png',
                        wc = self.IDT_condition
                    )                
                    
                # 保存 IDT 的相关数据
                np.savez(
                        inverse_path + "/IDT_data.npz",
                        true_idt_data = true_idt_data,
                        reduced_idt_data = reduced_idt_data,
                        cantera_idt_data = cantera_idt_data,
                        dnn_idt_data = np.array(final_idt),
                        true_hrr_data = true_hrr_data,
                        reduced_hrr_data = reduced_hrr_data,
                        cantera_hrr_data = cantera_hrr_data,
                        dnn_hrr_data = np.array(final_hrr),
                        Alist = A_init
                        )

                self.InverseLogger.info(f'plot compare idt picture done! time cost:{time.time() - t1} seconds')
            except Exception:
                exstr = traceback.format_exc()
                self.InverseLogger.info(f'!!ERROR:{exstr}')


    def gather_apart_data(self, save_path = "./data/APART_data/tmp", save_file_name = None, rm_tree = True, 
                        cover = True, logger = None, **kwargs):
        """
        重写特化于当前脚本的 gather apart data; 不继承原有的函数
        """
        t0 = time.time()
        save_file_name = self.apart_data_path if save_file_name is None else save_file_name
        files = [file for file in os.listdir(save_path) if file.find('.npz') != -1]
        filenums = len(files); logger = self.TrainAnetLogger if logger is None else logger
        if cover or not os.path.exists(save_file_name) or os.path.getsize(save_file_name) /1024/1024 <= 1:
            all_idt_data, all_T_data, Alist_data, all_hrr_data = [], [], [], []
            for target_file in files:
                target_file = os.path.join(save_path, target_file)
                if os.path.getsize(target_file) >= 5:
                    tmp = np.load(target_file)
                    if len(tmp['Alist']) == 0: break
                    all_idt_data.append(tmp['IDT'].tolist());   all_T_data.append(tmp['T'].tolist())
                    Alist_data.append(tmp['Alist'].tolist()); all_hrr_data.append(tmp['HRR'].tolist())
                    if len(Alist_data) % 100 == 0:
                        logger.info(f"Cost {time.time() - t0:.1f}, Gather Data Process has finished {len(Alist_data)/filenums * 100:.2f} %")
            np.savez(save_file_name,
                            all_idt_data = all_idt_data, all_T_data = all_T_data, 
                            Alist = Alist_data, 
                            all_hrr_data = all_hrr_data)
            logger.info("apart_data Saved!")
            if rm_tree:
                # 不使用 rm -rf 的原因是因为 rm -rf 删除大批量文件的效率太低
                # 使用 rsync 命令删除文件夹
                logger.info(f"removing the tmp files in {save_path}")
                mkdirplus("./data/APART_data/blank_dir")
                subprocess.run(f"rsync --delete-before -d -a ./data/APART_data/blank_dir/ {save_path}/", shell = True)
            return all_idt_data, all_T_data, Alist_data
        else:
            self.GenAPARTDataLogger.warning("gather_apart_data function is out-of-commision")


    def SortALISTStat(self, apart_data_path = None, father_sample_path = None):
        """
        用于分析每步采样后得到的 apart_data 中数据的分布情况，分为 

        1.  Alist 与 A0 之间的距离，Alist 之间的距离，Alist 在每个维度上的方差，Alist 在每个维度上分布的图像
        2.  IDT 在每个工况上的分布情况
        4.  Mole 在每个工况上的分布情况; Mole 与 IDT 的相关性
        5.  LFS 在每个工况上的分布情况; LFS 与 IDT 的相关性
        """
        np.set_printoptions(precision = 2, suppress = True)
        apart_data_path = self.apart_data_path if apart_data_path is None else apart_data_path
        logger = Log(f"./log/SortALISTStat_circ={self.circ}.log"); mkdirplus("./analysis")
        # 读取数据
        apart_data = np.load(apart_data_path)
        Alist = apart_data['Alist']; IDT = apart_data['all_idt_data']
        HRR = apart_data['all_hrr_data']
        # 样本点与原点 A0 的平均距离
        dist_A0 = np.mean(Alist - self.reduced_mech_A0, axis = 0)
        experiment_time = dist_A0.shape[0]
        # 样本点之间的平均距离
        dist_center = np.zeros(experiment_time)
        print(f"shape of dist center and Alist is {dist_center.shape} and {dist_A0.shape}")
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

        # 调用 sample_distribution 函数绘制 IDT 的分布情况
        ## 加载上一步中的最优样本点
        previous_best_chem = f"./data/APART_data/reduced_data/previous_best_chem_circ={self.circ}.yaml"
        previous_best_chem_IDT, previous_best_chem_HRR, _ = yaml2idt_hrr(
            previous_best_chem, IDT_mode = self.IDT_mode,
            IDT_condition = self.IDT_condition, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
            cut_time = self.idt_cut_time, idt_defined_T_diff = self.idt_defined_T_diff, time_multiple = self.idt_defined_time_multiple
        )
        ## 调用 sample_distribution 函数绘制 IDT 的分布情况
        from Apart_Package.APART_plot.APART_plot import sample_distribution
        asamples = np.load(os.path.dirname(apart_data_path) + f"/Asamples_{self.circ}.npy")
        network = load_best_dnn(Network_PlainDoubleHead, self.model_current_json, device = 'cpu')
        IDT_func = lambda x: network.forward_Net1(torch.tensor(x, dtype = torch.float32)).detach().numpy()
        sample_distribution_IDT(
                                np.log10(IDT),
                                np.log10(self.true_idt_data),
                                np.log10(self.reduced_idt_data),
                                marker_idt=np.log10(previous_best_chem_IDT),
                                IDT_func = IDT_func,
                                asamples = asamples,
                                wc_list = self.IDT_condition,
                                save_path = f"./analysis/SampleDistribution_IDT_circ={self.circ}.png",
                                )
        HRR_func = lambda x: network.forward_Net2(torch.tensor(x, dtype = torch.float32)).detach().numpy()
        sample_distribution_IDT(
                                np.log10(HRR),
                                np.log10(self.true_hrr_data),
                                np.log10(self.reduced_hrr_data),
                                IDT_func = HRR_func,
                                asamples = asamples,
                                marker_idt = np.log10(previous_best_chem_HRR),
                                IDT_condition = self.IDT_condition,
                                save_path = f"./analysis/SampleDistribution_HRR_circ={self.circ}.png"
                                )
        logger.info("IDT, HRR 的分布情况已保存至 ./analysis/ 中")
        return dist_A0, dist_center, std_sample



@func_set_timeout(300)
def GenOneDataIDT_HRR(IDT_condition: np.ndarray, Alist:list, eq_dict:dict, reduced_mech:str, 
                      index:int,  my_logger:Log, tmp_chem_path:str = None, fuel:str = None, oxidizer: str = None, 
                      IDT_mode = 0, remove_chem = True, idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                      cut_time_alpha = 10, idt_defined_T_diff = 400, idt_defined_time_multiple = 2, save_path = 'data/APART_data/tmp', 
                      IDT_fuel = None, IDT_oxidizer = None ,**kwargs):

    tmp_chem_path = save_path + f'/{index}th.yaml' if tmp_chem_path is None else tmp_chem_path
    save_path = save_path + f'/{index}th.npz'
    Adict2yaml(reduced_mech, tmp_chem_path, eq_dict = eq_dict, Alist = Alist)
    IDT_condition = np.array(IDT_condition)
    t0 = time.time()
    if IDT_fuel is None: IDT_fuel = fuel
    if IDT_oxidizer is None: IDT_oxidizer = oxidizer
    idt_T = _GenOneIDT_HRR(IDT_condition, IDT_fuel, IDT_oxidizer, tmp_chem_path,  index,  my_logger, IDT_mode,
                     idt_arrays, cut_time, cut_time_alpha, idt_defined_T_diff, idt_defined_time_multiple, **kwargs)
    if remove_chem: os.remove(tmp_chem_path)
    if isinstance(idt_T, int):
        return idt_T
    else:
        IDT, HRR, T = idt_T
        my_logger.info(f"Mechanism {index} : cost {time.time()-t0:.2f} s, the first IDT element is {np.log10(IDT[0]):.3e}," + 
                       f"first HRR element is {HRR[0]:.3e}, Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024:.2e} GB")
        np.savez(save_path, IDT = IDT, T = T, Alist = Alist, HRR = HRR)
        return None
    
