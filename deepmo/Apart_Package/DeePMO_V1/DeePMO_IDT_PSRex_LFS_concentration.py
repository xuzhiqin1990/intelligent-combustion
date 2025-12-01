# -*- coding:utf-8 -*-

import os, time, shutil, psutil, GPUtil, traceback, subprocess
import matplotlib.pyplot as plt

# 在程序运行之前且导入 torch 之前先确定是否使用 GPU
try:
    device = GPUtil.getFirstAvailable(maxMemory=0.5, maxLoad=0.5)
    print("Avaliable GPU is ", device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
except:
    pass

import numpy as np, pandas as pd, seaborn as sns, cantera as ct, torch.nn as nn

from typing import Iterable, final
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .basic_set import _DeePMO, PSRex_shrink_error
from APART_base import _GenOneIDT, _GenOneLFS, _GenOnePSR_with_extinction, _GenOnePSR_concentration
from APART_plot.APART_plot import compare_nn_train3, sample_distribution_IDT, sample_distribution_PSR, sample_distribution_PSRex, compare_PSR_concentration
from utils.cantera_utils import *
from utils.cantera_PSR_definations import *
from utils.setting_utils import *
from utils.yamlfiles_utils import * 
from utils.cantera_multiprocess_utils import *
from .DeePMO_V1_Network import Network_PlainTripleHead_PSRconcentration, Network_PlainSingleHead, DATASET_TripleHead, DATASET_SingleHead
from APART_plot.Result_plot import CompareDRO_PSR_lineplot


class DeePMO_IDT_PSRex_LFS_concentration(_DeePMO):
    """
    在 DeePMO_IDT_PSRex_LFS 的基础上增加了浓度的处理
    """
    def __init__(self, circ = 0, basic_set = True, GenASampleRange = True, setup_file: str = './settings/setup.yaml', 
                 cond_file: str = "./data/true_data/true_psr_data.npz", IDT_mode: int = None, 
                 PSR_mode = True,  SetAdjustableReactions_mode:int = None, GenASampleRange_mode = None,
                 previous_best_chem:str = None, init_res_time = None, need_true_PSRex = False, 
                 PSRex_cutdown = None, **kwargs) -> None:
        
        super().__init__(circ, basic_set, setup_file, cond_file, 
                         SetAdjustableReactions_mode = SetAdjustableReactions_mode, **kwargs)
        GenASampleRange_mode = GenASampleRange_mode if GenASampleRange_mode is not None else self.APART_args.get('GenASampleRange_mode', 'default')
        
        # 确定 init_res_time 为第二个 RES_TIME_LIST 数值
        self.init_res_time = self.RES_TIME_LIST[:,1] if init_res_time is None else init_res_time
        # 确定真实情况下的点火极限，为 RES TIME LIST 中最后一个数值
        # 加载 PSRex 中每步的缩减指数 decay_exp: 计算 PSRex 时会以 0.5 ** decay_exp 为步长进行缩减
        self.PSRex_decay_rate = 2 ** self.APART_args.get("PSR_EXP_FACTOR", 0.5)
        self.cal_reduced_PSRex(need_true = need_true_PSRex)
        # 从 self.APART_args 中获取 PSRex_cutdown 用于指示是否采用 cutdown 策略
        self.PSRex_cutdown = self.APART_args.get("PSRex_cutdown", True) if PSRex_cutdown is None else PSRex_cutdown
        if GenASampleRange:
            if previous_best_chem is None:
                if self.circ == 0: 
                    previous_best_chem = self.reduced_mech
                    previous_best_chem_IDT = self.reduced_idt_data
                    previous_best_chem_LFS = self.reduced_lfs_data
                    previous_best_chem_PSRex = self.reduced_extinction_time
                    previous_best_chem_PSR_concentration = self.reduced_psr_concentration_data
                else:
                    # 加载上一个循环的 psr_mean 和 psr_std
                    previous_json_data = read_json_data(
                        f"./model/model_pth/settings_circ={self.circ - 1}.json"
                    )
                    psr_mean = previous_json_data['psr_mean']; psr_std = previous_json_data['psr_std']
                    previous_best_chem, previous_best_chem_IDT, _, previous_best_chem_PSRex, previous_best_chem_LFS, previous_best_chem_PSR_concentration = self.SortALIST(
                        apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                        experiment_time = 1,
                        T_threshold_ratio = self.APART_args.get('T_threshold_ratio', None),
                        psr_mean = psr_mean, psr_std = psr_std,
                        return_all = True,
                    )
                    previous_best_chem = np.squeeze(previous_best_chem)
                    previous_eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ - 1}.json")
                    previous_best_chem = Adict2yaml(self.reduced_mech, f"./data/APART_data/reduced_data/previous_best_chem_circ={self.circ}.yaml", previous_eq_dict, previous_best_chem)
            # previous_best_chem = f"./inverse_skip/circ={self.circ}/0/optim_chem.yaml" if previous_best_chem is None else previous_best_chem
            self.GenASampleRange(mode = GenASampleRange_mode, target_chem = previous_best_chem, **kwargs)
            self.GenASampleThreshold( 
                best_chem_IDT = previous_best_chem_IDT, best_chem_LFS = previous_best_chem_LFS, best_chem_PSRex = previous_best_chem_PSRex,
                best_chem_PSR_concentration = previous_best_chem_PSR_concentration, **kwargs
            )
        else:
            # 在不调用 GenAsampleRange 的情况下, 需要根据之前生成的 eq_dict 更新 self.eq_dict 和 self.A0
            self.eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json")
            self.A0 = eq_dict2Alist(self.eq_dict)
        # 更新 Current APART args
        # 
        self.WriteCurrentAPART_args(
            init_res_time = self.init_res_time,
            PSR_EXP_FACTOR = self.APART_args.get("PSR_EXP_FACTOR", 0.5),
            PSRex_cutdown = self.APART_args.get("PSRex_cutdown", True),
            GenASampleRange_mode = GenASampleRange_mode,
            GenASampleRange = GenASampleRange,
            PSR_concentration_kwargs = self.PSR_concentration_kwargs
            # **currentAPART_args
        )


    def cal_reduced_PSRex(self, need_true = False, save_path = "./data/true_data/true_psrex.npz"):
        """
        计算简化机理的 PSR extinction time；一般情况下该值应该是与真实机理相同的，但是由于简化机理本身的缺陷，确实可能存在
        简化机理的 PSRex 与真实机理的相差较大的情况。因此，需要计算简化机理的 PSRex
        add:
            self.reduced_extinction_time
        """
        if save_path is not None and os.path.exists(save_path):
            data = np.load(save_path)
            self.true_extinction_time = data['true_extinction_time']
            self.reduced_extinction_time = data['reduced_extinction_time']
        else:
            if need_true:
                t0 = time.time()
                self.true_extinction_time = yaml2psr_extinction_time(
                    self.detail_mech, self.PSR_condition, fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer,
                    psr_tol = 10, init_res_time = self.init_res_time, exp_factor = self.PSRex_decay_rate
                )
                self.GenAPARTDataLogger.info(f"True extinction time calculation costs {time.time() - t0} s")
            else:
                self.true_extinction_time = self.RES_TIME_LIST[:,-1]
            self.reduced_extinction_time = yaml2psr_extinction_time(
                self.reduced_mech, self.PSR_condition, fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer, 
                psr_tol = 10, init_res_time = self.init_res_time, exp_factor = self.PSRex_decay_rate
            )
            if save_path is not None:
                np.savez(save_path, true_extinction_time = self.true_extinction_time, reduced_extinction_time = self.reduced_extinction_time)


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
                    requirement: self.APART_args 需要以下 keys: PSRex_{first/midst/miden/last}_{l/r}_alpha
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
    
            case 'locally_PSR_sensitivity':  
                original_eq_dict = copy.deepcopy(self.eq_dict)
                self.l_alpha, self.r_alpha = [], []; self.alpha_dict = {}; self.eq_dict = {}
                save_jsonpath = kwargs.get('save_jsonpath', None)  
                PSRsensitivity_json_path = f"./data/APART_data/reduced_data/PSR_sensitivity_circ={self.circ}.json"  
                # 计算 original_eq_dict 中所有反应关于 PSR 的敏感度
                if not os.path.exists(PSRsensitivity_json_path):
                    PSR_sensitivity = yaml2psr_sensitivity(
                                target_chem,
                                PSR_condition = self.PSR_condition,
                                RES_TIME_LIST = self.RES_TIME_LIST[:, -3:],
                                fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer,
                                mode = self.PSR_mode, save_path = save_jsonpath,
                                specific_reactions = list(original_eq_dict.keys())
                            )          
                    # PSR_sensitivity 内所有 value 取绝对值后求平均值，替换原来的位置
                    PSR_sensitivity = {k: np.mean(np.abs(v)) for k, v in PSR_sensitivity.items()}
                    # 所有的 value 标准化: value - min(value) / (max(value) - min(value))
                    PSR_sensitivity = {k: (v - min(PSR_sensitivity.values())) / (max(PSR_sensitivity.values()) - min(PSR_sensitivity.values())) for k, v in PSR_sensitivity.items()}                      
                    # PSR_sensitivity 按照 value 重排序并保存
                    PSR_sensitivity = {k: v for k, v in sorted(PSR_sensitivity.items(),
                                                                key=lambda item: item[1], reverse=True)}
                    write_json_data(PSRsensitivity_json_path, PSR_sensitivity)
                else:
                    PSR_sensitivity = read_json_data(PSRsensitivity_json_path)
                # 根据 PSR_sensitivity 中的 value 对 original_eq_dict 进行分组: value < 1e-4, 1e-4 < value < 1e-2, 1e-2 < value < 1e-1, 1e-1 < value < 1
                reactions_group = [
                        [k for k, v in PSR_sensitivity.items() if 1e-1 <= v < 1 + 1e-3],
                        [k for k, v in PSR_sensitivity.items() if 1e-2 <= v < 1e-1],
                        [k for k, v in PSR_sensitivity.items() if 1e-4 <= v < 1e-2],
                        [k for k, v in PSR_sensitivity.items() if v < 1e-4],
                    ]
                # 读取 APART_args 中三组 l_alpha 与 r_alpha 数值: 
                # PSRse_last_l_alpha, PSRse_last_r_alpha 用于统一赋值给第一组 reactions_group[-1]
                # PSRse_miden_l_alpha, PSRse_miden_r_alpha 用于给第二组 reactions_group[-2] 作为最大调整范围
                # PSRse_midst_l_alpha, PSRse_midst_r_alpha 用于给第2组 reactions_group[-2] 作为最小调整范围
                # PSRse_first_l_alpha, PSRse_first_r_alpha 用于统一赋值给第3组 reactions_group[-3]
                # 第四组 reactions_group[-4] 不做调整 (采样区间为 0)
                for i, sind in enumerate(['overse', 'first', ['midst', 'miden'], 'last']):
                    tmp_A0, tmp_eq_dict = yaml_eq2A(target_chem, *reactions_group[i], )
                    # 检测 tmp_eq_dict 是否为空
                    if len(reactions_group[i]) == 0:
                        self.GenAPARTDataLogger.info(f"reactions_group[{i}] is empty, break from the GenASampleRange Process!")
                        continue
                    if i != 2 and i != 0:
                        tmp_l_alpha = self.APART_args[f'PSRse_{sind}_l_alpha']
                        tmp_r_alpha = self.APART_args[f'PSRse_{sind}_r_alpha']
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
                        max_PSR_mid = np.log10(np.max([PSR_sensitivity[k] for k in reactions_group[2]]))
                        min_PSR_mid = np.log10(np.min([PSR_sensitivity[k] for k in reactions_group[2]]))
                        if max_PSR_mid == min_PSR_mid:
                            max_PSR_mid += 1e-5
                        tmp_l_alpha1 = self.APART_args[f'PSRse_midst_l_alpha']
                        tmp_r_alpha1 = self.APART_args[f'PSRse_midst_r_alpha']
                        tmp_l_alpha1 = tmp_l_alpha1[self.circ] if isinstance(tmp_l_alpha1, Iterable) else tmp_l_alpha1
                        tmp_r_alpha1 = tmp_r_alpha1[self.circ] if isinstance(tmp_r_alpha1, Iterable) else tmp_r_alpha1
                        tmp_l_alpha2 = self.APART_args[f'PSRse_miden_l_alpha']
                        tmp_r_alpha2 = self.APART_args[f'PSRse_miden_r_alpha']
                        tmp_l_alpha2 = tmp_l_alpha2[self.circ] if isinstance(tmp_l_alpha2, Iterable) else tmp_l_alpha2
                        tmp_r_alpha2 = tmp_r_alpha2[self.circ] if isinstance(tmp_r_alpha2, Iterable) else tmp_r_alpha2
                    
                        # 打印存在于 reactions_group 但是不存在于 tmp_eq_dict 中的key
                        print(f"reactions_group[2] - tmp_eq_dict.keys(): {set(reactions_group[2]) - set(tmp_eq_dict.keys())}")

                        for reac in reactions_group[2]:
                            A0_len = len(np.array(tmp_eq_dict[reac]).flatten())
                            sensitivity = np.log10(PSR_sensitivity[reac])
                            tmp_l_alpha = tmp_l_alpha1 + (tmp_l_alpha2 - tmp_l_alpha1) * (max_PSR_mid - sensitivity) / (max_PSR_mid - min_PSR_mid)
                            tmp_r_alpha = tmp_r_alpha1 + (tmp_r_alpha2 - tmp_r_alpha1) * (max_PSR_mid - sensitivity) / (max_PSR_mid - min_PSR_mid)
                            self.alpha_dict.update(
                                {reac: (tmp_l_alpha, tmp_r_alpha)}
                            )
                            self.l_alpha.extend([tmp_l_alpha] * A0_len)
                            self.r_alpha.extend([tmp_r_alpha] * A0_len)
                    # 第 0 类推荐完全不调整; 但是依然可以设置 overse_l_alpha 和 overse_r_alpha 来调整
                    elif i == 0:
                        # 获取 overse_l_alpha 和 overse_r_alpha
                        overse_l_alpha = self.APART_args.get('PSRse_overse_l_alpha', 0)
                        overse_r_alpha = self.APART_args.get('PSRse_overse_r_alpha', 0)
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
        self.APART_args['eq_dict'] = self.eq_dict
        self.l_alpha = np.array(self.l_alpha); self.r_alpha = np.array(self.r_alpha)
        self.GenAPARTDataLogger.info(f"Current CIRC alphas are {np.unique(self.l_alpha)} ~ {np.unique(self.r_alpha)}")
        write_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json", self.eq_dict, cover = True)
        write_json_data(f"./data/APART_data/reduced_data/alpha_dict_circ={self.circ}.json", self.alpha_dict, cover = True)
        self.WriteCurrentAPART_args(eq_dict = self.eq_dict, l_alpha = self.l_alpha, r_alpha = self.r_alpha)


    def GenASampleThreshold(self, best_chem_IDT, best_chem_LFS, best_chem_PSRex, best_chem_PSR_concentration,
                            threshold_expand_factor = None, **kwargs):
        """
        生成采样的阈值; 根据 best_chem_IDT, best_chem_LFS, best_chem_PSRex 生成采样阈值. 需要在 init 中就启动
        生成的具体策略为: 
            根据 best_chem_IDT/LFS/PSRex 中的每一项，计算其与真实值的误差 Rlos，乘以系数 threshold_expand_factor 后，
            以此作为 IDT/LFS/PSRex 采样阈值；同时，如果 thresholds 是一个列表，将阈值设置为
            Rlos * threshold_expand_factor * thresholds[i] / thresholds[i - 1]
        add
            self.idt_threshold, self.lfs_threshold, self.psr_extinction_threshold
        """
        self.GenAPARTDataLogger.info(f"Start GenASampleThreshold with best_chem_IDT: {best_chem_IDT}, best_chem_PSR: {best_chem_LFS}, best_chem_PSRex: {best_chem_PSRex}")
        idt_threshold = self.APART_args['idt_threshold']
        lfs_threshold = self.APART_args['lfs_threshold']
        psrex_threshold = self.APART_args['psrex_threshold']
        psr_concentration_threshold = self.APART_args['psr_concentration_threshold']
        
        idt_threshold = np.array(idt_threshold)[self.circ - 1] if isinstance(idt_threshold, Iterable) else idt_threshold
        lfs_threshold = np.array(lfs_threshold)[self.circ - 1] if isinstance(lfs_threshold, Iterable) else lfs_threshold
        psrex_threshold = np.array(psrex_threshold)[self.circ - 1] if isinstance(psrex_threshold, Iterable) else psrex_threshold 
        psr_concentration_threshold = np.array(psr_concentration_threshold)[self.circ - 1] if isinstance(psr_concentration_threshold, Iterable) else psr_concentration_threshold
        threshold_expand_factor = self.APART_args.get('threshold_expand_factor', 1.5) if threshold_expand_factor is None else threshold_expand_factor

        idt_Rlos = 10 ** np.amax(np.abs(np.log10(best_chem_IDT) - np.log10(self.true_idt_data)))  
        lfs_Rlos = np.amax(np.abs(best_chem_LFS - self.true_lfs_data))
        psrex_Rlos = PSRex_shrink_error(np.log2(best_chem_PSRex), np.log2(self.true_extinction_time))
        psr_concentration_Rlos = np.amax(np.abs(best_chem_PSR_concentration - self.true_psr_concentration_data))
        self.GenAPARTDataLogger.info(f"Current CIRC Rlos are IDT:{idt_Rlos} ~ LFS:{lfs_Rlos} ~ PSRex:{psrex_Rlos} ~ PSR_concentration:{psr_concentration_Rlos}")
        if self.circ >= 1:
            if isinstance(idt_threshold, Iterable):
                self.idt_threshold = min(idt_Rlos * threshold_expand_factor * idt_threshold[self.circ] / idt_threshold[self.circ - 1], idt_threshold)
            else:
                self.idt_threshold = min(idt_Rlos * threshold_expand_factor, idt_threshold)
            
            if isinstance(lfs_threshold, Iterable):
                self.lfs_threshold = min(lfs_Rlos * threshold_expand_factor * lfs_threshold[self.circ] / lfs_threshold[self.circ - 1], lfs_threshold)
            else:
                self.lfs_threshold = min(lfs_Rlos * threshold_expand_factor, lfs_threshold)

            if isinstance(psrex_threshold, Iterable):
                self.psrex_threshold = min(psrex_Rlos * threshold_expand_factor * psrex_threshold[self.circ] / psrex_threshold[self.circ - 1], psrex_threshold)
                self.psrex_threshold = max([min(self.psrex_threshold, psrex_threshold), 2])
            else:
                self.psrex_threshold = max([min(psrex_Rlos * threshold_expand_factor, psrex_threshold), 2])
            if isinstance(psr_concentration_threshold, Iterable):
                self.psr_concentration_threshold = min(psr_concentration_Rlos * threshold_expand_factor * psr_concentration_threshold[self.circ] / psr_concentration_threshold[self.circ - 1], psr_concentration_threshold)
            else:
                self.psr_concentration_threshold = min(psr_concentration_Rlos * threshold_expand_factor, psr_concentration_threshold)
        else:
            idt_threshold = np.array(idt_threshold)[0] if isinstance(idt_threshold, Iterable) else idt_threshold
            lfs_threshold = np.array(lfs_threshold)[0] if isinstance(lfs_threshold, Iterable) else lfs_threshold
            psrex_threshold = np.array(psrex_threshold)[0] if isinstance(psrex_threshold, Iterable) else psrex_threshold 
            self.idt_threshold = min(idt_Rlos * threshold_expand_factor, idt_threshold)
            self.lfs_threshold = min(lfs_Rlos * threshold_expand_factor, lfs_threshold)
            self.psrex_threshold = max(min(psrex_Rlos * threshold_expand_factor, psrex_threshold), 1.5)
            self.psr_concentration_threshold = max(min(psr_concentration_Rlos * threshold_expand_factor, psr_concentration_threshold), 0.01)

        self.GenAPARTDataLogger.info(f"Current CIRC thresholds are fixed as {self.idt_threshold} ~ {self.lfs_threshold} ~ {self.psrex_threshold} ~ {self.psr_concentration_threshold}")
        self.GenAPARTDataLogger.info("=" * 50)
        # self.WriteCurrentAPART_args(idt_threshold = self.idt_threshold, lfs_threshold = self.lfs_threshold, psrex_threshold = self.psrex_threshold)


    def SortALIST(self, w1 = None, w2 = None, w3 = None, w4 = None, apart_data_path = None, experiment_time = 50, cluster_ratio = False, 
                  IDT_reduced_threshold = None, SortALIST_T_threshold:float = None, SortALIST_T_threshold_ratio:list|float = None,
                  need_idt = False, need_psr = False, need_lfs = False, return_all = False, father_sample_save_path = None,
                  logger = None, ord = np.inf, PSRex_cutdown = None,**kwargs) -> np.ndarray:
        """
        从 apart_data.npz 中筛选出来最接近真实 IDT 的采样点，以反问题权重作为 IDT 和 PSR 的筛选权重
        增加一个限制： 筛选出的结果到真实值的距离不能比 Reduced 结果差大于 IDT_reduced_threshold 和 PSR_reduced_threshold 倍
        1. 以此作为反问题初值; 2. 用于 ASample 筛选
        params:
            w1, w2, w3, w4: float, 可选，IDT, PSR, LFS, PSR concentration 的权重
            apart_data_path: 输入 apart_data.npz
            experiment_time: 最后返回的列表大小
            cluster_ratio: 是否使用聚类模式; 如果为 int 类型，则表示聚类中初始点的数量
            need_idt: 如果需要筛选出的 Alist 对应的 apart_data， 将此键值改为 True
            father_sample_save_path: 保存 father_sample 的路径
            SortALIST_T_threshold: 筛选出的结果中，温度误差不能高于 SortALIST_T_threshold
            SortALIST_T_threshold_ratio: 筛选出的结果中，
                        温度误差不能高于 SortALIST_T_threshold_ratio * self.Temperature_Diff; 优先级高于 SortALIST_T_threshold
            psr_mean; psr_std
            need_idt: 如果需要筛选出的 Alist 对应的 apart_data， 将此键值改为 True
            return_all: 是否返回所有筛选出的 Alist
                return (Alist, idt, psr, psrex, T, lfs, psr_concentration) if True
        """
        if w1 is None: w1 = self.IDT_weight
        if w2 is None: w2 = self.PSR_weight
        if w3 is None: w3 = self.LFS_weight
        if w4 is None: w4 = self.PSR_concentration_weight
        if apart_data_path is None: apart_data_path = self.apart_data_path
        if logger is None: logger = self.GenAPARTDataLogger
        if PSRex_cutdown is None: PSRex_cutdown = self.PSRex_cutdown

        apart_data = np.load(apart_data_path)
        apart_data_idt = apart_data['all_idt_data']; Alist = apart_data['Alist']; 
        apart_data_psr = apart_data['all_psr_data']; apart_data_T = apart_data['all_T_data']
        apart_data_psrex = apart_data['all_psr_extinction_data']; apart_data_lfs = apart_data['all_lfs_data']
        apart_data_psr_concentration = apart_data['all_psr_concentration_data']
        
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

        # 计算 diff_psrex 并筛选出其值最小的样本集合
        if PSRex_cutdown:
            apart_data_psrex_log2 = np.maximum(np.log2(apart_data_psrex), np.log2(self.true_extinction_time))
            diff_psrex = np.abs(apart_data_psrex_log2 - np.log2(self.true_extinction_time))
            ## 筛选出 diff_psrex 最小的样本集合
            if_idt_pass *= np.all(diff_psrex <= np.amin(diff_psrex, axis = 0) + 1, axis = 1)
            if if_idt_pass.sum() == 0:
                raise IndexError(
                    f"No sample pass the psrex filter! apart_data_psrex = {apart_data_psrex} and true_extinction_time = {np.log2(self.true_extinction_time)}"
                    )    
            diff_psrex = np.linalg.norm(apart_data_psrex_log2 - np.log2(self.true_extinction_time), axis = 1, ord = ord)
        else:
            diff_psrex = PSRex_shrink_error(np.log2(apart_data_psrex), np.log2(self.true_extinction_time), ord = ord, return_Rlos=True)
            ## 筛选出 diff_psrex 最小的样本集合
            if_idt_pass *= np.all(diff_psrex <= np.amin(diff_psrex, axis = 0) + 1, axis = 1)
            if if_idt_pass.sum() == 0: 
                raise IndexError(
                    f"No sample pass the psrex filter! apart_data_psrex = {apart_data_psrex} and true_extinction_time = {np.log2(self.true_extinction_time)}"
                    )            
            diff_psrex = PSRex_shrink_error(np.log2(apart_data_psrex), np.log2(self.true_extinction_time), ord = ord)

        # 筛选出的结果温度误差不能高于 SortALIST_T_threshold
        if not SortALIST_T_threshold is None:
            tmp_if_idt_pass = if_idt_pass * np.all(np.abs(self.true_T_data - apart_data_T) <= SortALIST_T_threshold, axis = 1)
            # 若 tmp_if_idt_pass 全为 False， 则跳过这一步骤
            if np.any(tmp_if_idt_pass):
                if_idt_pass = tmp_if_idt_pass
            else:
                logger.warning(f'All data is filtered out by SortALIST_T_threshold ratio: {SortALIST_T_threshold_ratio} and SortALIST_T_threshold: {SortALIST_T_threshold},'
                                        + f'But the np.abs(self.true_T_data - apart_data_T) is {np.abs(self.true_T_data - apart_data_T[0])} so skip this step!')
                
        # 筛选出的结果不能比 Reduced 结果差大于 IDT_reduced_threshold 倍
        if not IDT_reduced_threshold is None:
            if_idt_pass *= np.all(np.abs(true_idt - np.log10(apart_data_idt)) <= IDT_reduced_threshold * np.abs(reduced_idt - true_idt), axis = 1)
            if if_idt_pass.sum() == 0: logger.warning(
                f"No sample pass the IDT_reduced_threshold filter! apart_data - true = {np.mean(np.abs(true_idt - np.log10(apart_data_idt)))} and true - reduce = {np.abs(reduced_idt - true_idt)}"
                )

        # 计算 DIFF_idt 和 DIFF_psr
        diff_idt = np.linalg.norm(
            w1 * (np.log10(apart_data_idt) - true_idt), 
            axis = 1, ord = ord)
        diff_lfs = np.linalg.norm(
            w3 * (apart_data_lfs - self.true_lfs_data), 
            axis = 1, ord = ord)
        diff_psr_concentration = np.linalg.norm(
            w4 * (apart_data_psr_concentration - self.true_psr_concentration_data), 
            axis = 1, ord = ord)
        
        diff_idt = diff_idt[if_idt_pass]; diff_psrex = diff_psrex[if_idt_pass]; diff_lfs = diff_lfs[if_idt_pass]; diff_psr_concentration = diff_psr_concentration[if_idt_pass]
        
        Alist = Alist[if_idt_pass, :]; apart_data_idt = apart_data_idt[if_idt_pass, :]
        apart_data_psr = apart_data_psr[if_idt_pass, :]; apart_data_psrex = apart_data_psrex[if_idt_pass, :]
        apart_data_lfs = apart_data_lfs[if_idt_pass, :]; apart_data_T = apart_data_T[if_idt_pass, :]
        apart_data_psr_concentration = apart_data_psr_concentration[if_idt_pass, :]

        diff = diff_idt + diff_psrex * w2[0] + diff_lfs + diff_psr_concentration
        index = np.argsort(diff); Alist = Alist[index,:]
        apart_data_idt = apart_data_idt[index,:]; apart_data_psr = apart_data_psr[index,:]; apart_data_psrex = apart_data_psrex[index,:]
        apart_data_T = apart_data_T[index,:]; apart_data_lfs = apart_data_lfs[index,:]
        apart_data_psr_concentration = apart_data_psr_concentration[index,:]

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
           np.savez(father_sample_save_path, Alist = Alist[0:experiment_time, :], IDT = apart_data_idt[0:experiment_time, :], PSR = apart_data_psr[0:experiment_time, :]) 
        
        return_tuple = (
            Alist[0:experiment_time, :], 
        )
        if need_idt:
            return_tuple += (apart_data_idt[0:experiment_time, :],)
        if need_psr:
            return_tuple += (apart_data_psr[0:experiment_time, :],)
        if need_lfs:
            return_tuple += (apart_data_lfs[0:experiment_time, :],)
        if return_all:
            return_tuple = (
                Alist[0:experiment_time, :], 
                apart_data_idt[0:experiment_time, :],
                apart_data_psr[0:experiment_time, :],
                apart_data_psrex[0:experiment_time, :],
                # apart_data_T[0:experiment_time, :],
                apart_data_lfs[0:experiment_time, :],
                apart_data_psr_concentration[0:experiment_time, :]
            )
        
        return return_tuple if len(return_tuple) > 1 else return_tuple[0]
        

    def ASample(self, sample_size = None, coreAlist = None, IDT_weight = None, PSRex_weight = None, LFS_weight = None, PSR_concentration_weight = None,
                passing_rate_upper_limit = None, if_LFS_filter:bool = True, if_PSRex_filter:bool = True,
                IDT_reduced_threshold = None, PSR_reduced_threshold = None, shrink_delta = None,
                cluster_ratio = False, father_sample_save_path = None, start_circ = 0, 
                shrink_ratio = None, average_timeout_time = 0.072, sampling_expand_factor = None, **kwargs):
        """
        生成 ASample 的采样点; 采样点的生成方式为:
            1. 生成均匀采样点;
            2. 根据 IDT_weight, PSRex_weight, LFS_weight, PSR_concentration_weight 对均匀采样点进行加权;
            3. 根据 apart_data.npz 中的 IDT, PSR, LFS
        进行筛选，筛选出最接近真实值的采样点;
        params:

        """
        np.set_printoptions(precision = 2, suppress = True); t0 = time.time()

        # 预设置
        self.GenAPARTDataLogger.info(f"Start The ASample Process; Here we apply three aspect into consideration: IDT: True, LFS: {if_LFS_filter} and PSRextinction: {if_PSRex_filter}")
        # 提取采样的左右界限 + 采样阈值
        if IDT_weight is None: IDT_weight = self.IDT_weight
        if PSRex_weight is None: PSRex_weight = self.PSR_weight
        if LFS_weight is None: LFS_weight = self.LFS_weight
        if PSR_concentration_weight is None: PSR_concentration_weight = self.PSR_concentration_weight

        # 检测类中是否存在 self.idt_threshold, self.lfs_threshold, self.psrex_threshold
        if not hasattr(self, 'idt_threshold') or not hasattr(self, 'lfs_threshold') or not hasattr(self, 'psrex_threshold') or not hasattr(self, 'psr_concentration_threshold'):
            idt_threshold = self.APART_args['idt_threshold']
            lfs_threshold = self.APART_args['lfs_threshold']
            psrex_threshold = self.APART_args['psrex_threshold']
            psr_concentration_threshold = self.APART_args['psr_concentration_threshold']
            self.idt_threshold = np.array(idt_threshold)[self.circ - 1] if isinstance(idt_threshold, Iterable) else idt_threshold
            self.lfs_threshold = np.array(lfs_threshold)[self.circ - 1] if isinstance(lfs_threshold, Iterable) else lfs_threshold
            self.psrex_threshold = np.array(psrex_threshold)[self.circ - 1] if isinstance(psrex_threshold, Iterable) else psrex_threshold  
            self.psr_concentration_threshold = np.array(psr_concentration_threshold)[self.circ - 1] if isinstance(psr_concentration_threshold, Iterable) else psr_concentration_threshold
            
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
        sampling_expand_factor = self.APART_args.get('sampling_expand_factor',2 / threshold_expand_factor) if sampling_expand_factor is None else sampling_expand_factor
        psrex_loss_func = partial(PSRex_shrink_error, ord = 2, return_Rlos = True) if self.PSRex_cutdown else None

        # 将YAML文件的A值提取并构建均匀采样点
        if self.circ == 0 or self.circ == start_circ:
            self.samples = sample_constant_A(sample_size, self.reduced_mech_A0, self.l_alpha, self.r_alpha)
            if shrink_strategy:
                self.boarder_samples = sample_constant_A(int(sample_size * shrink_ratio), self.reduced_mech_A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha),)
        else:
            net = load_best_dnn(Network_PlainTripleHead_PSRconcentration, self.model_previous_json, device = 'cpu')
            psrex_net = load_best_dnn(Network_PlainSingleHead, self.model_previous_json, device = 'cpu', prefix = "PSR_extinction")
            self.GenAPARTDataLogger.info(f"idt_threshold: log10({self.idt_threshold}) = {np.log10(self.idt_threshold)}; lfs_threshold: {self.lfs_threshold}; sample_size: {sample_size}; father sample size: {core_size}")
            self.GenAPARTDataLogger.info(f"psrex_threshold: log2({self.psrex_threshold}) = {np.log2(self.psrex_threshold)}; shrink_delta: {shrink_delta}; shrink_strategy: {shrink_strategy}")
            self.GenAPARTDataLogger.info(f"IDT_reduced_threshold: {IDT_reduced_threshold}; PSR_reduced_threshold: {PSR_reduced_threshold}; passing_rate_upper_limit: {passing_rate_upper_limit}")
            self.GenAPARTDataLogger.info("="*100)

            # 读取之前的 apart_data.npz 从中选择前 1% 的最优采样点作为核心; 依然选择 IDT 作为指标，不涉及 PSR

            if coreAlist is None:
                cluster_weight = kwargs.get('cluster_weight', 0.1)
                previous_coreAlist = self.SortALIST(IDT_weight, PSRex_weight, LFS_weight, PSR_concentration_weight,
                        apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                        experiment_time = core_size, IDT_reduced_threshold = IDT_reduced_threshold, 
                        PSR_reduced_threshold = PSR_reduced_threshold, father_sample_save_path = father_sample_save_path, 
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
                
            t0 = time.time(); idt_zero_father_sample_times = 0; lfs_zero_father_sample_times = 0; psrex_zero_father_sample_times = 0; psr_concentration_zero_father_sample_times = 0
            self.samples = []; self.boarder_samples = []; tmp_sample_size = int(2 * (sample_size) // core_size); tmp_sample = []  
            while len(self.samples) < sample_size:
                # 每次采样的样本点不能太少，也不能太多；因此创建自适应调节机制
                if len(tmp_sample) >= sample_size * 0.02:
                    tmp_sample_size = int(sampling_expand_factor * (sample_size - len(self.samples)) // core_size)
                else:
                    tmp_sample_size = int(sampling_expand_factor * sample_size // core_size)
                for A0 in coreAlist:
                    try:  
                        tmp_sample, idt_pred_data = SampleAWithNet(net.forward_Net1, np.log10(self.true_idt_data), threshold = np.log10(self.idt_threshold), 
                                                    size = tmp_sample_size, A0 = A0, l_alpha = self.l_alpha, passing_rate_upper_limit = np.sqrt(passing_rate_upper_limit),
                                                    r_alpha = self.r_alpha, save_path = None, debug = True, reduced_data = np.log10(self.reduced_idt_data), reduced_threshold = None, 
                                                    reduced_mech_A0 = self.reduced_mech_A0, max_uncertainty_range = self.max_uncertainty_range)
                        if shrink_strategy:
                            tmp_boarder_sample = SampleAWithNet(net.forward_Net1, np.log10(self.true_idt_data), threshold = np.log10(self.idt_threshold) + 1, 
                                                    size = int(tmp_sample_size * shrink_ratio), A0 = A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                    r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha), 
                                                    reduced_data = np.log10(self.reduced_idt_data), reduced_threshold = None, debug = False, 
                                                    reduced_mech_A0 = self.reduced_mech_A0, max_uncertainty_range = self.max_uncertainty_range)
                        
                        idt_Rlos = np.abs(np.mean(idt_pred_data - np.log10(self.true_idt_data), axis = 0))
                        self.GenAPARTDataLogger.info(f"IDT: On Average, Working Condition Index which not satisfy threshold IS {np.where(idt_Rlos > np.log10(self.idt_threshold))[0]}")
                        # self.GenAPARTDataLogger.info(f"IDT: On Average, Working Condition which not satisfy threshold IS {self.IDT_condition[np.where(idt_Rlos > np.log10(self.idt_threshold))[0],:]}")
                        self.GenAPARTDataLogger.info(f"true IDT is {np.log10(self.true_idt_data)}")
                        self.GenAPARTDataLogger.info(f"First element of sample prediction IDT is {idt_pred_data[0]}")
                        self.GenAPARTDataLogger.info(f"Abs mean Difference between np.log10(self.true_idt_data) and idt_pred_data[-1] is {np.abs(np.log10(self.true_idt_data) - idt_pred_data[-1])}")
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
                    if if_PSRex_filter:
                        try:
                            tmp_sample, psrex_pred_data = SampleAWithNet(psrex_net, np.log2(self.true_extinction_time), threshold = np.log2(self.psrex_threshold), 
                                                        father_samples = tmp_sample, loss_func = psrex_loss_func, passing_rate_upper_limit = 1, debug = True,
                                                        reduced_mech_A0 = self.reduced_mech_A0, max_uncertainty_range = self.max_uncertainty_range)
                            if shrink_strategy:
                                tmp_boarder_sample = SampleAWithNet(psrex_net, np.log2(self.true_extinction_time), threshold = np.log2(self.psrex_threshold) + 1, 
                                                        father_samples = tmp_boarder_sample, loss_func = psrex_loss_func, debug = False,
                                                        reduced_mech_A0 = self.reduced_mech_A0, max_uncertainty_range = self.max_uncertainty_range)
                            psrex_Rlos = np.abs(np.mean(psrex_pred_data - np.log2(self.true_extinction_time), axis = 0))
                            self.GenAPARTDataLogger.info(f"PSRex: On Average, Working Condition Index which not satisfy threshold IS {np.where(psrex_Rlos > np.log2(self.true_extinction_time),)[0]}")
                            # self.GenAPARTDataLogger.info(f"PSR: On Average, Working Condition which not satisfy threshold IS {np.unique(np.tile(self.PSR_condition, (1,3))[np.where(lfs_Rlos > true_psr)[0],:], axis = 0)}")
                            self.GenAPARTDataLogger.info(f"true PSRex is {np.log2(self.true_extinction_time)}")
                            self.GenAPARTDataLogger.info(f"First element of sample prediction PSRex is {psrex_pred_data[0]}")
                            self.GenAPARTDataLogger.info(f"Abs mean Difference between true PSRex and sample prediction is {psrex_Rlos}")
                            self.GenAPARTDataLogger.info(f"IDT + PSRex: Sampled {len(tmp_sample)} in this iter, pass Rate is {len(tmp_sample) / tmp_sample_size * 100} %, total size is {len(self.samples) + len(tmp_sample)}! cost {time.time() - t0:.2f}s; Memory usage: {psutil.Process(os.getpid()).memory_info().rss:.2e} B")  
                    
                            self.GenAPARTDataLogger.info("——"*100)
                            if len(tmp_sample) == 0:
                                raise ValueError("tmp_sample is Empty!")
                        except ValueError or IndexError:
                            self.GenAPARTDataLogger.warning(f"PSRex: tmp_sample is Empty in This circle! psrex_zero_father_sample_times = {psrex_zero_father_sample_times}")
                            self.GenAPARTDataLogger.info("——"*100)
                            psrex_zero_father_sample_times += 1
                            continue
                    if if_LFS_filter:
                        try:
                            tmp_sample, lfs_pred_data = SampleAWithNet(net.forward_Net2, self.true_lfs_data, threshold = self.lfs_threshold, 
                                                        father_samples = tmp_sample, passing_rate_upper_limit = np.sqrt(passing_rate_upper_limit), debug = True,
                                                        reduced_data = self.reduced_lfs_data, reduced_threshold = None, reduced_mech_A0 = self.reduced_mech_A0, max_uncertainty_range = self.max_uncertainty_range
                                                        )
                            if shrink_strategy:
                                tmp_boarder_sample = SampleAWithNet(net.forward_Net2, self.true_lfs_data, threshold = self.lfs_threshold + 1, 
                                                        father_samples = tmp_boarder_sample, debug = False,reduced_mech_A0 = self.reduced_mech_A0, max_uncertainty_range = self.max_uncertainty_range
                                                    
                                                          )    
                                self.boarder_samples.extend(tmp_boarder_sample.tolist())                  
                            lfs_Rlos = np.abs(np.mean(lfs_pred_data - self.true_lfs_data, axis = 0))
                            self.GenAPARTDataLogger.info(f"lfs: On Average, Working Condition Index which not satisfy threshold IS {np.where(lfs_Rlos > self.true_lfs_data,)[0]}")
                            # self.GenAPARTDataLogger.info(f"lfs: On Average, Working Condition which not satisfy threshold IS {np.unique(np.tile(self.lfs_condition, (1,3))[np.where(lfs_Rlos > true_lfs)[0],:], axis = 0)}")
                            self.GenAPARTDataLogger.info(f"true lfs is {self.true_lfs_data,}")
                            self.GenAPARTDataLogger.info(f"First element of sample prediction lfs is {lfs_pred_data[0]}")
                            self.GenAPARTDataLogger.info(f"Abs Difference between true lfs and sample prediction is {lfs_Rlos}")
                            self.GenAPARTDataLogger.info(f"IDT + PSRex + LFS: Remained {len(tmp_sample)} in this iter, pass Rate is {len(tmp_sample) / tmp_sample_size * 100} %, total size is {len(self.samples)}! cost {time.time() - t0:.2f}s; Memory usage: {psutil.Process(os.getpid()).memory_info().rss:.2e} B")  
                            
                            self.GenAPARTDataLogger.info("-·"*100)
                            if len(tmp_sample) == 0:
                                raise ValueError("tmp_sample is Empty!")

                        except ValueError or IndexError:
                            self.GenAPARTDataLogger.warning(f"lfs: tmp_sample is Empty in This circle! lfs_zero_father_sample_times = {lfs_zero_father_sample_times}")
                            self.GenAPARTDataLogger.info("-·"*100)
                            lfs_zero_father_sample_times += 1
                            continue
                    
                    try:
                        tmp_sample, psr_concentration_pred_data = SampleAWithNet(net.forward_Net3, self.true_psr_concentration_data, threshold = self.psr_concentration_threshold,
                                                        father_samples = tmp_sample, passing_rate_upper_limit = np.sqrt(passing_rate_upper_limit), debug = True,
                                                        reduced_data = self.reduced_psr_concentration_data, reduced_threshold = None, reduced_mech_A0 = self.reduced_mech_A0, max_uncertainty_range = self.max_uncertainty_range
                                                        )
                        if shrink_strategy:
                            tmp_boarder_sample = SampleAWithNet(net.forward_Net3, self.true_psr_concentration_data, threshold = self.psr_concentration_threshold + 1, 
                                                        father_samples = tmp_boarder_sample, debug = False,
                                                        )
                            self.boarder_samples.extend(tmp_boarder_sample.tolist())
                        psr_concentration_Rlos = np.abs(np.mean(psr_concentration_pred_data - self.true_psr_concentration_data, axis = 0))
                        self.GenAPARTDataLogger.info(f"PSR_concentration: On Average, Working Condition Index which not satisfy threshold IS {np.where(psr_concentration_Rlos > self.true_psr_concentration_data,)[0]}")
                        # self.GenAPARTDataLogger.info(f"PSR_concentration: On Average, Working Condition which not satisfy threshold IS {np.unique(np.tile(self.PSR_concentration_condition, (1,3))[np.where(psr_concentration_Rlos > true_psr_concentration)[0],:], axis = 0)}")
                        self.GenAPARTDataLogger.info(f"true PSR_concentration is {self.true_psr_concentration_data,}")
                        self.GenAPARTDataLogger.info(f"First element of sample prediction PSR_concentration is {psr_concentration_pred_data[0]}")
                        self.GenAPARTDataLogger.info(f"Abs Difference between true PSR_concentration and sample prediction is {psr_concentration_Rlos}")
                        self.GenAPARTDataLogger.info(f"IDT + PSRex + LFS + PSR_concentration: Remained {len(tmp_sample)} in this iter, pass Rate is {len(tmp_sample) / tmp_sample_size * 100} %, total size is {len(self.samples)}! cost {time.time() - t0:.2f}s; Memory usage: {psutil.Process(os.getpid()).memory_info().rss:.2e} B")
                        
                        self.GenAPARTDataLogger.info("-·"*100)
                        if len(tmp_sample) == 0:
                            raise ValueError("tmp_sample is Empty!")
                    except ValueError or IndexError:
                        self.GenAPARTDataLogger.warning(f"PSR_concentration: tmp_sample is Empty in This circle!")
                        self.GenAPARTDataLogger.info("-·"*100)
                        psr_concentration_zero_father_sample_times += 1
                        continue
                    

                    self.samples.extend(tmp_sample.tolist())
                    pass_rate = len(tmp_sample) / tmp_sample_size
                    # pass_rate 太高调小 idt_threshold 和 psrex_threshold
                    if pass_rate ** 1/3 > passing_rate_upper_limit * 0.9 and pass_rate  ** 1/3 > 0.5:
                        self.idt_threshold = self.idt_threshold * 0.8
                        self.psrex_threshold = self.psrex_threshold * 0.8
                        self.lfs_threshold = self.lfs_threshold * 0.8
                        self.GenAPARTDataLogger.warning(f"IDT + PSRex + LFS: pass_rate is {pass_rate * 100} %, which is too high! idt_threshold have changed to log10({self.idt_threshold}) = {np.log10(self.idt_threshold)}; psrex_threshold have changed to log2({self.psrex_threshold}) = {np.log2(self.psrex_threshold)}")
                    
                    # 样本太多立即退出
                    if len(self.samples) >= sample_size * 1.2:
                        self.GenAPARTDataLogger.warning(f"IDT + PSRex + LFS: Stop the Sampling Process, Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                        self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                        break

                # 时间过长退出机制
                if time.time() - t0 > sample_size * average_timeout_time:
                    self.GenAPARTDataLogger.warning("Error: Function Asample has timed out!")
                    break
                self.GenAPARTDataLogger.info(f"In this Iteration, IDT + PSRex + LFS: Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                # 如果过长时间无法筛选到合适的点，稍微拓宽采样的 threshold
                if idt_zero_father_sample_times > 10 * len(coreAlist):
                    self.idt_threshold = 10 ** self.idt_threshold
                    self.GenAPARTDataLogger.warning(f"IDT: idt_threshold have changed to log10{self.idt_threshold} = {np.log10(self.idt_threshold)}")
                    idt_zero_father_sample_times = 0
                if lfs_zero_father_sample_times > 10* len(coreAlist):
                    self.lfs_threshold = np.amax(lfs_Rlos) * 1.1
                    self.GenAPARTDataLogger.warning(f"LFS: lfs_threshold: {np.amin(self.lfs_threshold)} ~ {np.amax(self.lfs_threshold)};")
                    lfs_zero_father_sample_times = 0 
                if psrex_zero_father_sample_times > 10* len(coreAlist):
                    self.psrex_threshold = 2 * self.psrex_threshold
                    self.GenAPARTDataLogger.warning(f"PSRex: psrex_threshold: log2 {self.psrex_threshold} = {np.log2(self.psrex_threshold)};")
                    psrex_zero_father_sample_times = 0
                if psr_concentration_zero_father_sample_times > 10 * len(coreAlist):
                    self.psr_concentration_threshold = np.amax(psr_concentration_Rlos) * 1.1
                    self.GenAPARTDataLogger.warning(f"PSR_concentration: psr_concentration_threshold: {np.amin(self.psr_concentration_threshold)} ~ {np.amax(self.psr_concentration_threshold)};")
                    psr_concentration_zero_father_sample_times = 0
        
        self.samples = np.array(self.samples)
        self.father_samples = coreAlist
        if len(self.samples) == 0:
            raise ValueError("self.samples is Empty! End the whole process! Please check the threshold!")
        np.save(f'./data/APART_data/Asamples_{self.circ}.npy', self.samples)  
        # 更新 idt_threshold 和 lfs_threshold 和 psrex_threshold
        self.APART_args['idt_threshold'] = self.idt_threshold
        self.APART_args['lfs_threshold'] = self.lfs_threshold
        self.APART_args['psrex_threshold'] = self.psrex_threshold
        self.APART_args['psr_concentration_threshold'] = self.psr_concentration_threshold
        self.APART_args['passing_rate_upper_limit'] = passing_rate_upper_limit
        self.APART_args['sample_size'] = sample_size
        # 保存 APART_args 到 self.current_json_path
        self.WriteCurrentAPART_args(cover = True)
        
        if shrink_strategy:
            self.boarder_samples = np.array(self.boarder_samples)
            # 截取 boarder_samples 前 shrink_ratio * sample_size 个点
            self.boarder_samples = self.boarder_samples[0:min(int(shrink_ratio * sample_size), len(self.boarder_samples))]
            np.save(f'./data/APART_data/Aboarder_samples_{self.circ}.npy', self.boarder_samples) 
        self.GenAPARTDataLogger.info(f"End The ASample Progress! The size of samples is {len(self.samples)}, cost {time.time() - t0:.2f}s")  
        return self.samples
    
        
    def GenDataFuture(self, samples:np.ndarray = None, idt_cut_time_alpha = 1.5, psr_error_tol = None, 
                          start_sample_index = 0, cpu_process = None, ignore_error_path = None, 
                          save_path = "./data/APART_data/tmp", PSRex_exp_factor = None, **kwargs):
        """
        使用 future 模块的 ProcessPoolExecutor 生成数据; 
        params:
            samples: np.ndarray, 用于生成数据的初始点集
            idt_cut_time_alpha: float, 计算 IDT 的 cut time 阈值系数
            psr_error_tol: float, PSR 误差容忍度
            start_sample_index: int, 从 samples 的第几个点开始生成数据
            cpu_process: int, 使用的 cpu 核心数
            ignore_error_path: str, 保存 ignore_error 的路径
            save_path: str, 保存数据的路径
            PSRex_exp_factor: float, PSRex 的缩减指数因子
        """
        self.GenAPARTDataLogger.info(f"Start the GenAPARTData_multiprocess at circ {self.circ}")
        mkdirplus(save_path)
        
        cpu_process = self.APART_args['cpu_process'] if cpu_process is None else cpu_process
        psr_error_tol = self.APART_args.get("psr_error_tol", 50) if psr_error_tol is None else psr_error_tol
        expected_sample_size = self.APART_args['sample_size'] if isinstance(self.APART_args['sample_size'], int) else self.APART_args['sample_size'][self.circ]
        PSRex_exp_factor = self.PSRex_decay_rate if PSRex_exp_factor is None else PSRex_exp_factor
        
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
                            GenOneDataIDT_LFS_PSRex_concentration,
                            index = index + start_sample_index,
                            IDT_condition = self.IDT_condition,
                            PSR_condition = self.PSR_condition,
                            LFS_condition = self.LFS_condition,
                            RES_TIME_LIST = self.RES_TIME_LIST,
                            init_res_time = self.init_res_time,
                            Alist = samples[index],
                            eq_dict = self.eq_dict,
                            IDT_fuel = self.IDT_fuel,
                            IDT_oxidizer = self.IDT_oxidizer,
                            PSR_fuel = self.PSR_fuel,
                            PSR_oxidizer = self.PSR_oxidizer,
                            LFS_fuel = self.LFS_fuel,
                            LFS_oxidizer = self.LFS_oxidizer,
                            PSR_concentration_kwargs = self.PSR_concentration_kwargs,
                            reduced_mech = self.reduced_mech,
                            my_logger = self.GenAPARTDataLogger,
                            IDT_mode = self.IDT_mode,
                            # 将 idt_arrays 采取如下设置: 其每个分量是 true_idt_data 和 reduced_idt_data 对应分量中大的那个
                            idt_arrays = np.maximum(self.true_idt_data, self.reduced_idt_data),
                            cut_time_alpha = idt_cut_time_alpha,
                            psr_error_tol = psr_error_tol,
                            exp_factor = PSRex_exp_factor,
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
                              PSRex_cutdown = None, rate = 0.8, device = 'cuda', 
                              one_data_save_path = "./data/APART_data/tmp", 
                              net_data_dirpath = "./data/APART_data/ANET_data", **kwargs) -> None:
        """
        基本上和 APART 中的对应函数功能相同，增加了将 0D 末态温度同时也作为训练网络数据集的一部分的功能
        """
        # gather_data 部分
        aboarder_data_path = os.path.dirname(self.apart_data_path) + f"/Aboarder_apart_data_circ={self.circ}.npz"
        aboarder_one_data_path = os.path.dirname(one_data_save_path) + f"/boarder_tmp"
        if not os.path.exists(self.apart_data_path) or os.path.getsize(self.apart_data_path) /1024 /1024 <= 1:
            self.gather_apart_data(save_path = one_data_save_path, rm_tree = rm_tree, PSR_extinction_mode = True, **kwargs)
        if (shrink_strategy and os.path.exists(aboarder_one_data_path)) or \
            (os.path.exists(aboarder_data_path) and os.path.getsize(aboarder_data_path) /1024 /1024 > 1):
            load_file_name = [
                self.apart_data_path,
                aboarder_data_path,
            ]
            if not os.path.exists(aboarder_data_path) or os.path.getsize(aboarder_data_path) /1024 /1024 <= 1:
                self.gather_apart_data(
                    save_path = aboarder_one_data_path, rm_tree = rm_tree, save_file_name = aboarder_data_path
                    , PSR_extinction_mode = True, **kwargs
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
            all_T_data = Data['all_T_data']; all_psr_data = Data['all_psr_data'] 
            all_psr_extinction_data = Data['all_psr_extinction_data']
            all_lfs_data = Data['all_lfs_data']
            all_psr_concentration_data = Data['all_psr_concentration_data']
            # 判断是否加载之前的 data:
        else:
            all_idt_data, all_T_data, Alist_data, all_psr_data, all_lfs_data, all_psr_concentration_data \
                = [], [], [], [], [], []
            all_psr_extinction_data = []
            for file in load_file_name:   
                Data = np.load(file)
                Alist_data.extend(Data['Alist']); all_idt_data.extend(Data['all_idt_data']); 
                all_T_data.extend(Data['all_T_data']); all_psr_data.extend(Data['all_psr_data'])
                all_psr_extinction_data.extend(Data['all_psr_extinction_data'])
                all_lfs_data.extend(Data['all_lfs_data']); all_psr_concentration_data.extend(Data['all_psr_concentration_data'])
                
        
        # 进入抽取模式，抽取之前数据集中表现最优秀的前 5% 样本点
        if extract_strategy and self.circ != 0:
            extract_rate = kwargs.get('extract_rate', 0.05)
            self.TrainAnetLogger.info(f'Extracting DNN Data From Previous Best Sample..., extract_rate = {extract_rate}')
            for circ in range(self.circ):
                apart_data_path = os.path.dirname(self.apart_data_path) + f"/apart_data_circ={circ}.npz"
                json_data = read_json_data(os.path.dirname(self.model_current_json) + f"/settings_circ={circ}.json")
                if os.path.exists(apart_data_path) and os.path.getsize(apart_data_path) /1024 /1024 > 1:
                    Alist, IDT, PSR, PSRex, LFS, PSR_concentration = self.SortALIST(
                        apart_data_path = apart_data_path,
                        experiment_time = extract_rate,
                        return_all = True,
                        psr_mean = json_data['psr_mean'],
                        psr_std = json_data['psr_std'],
                        logger = self.TrainAnetLogger
                    )
                    all_idt_data.extend(IDT); Alist_data.extend(Alist)
                    all_psr_data.extend(PSR); all_psr_extinction_data.extend(PSRex)
                    all_lfs_data.extend(LFS); all_psr_concentration_data.extend(PSR_concentration)
                else:
                    self.TrainAnetLogger.warning(f"Extracting: Can't find {apart_data_path} at circ = {circ}, skip")
        
        # 转化为 ndarray 加快转化 tensor 速度 + IDT 对数处理 + PSR_extinction log2 处理
        all_idt_data, all_T_data, all_psr_data, Alist_data, all_psr_extinction_data, all_lfs_data, all_psr_concentration_data = \
             np.log10(all_idt_data), np.array(all_T_data), np.array(all_psr_data), np.array(Alist_data), np.log2(all_psr_extinction_data), np.array(all_lfs_data), np.array(all_psr_concentration_data)
        self.TrainAnetLogger.info(f'shape of all_idt_data is {all_idt_data.shape}, shape of all_T_data is {all_T_data.shape}, shape of all_psr_data is {all_psr_data.shape}, shape of Alist_data is {Alist_data.shape}, shape of all_psr_extinction_data is {all_psr_extinction_data.shape}, shape of all_lfs_data is {all_lfs_data.shape}, shape of all_psr_concentration_data is {all_psr_concentration_data.shape}')
        # 如果样本点 PSRex 比真实值还小，我们乐于见到这样的现象，故而抹平其中的 gap
        if PSRex_cutdown:
            all_psr_extinction_data = np.maximum(all_psr_extinction_data, np.log2(self.true_extinction_time))
        
        # 如果 Alist_data 是 1 维数组，提示错误：检查 Alist 的长度是否一样
        if len(Alist_data.shape) == 1: 
            raise ValueError(f"Alist_data is 1D array, check the length of Alist_data is same or not")

        self.TrainAnetLogger.info(f"The size of all_idt_data is {all_idt_data.shape}, all_T_data is {all_T_data.shape}, \n all_psr_data is {all_psr_data.shape}, Alist_data is {Alist_data.shape}, all_psr_extinction_data is {all_psr_extinction_data.shape}, all_lfs_data is {all_lfs_data.shape}")
        # 使用 numpy 剔除 all_psr_extinction_data 中与 self.extinction_time 中相差大于 4 的数据点
        self.TrainAnetLogger.info(f"all_psr_extinction_data is {all_psr_extinction_data}, the first one is {all_psr_extinction_data[0]}, with max {np.max(all_psr_extinction_data)} and min {np.min(all_psr_extinction_data)}")
        self.TrainAnetLogger.info(f"self.true_extinction_time is {np.log2(self.true_extinction_time)}, with max {np.max(np.log2(self.true_extinction_time))} and min {np.min(np.log2(self.true_extinction_time))}")
        
        # 加载 current_json_path 中的参数
        tmp_APART_args = read_json_data(self.model_current_json)
        PSRex_threshold = tmp_APART_args.get("psrex_threshold", 0.5)
        PSRex_threshold = PSRex_threshold[self.circ] if isinstance(PSRex_threshold, Iterable) else PSRex_threshold
        # psr_init_tem = np.repeat(self.PSR_condition, 3, axis = 0)[:, 1]
        if_pass = np.all(np.abs(all_psr_extinction_data - np.log2(self.true_extinction_time)) <= PSRex_threshold + 2, axis = 1)
        # if_pass = np.all(np.abs(all_psr_extinction_data - np.log2(self.true_extinction_time)) <= PSRex_threshold, axis = 1) *\
        #     np.all(np.abs(all_psr_data - psr_init_tem) >= psr_error_tol, axis = 1)
        self.TrainAnetLogger.info(f"DataProcessing: PSRex_threshold: {PSRex_threshold} Passing PSRex: if_pass rate is {np.sum(if_pass) / if_pass.shape[0] * 100} %")
        passed_idt_data, passed_T_data, passed_psr_data, passed_A, passed_lfs_data, passed_psr_extinction_data, passed_psr_concentration_data = \
            all_idt_data[if_pass], all_T_data[if_pass], all_psr_data[if_pass], Alist_data[if_pass], all_lfs_data[if_pass], all_psr_extinction_data[if_pass], all_psr_concentration_data[if_pass]

        # 只保留 sample size 个数据点
        expected_sample_size = self.APART_args['sample_size'] if isinstance(self.APART_args['sample_size'], int) else self.APART_args['sample_size'][self.circ]
        if passed_idt_data.shape[0] > expected_sample_size:
            passed_idt_data, passed_T_data, passed_psr_data, passed_A, passed_lfs_data, passed_psr_extinction_data, passed_psr_concentration_data = \
                passed_idt_data[:expected_sample_size], passed_T_data[:expected_sample_size], \
                passed_psr_data[:expected_sample_size], passed_A[:expected_sample_size], passed_lfs_data[:expected_sample_size], passed_psr_extinction_data[:expected_sample_size], passed_psr_concentration_data[:expected_sample_size]

        assert passed_idt_data.shape[0] > 0 and passed_T_data.shape[0] > 0 and passed_psr_data.shape[0] > 0 and passed_A.shape[0] > 0, \
            f"Error! passed_idt_data.shape = {passed_idt_data.shape}, passed_T_data.shape = {passed_T_data.shape}, passed_psr_data.shape = {passed_psr_data.shape}, passed_A.shape = {passed_A.shape}"
        
        assert passed_lfs_data.shape[0] == passed_idt_data.shape[0] == passed_T_data.shape[0] == passed_psr_data.shape[0] == passed_A.shape[0], \
            f"Error The shape at axis 0 is not same! passed_lfs_data.shape = {passed_lfs_data.shape}, passed_idt_data.shape = {passed_idt_data.shape}, passed_T_data.shape = {passed_T_data.shape}, passed_psr_data.shape = {passed_psr_data.shape}, passed_A.shape = {passed_A.shape}"
        # PSR data, T data 做 zscore 标准化
        passed_psr_data, psr_mean, psr_std = zscore(passed_psr_data, mask_on_threshold = np.amax(self.PSR_condition[:,1]))
        passed_T_data, T_mean, T_std = zscore(passed_T_data)
        self.APART_args.update({'psr_mean': psr_mean.tolist(), "psr_std": psr_std.tolist()})
        self.APART_args.update({'T_mean': T_mean.tolist(), "T_std": T_std.tolist()})

        self.APART_args['input_dim'], self.APART_args['output_dim'] = passed_A.shape[1], [passed_idt_data.shape[1], passed_lfs_data.shape[1], passed_psr_concentration_data.shape[1]]
        self.APART_args['train_size']  = int(passed_A.shape[0] * rate)
        self.APART_args['test_size'] = passed_A.shape[0] - self.APART_args['train_size']

        self.APART_args['PSR_extinction_input_dim'], self.APART_args['PSR_extinction_output_dim'] = \
            Alist_data.shape[1], all_psr_extinction_data.shape[1]
        self.APART_args['PSR_extinction_train_size'] = int(Alist_data.shape[0] * rate)
        self.APART_args['PSR_extinction_test_size'] = Alist_data.shape[0] - self.APART_args['PSR_extinction_train_size']

        self.TrainAnetLogger.info(f"The size of IDT & LFS train set is {self.APART_args['train_size']}; test set is {self.APART_args['test_size']}")
        self.TrainAnetLogger.info(f"The size of PSR extinction train set is {self.APART_args['PSR_extinction_train_size']}; test set is {self.APART_args['PSR_extinction_test_size']}")

        dataset = DATASET_TripleHead(
            data_A = torch.tensor(passed_A, dtype = torch.float32),
            data_QoI1 = torch.tensor(passed_idt_data, dtype = torch.float32),
            data_QoI2 = torch.tensor(passed_lfs_data, dtype = torch.float32),
            data_QoI3= torch.tensor(passed_psr_concentration_data, dtype = torch.float32),
            device = device
        )
        train_data, test_data = random_split(dataset, lengths = [self.APART_args['train_size'], self.APART_args['test_size']])

        self.train_loader = DataLoader(train_data, 
                                       shuffle = True, 
                                       batch_size = self.APART_args['batch_size'],
                                       drop_last = True,)
        
        # 单独写出来 test 的数据
        test_loader = DataLoader(test_data, 
                                shuffle = False, 
                                batch_size = len(test_data),)

        for x_test, idt_test, lfs_test, psr_concentration_test in test_loader:
            self.test_loader = (x_test.to(device), idt_test.to(device), lfs_test.to(device), psr_concentration_test.to(device))
 

        # PSR_extinction 数据集单独列出
        PSR_extinction_dataset = DATASET_SingleHead(
            data_A = torch.tensor(Alist_data, dtype = torch.float32),
            data_QoI = torch.tensor(all_psr_extinction_data, dtype = torch.float32),
            device = device
        )
        PSR_extinction_train_data, PSR_extinction_test_data = random_split( PSR_extinction_dataset
            , lengths = [self.APART_args['PSR_extinction_train_size'], self.APART_args['PSR_extinction_test_size']])
        self.PSR_extinction_train_loader = DataLoader(PSR_extinction_train_data,
                                                        shuffle = True,
                                                        batch_size = self.APART_args['batch_size'],
                                                        drop_last = True)   
        # 单独写出来 test 的数据
        PSR_extinction_test_loader = DataLoader(PSR_extinction_test_data,
                                                shuffle = False,
                                                batch_size = len(PSR_extinction_test_data))
        for x_test, psr_extinction_test in PSR_extinction_test_loader:
            self.PSR_extinction_test_loader = (x_test.to(device), psr_extinction_test.to(device))


        if not net_data_dirpath is None:
            np.savez(
                net_data_dirpath + f"/ANET_Dataset_train={self.APART_args['train_size']}_circ={self.circ}.npz",
                train_data_x = dataset[train_data.indices][0].cpu().numpy(),
                train_data_y = dataset[train_data.indices][1].cpu().numpy(),
                test_data_x =  dataset[test_data.indices][0].cpu().numpy(),
                test_data_y =  dataset[test_data.indices][1].cpu().numpy(),
                PSR_extinction_train_data_x = PSR_extinction_dataset[PSR_extinction_train_data.indices][0].cpu().numpy(),
                PSR_extinction_train_data_y = PSR_extinction_dataset[PSR_extinction_train_data.indices][1].cpu().numpy(),
                PSR_extinction_test_data_x = PSR_extinction_dataset[PSR_extinction_test_data.indices][0].cpu().numpy(),
                PSR_extinction_test_data_y = PSR_extinction_dataset[PSR_extinction_test_data.indices][1].cpu().numpy(),
            )


    def _DeePMO_train(self, device = None, LFS_train_outside_weight = 1, **kwargs):
        
        """
        在原有训练代码的基础上，增加了对 LFS_extinction 的训练。首先进行 IDT / LFS 的训练，
        紧接着再次单独训练一个 LFS extinction 网络
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        model_pth_path  = self.model_path
        model_loss_path = self.model_loss_path

        t0 = time.time()
        input_dim, output_dim = self.APART_args['input_dim'], self.APART_args['output_dim']
        tmp_save_path = mkdirplus(f'{model_pth_path}/tmp')
        self.TrainAnetLogger.info(f'A -> IDT & LFS & Concentration & PSRex training has started...; circ = {self.circ}')
        ANET = Network_PlainTripleHead_PSRconcentration(input_dim, self.APART_args['hidden_units'], output_dim).to(device)
        if self.APART_args['optimizer'] == 'Adam': optimizer = optim.Adam(ANET.parameters(), lr = self.APART_args['learning_rate'])
        if self.APART_args['optimizer'] == 'SGD':  optimizer = optim.SGD(ANET.parameters(), lr = self.APART_args['learning_rate'], momentum = 0.1)
        # 第一次保存初始模型
        state = {'model':ANET.state_dict()}   
        torch.save(state, f'{model_pth_path}/Network_PlainTripleHead_PSRconcentration_initialized_circ={self.circ}.pth')  # 保存DNN模型
        epoch_nums = self.APART_args.get('epoch', 1000)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.APART_args['lr_decay_step'], gamma = self.APART_args['lr_decay_rate'])

        epoch_index, train_his1, test_his1= [], [], []
        for epoch in range(epoch_nums):
            # 预测
            with torch.no_grad():
                x_dnn_test, idt_test, lfs_test, psr_concentration_test = self.test_loader
                idt_pred, lfs_pred = ANET.forward_Net1(x_dnn_test), ANET.forward_Net2(x_dnn_test)
                concentration_pred = ANET.forward_Net3(x_dnn_test)
                loss1 = criterion(idt_pred, idt_test); loss2 = criterion(lfs_pred, lfs_test); loss3 = criterion(concentration_pred, psr_concentration_test)
                test_loss = [float(loss1.cpu().detach()), float(loss2.cpu().detach()), float(loss3.cpu().detach())]
            # 训练
            idt_loss, lfs_loss = 0, 0; concentration_loss = 0
            for _, (x_train_batch, idt_train_batch, lfs_train_batch, concentration_batch) in enumerate(self.train_loader):  # 按照 batch 进行训练
                # x_train_batch, idt_train_batch, lfs_train_batch = \
                #     x_train_batch.to(device), idt_train_batch.to(device), lfs_train_batch.to(device)
                idt_pred, lfs_pred = ANET.forward_Net1(x_train_batch), ANET.forward_Net2(x_train_batch)
                concentration_pred = ANET.forward_Net3(x_train_batch)
                loss1 = criterion(idt_pred, idt_train_batch); loss2 = criterion(lfs_pred, lfs_train_batch)
                loss3 = criterion(concentration_pred, concentration_batch)

                train_loss_batch = loss1 / loss1.detach() + loss2 / loss2.detach() * LFS_train_outside_weight + loss3 / loss3.detach()
                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()
                idt_loss += float(loss1.detach()); lfs_loss += float(loss2.detach()); concentration_loss += float(loss3.detach())
            
            scheduler.step()
            batch_num = len(self.train_loader)
            idt_loss /=  batch_num; lfs_loss /=  batch_num; concentration_loss /= batch_num
            train_his1.append([idt_loss, lfs_loss, concentration_loss]); test_his1.append(test_loss)
            epoch_index.append(epoch);           

            if epoch % 5 == 0:
                GPUtil.showUtilization()
                self.TrainAnetLogger.info(f"epoch: {epoch}\t train loss: {idt_loss:.3e} + {lfs_loss:.3e} + {concentration_loss:.3e},"+
                                            f"test loss: {test_his1[-1][0]:.3e} + {test_his1[-1][1]:.3e} + {test_his1[-1][2]:.3e}, " +
                                            f"time cost: {int(time.time()-t0)} s   lr:{optimizer.param_groups[0]['lr']:.2e}")
            if (epoch == 0) or ((epoch - 25) % 50 == 0):
                
                state = {'model':ANET.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}   
                torch.save(state, f'{model_pth_path}/tmp/ANET_epoch_{epoch}.pth')  # 保存DNN模型   

        
        # 构建 early stopping 必需的文件夹, 注意这里的 early stopping 是以 IDT 作为基准的，因为相比其他两个指标 IDT 更加重要
        if not os.path.exists(f'{model_pth_path}/early_stopping'): # 创建临时保存网络参数的文件夹
            os.makedirs(f'{model_pth_path}/early_stopping', exist_ok=   True)
        train_his1, test_his1 = np.array(train_his1), np.array(test_his1)

        # early stopping 平均 earlystopping_step 的误差求最小
        earlystopping_step = min(epoch_nums, 50)

        test_loss_sum = np.sum(test_his1.reshape(-1, earlystopping_step, 3)[...,0], axis = 1)
        
        stop_index = int(earlystopping_step * np.argmin(test_loss_sum) + earlystopping_step / 2)
        print(earlystopping_step, test_loss_sum, stop_index, test_his1.shape, test_his1.reshape(-1, earlystopping_step, 3).shape)
        best_pth_path = f'{model_pth_path}/early_stopping/model_best_stopat_{stop_index}_circ={self.circ}.pth'
        self.APART_args['best_ppth'] = best_pth_path
        shutil.copy(f'{model_pth_path}/tmp/ANET_epoch_{stop_index}.pth', best_pth_path)
        try:
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

        t1 = time.time()
        self.TrainAnetLogger.info(f'A -> PSR extinction time training has started...; circ = {self.circ}')
        self.TrainAnetLogger.info(f"ANET_train_IDT_PSR: total time cost: {int(t1-t0)} s")

        tmp_save_path = mkdirplus(f'{model_pth_path}/tmp')
        input_dim, output_dim = self.APART_args['PSR_extinction_input_dim'], self.APART_args['PSR_extinction_output_dim']
        hidden_units = self.APART_args.get('PSR_extinction_hidden_units', self.APART_args['hidden_units'])
        ANET = Network_PlainSingleHead(input_dim, hidden_units, output_dim).to(device)
        optimizer = torch.optim.Adam(ANET.parameters(), lr = self.APART_args['learning_rate'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.APART_args['lr_decay_step'], gamma = self.APART_args['lr_decay_rate'])
        epoch_nums = self.APART_args.get('PSRex_epoch', self.APART_args['epoch'])
        # 第一次保存初始模型
        state = {'model':ANET.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':0}
        torch.save(state, f'{tmp_save_path}/Network_PlainSingleHead_epoch_{0}.pth')  # 保存DNN模型
        # 开始训练
        train_his2, test_his2 = [], []; criterion = nn.MSELoss()
        epoch_index = []
        for epoch in range(epoch_nums):
            # 预测
            with torch.no_grad():
                x_dnn_test, psr_extinction_test = self.PSR_extinction_test_loader
                psr_extinction_pred = ANET.forward(x_dnn_test)
                loss = criterion(psr_extinction_pred, psr_extinction_test)
                test_loss = loss.item()
            # 训练
            train_loss = 0
            for _, (x_train_batch, psr_extinction_train_batch) in enumerate(self.PSR_extinction_train_loader):  # 按照 batch 进行训练
                # x_train_batch, psr_extinction_train_batch = x_train_batch.to(device), psr_extinction_train_batch.to(device)
                psr_extinction_train_batch_pred = ANET.forward(x_train_batch)
                train_loss_batch = criterion(psr_extinction_train_batch_pred, psr_extinction_train_batch)
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
                torch.save(state, f'{tmp_save_path}/Network_PlainSingleHead_epoch_{epoch}.pth')  # 保存DNN模型  

        # 构建 early stopping 必需的文件夹, 注意这里的 early stopping 是以 IDT 作为基准的，因为相比其他两个指标 IDT 更加重要
        early_stopping_file = mkdirplus(f'{model_pth_path}/early_stopping')
        train_his2, test_his2 = np.array(train_his2), np.array(test_his2)

        # early stopping 平均 earlystopping_step 的误差求最小
        earlystopping_step = min(epoch_nums, 50)

        test_loss_sum = np.sum(test_his2.reshape(-1, earlystopping_step, 1)[...,0], axis = 1)
        stop_index = int(earlystopping_step * np.argmin(test_loss_sum) + earlystopping_step / 2)
        PSR_extinction_best_pth_path = f'{early_stopping_file}/PSR_extinction_best_stopat_{stop_index}_circ={self.circ}.pth'
        self.APART_args['PSR_extinction_best_ppth'] = PSR_extinction_best_pth_path
        shutil.copy(f'{tmp_save_path}/Network_PlainSingleHead_epoch_{stop_index}.pth', PSR_extinction_best_pth_path)
        try:
            shutil.rmtree(tmp_save_path, ignore_errors=True); mkdirplus(tmp_save_path) # 删除中间文件 
        except:
            self.TrainAnetLogger.warning(f"copy file error, {tmp_save_path}/Network_PlainSingleHead_epoch_{stop_index}.pth not exist")

        # 保存实验结果
        np.savez(f'{model_loss_path}/Network_PlainSingleHead_circ={self.circ}.npz', # 保存DNN的loss
                epoch_index = epoch_index, 
                train_his = train_his2, 
                test_his = test_his2,
                stop_index = stop_index,
                circ = self.circ)
        self.APART_args['PSR_extinction_stop_epoch'] = int(stop_index)

        return best_pth_path, PSR_extinction_best_pth_path


    def DeePMO_train(self, concat_pre = False, rm_tree = True, shrink_strategy = None, extract_strategy = None, 
                           rate = 0.8, device = None, PSR_train_outside_weight = 1, PSRex_cutdown = None,
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
        PSRex_cutdown = self.PSRex_cutdown if PSRex_cutdown is None else PSRex_cutdown

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
            PSRex_cutdown = PSRex_cutdown,
            **kwargs
        )
        best_ppth, PSRex_best_ppth = self._DeePMO_train(device = device, PSR_train_outside_weight = PSR_train_outside_weight)
        
        # 保存DNN的超参数到JSON文件中
        self.WriteCurrentAPART_args(cover = True)
        if lock_json: 
            subprocess.run(f"chmod 444 {self.model_current_json}", shell = True)
        self.TrainAnetLogger.info("="*100)  
        self.TrainAnetLogger.info(f"{best_ppth=}")  
        self.TrainAnetLogger.info(f"{PSRex_best_ppth=}")
        tmp_data = np.load(f"{self.model_loss_path}/APART_loss_his_circ={self.circ}.npz")
        train_his1 = tmp_data['train_his']; test_his1 = tmp_data['test_his']; stop_index = tmp_data['stop_index']
        tmp_data = np.load(f'{self.model_loss_path}/Network_PlainSingleHead_circ={self.circ}.npz')
        train_his2 = tmp_data['train_his']; test_his2 = tmp_data['test_his']; stop_index2 = tmp_data['stop_index']
        
        fig, ax = plt.subplots(1, 4, figsize = (16, 4))
        ax[0].semilogy(train_his1[:,0], lw=1, label='IDTtrain')
        ax[0].semilogy(test_his1[:,0], 'r', lw=1.2, label='IDTtest')
        ax[0].axvline(stop_index, label = 'early stopping', color = 'green', )
        ax[0].set_xlabel('epoch'); 
        ax[0].legend(loc='upper right')
        ax[1].semilogy(train_his1[:,1], lw=1, label='LFStrain')
        ax[1].semilogy(test_his1[:,1], 'r', lw=1.2, label='LFStest')
        ax[1].axvline(stop_index, label = 'early stopping', color = 'green')
        ax[1].set_xlabel('epoch'); 
        ax[1].set_ylabel('loss (log scale)')
        ax[1].legend(loc='upper right')

        ax[2].semilogy(train_his2, lw=1, label='PSRextinction_train')
        ax[2].semilogy(test_his2, 'r', lw=1.2, label='PSRextinction_PSRtest')
        ax[2].axvline(stop_index, label = 'early stopping', color = 'green')
        ax[2].set_xlabel('epoch');
        ax[2].set_ylabel('loss (log scale)')
        ax[2].legend(loc='upper right')
        
        ax[3].semilogy(train_his1[:,2], lw=1, label='Concentrationtrain')
        ax[3].semilogy(test_his1[:,2], 'r', lw=1.2, label='Concentrationtest')
        ax[3].axvline(stop_index, label = 'early stopping', color = 'green')
        ax[3].set_xlabel('epoch'); 
        ax[3].set_ylabel('loss (log scale)')
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/loss_his_circ={self.circ}.png')
        plt.close(fig)

        self.TrainAnetLogger.info(f"Finished Train DNN! Total cost {time.time() - t0:.2f} s")


    def SkipSolveInverse(self, w1 = None, w2 = None, w3 = None, w4 = None, father_sample:str = None, save_dirpath = f'./inverse_skip', 
                             csv_path = None, device = 'cpu', IDT_reduced_threshold = None, PSR_reduced_threshold = None,
                             raw_data = False, experiment_time = 15, **kwargs):
        """
        自动跳过 Inverse 部分直接使用最优样本就可以拿到结果，可以直接放在训练步骤之后使用
        """
        np.set_printoptions(suppress=True, precision=3)
        save_folder = mkdirplus(save_dirpath)
        # 读取本次 json 文件中的 psr_mean 与 psr_std
        json_data = read_json_data(self.model_current_json)
        psr_mean = np.array(json_data['psr_mean']); psr_std = np.array(json_data['psr_std'])
        if w1 is None: w1 = self.IDT_weight
        if w2 is None: w2 = self.PSR_weight
        if w3 is None: w3 = self.LFS_weight
        if w4 is None: w4 = self.PSR_concentration_weight

        # 加载最优的样本
        if not father_sample is None and os.path.exists(father_sample):
            tmp_father_sample = np.load(father_sample)
            inverse_alist = tmp_father_sample['Alist']
        else:
            inverse_alist = self.SortALIST(w1, w2, w3, w4, self.apart_data_path, experiment_time = experiment_time,
                    IDT_reduced_threshold = IDT_reduced_threshold, psr_mean = psr_mean, psr_std = psr_std, logger = self.InverseLogger)
        # 加载网络
        optim_net = load_best_dnn(Network_PlainTripleHead_PSRconcentration, self.model_current_json, device = device)
        optim_psrex_net = load_best_dnn(Network_PlainSingleHead, self.model_current_json, device = device, prefix = "PSR_extinction")
        for index in range(experiment_time):
            self.InverseLogger.info(f'experiment_index: {index}')
            inverse_path = mkdirplus(save_folder + f'/{index}')
            try:  
                # IDT图像部分
                t1 = time.time()
                # 生成初值机理
                A_init = np.array(inverse_alist[index], dtype = np.float64)
                Adict2yaml(eq_dict = self.eq_dict, original_chem_path = self.reduced_mech, chem_path = inverse_path +'/optim_chem.yaml', Alist = A_init)
                # 根据 A_init 查找 self.apart_data_path 中的数据
                apart_data = np.load(self.apart_data_path)
                Alist_data = apart_data['Alist']; idt_data = apart_data['all_idt_data']; psr_data = apart_data['all_psr_data']; 
                psr_extinction_data = apart_data['all_psr_extinction_data']; lfs_data = apart_data['all_lfs_data']; psr_concentration_data = apart_data['all_psr_concentration_data']
                # 查找 Ainit 的 index
                init_index = np.where(np.all(Alist_data == A_init, axis = 1))[0][0] 
                cantera_idt_data = idt_data[init_index]; cantera_psr_data = psr_data[init_index]; cantera_psrex_data = psr_extinction_data[init_index]; cantera_lfs_data = lfs_data[init_index]
                cantera_psr_concentration_data = psr_concentration_data[init_index]
                
                # 简化机理指标 vs 真实机理指标的绘图
                # IDT part
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
                
                # PSR part
                relative_error = np.mean(np.abs((cantera_psr_data - self.true_psr_data) / self.true_psr_data)) * 100
                self.InverseLogger.info(f"Relative Error is {relative_error * 100} %")  
                self.InverseLogger.info("Compare First PSR:" + "\n" + f"True:{self.true_psr_data}; " + "\n" + f"Reduced:{self.reduced_psr_data};\n" + 
                                        f"Cantera:{cantera_psr_data};")
                self.InverseLogger.info("-" * 90)                

                true_psr_data = self.true_psr_data.reshape(self.RES_TIME_LIST.shape[0], -1)
                cantera_psr_data = cantera_psr_data.reshape(self.RES_TIME_LIST.shape[0], -1)
                reduced_psr_data = self.reduced_psr_data.reshape(self.RES_TIME_LIST.shape[0], -1)
                CompareDRO_PSR_lineplot(
                    detail_psr = true_psr_data,
                    reduced_psr = reduced_psr_data,
                    optimal_psr = cantera_psr_data,
                    detail_res_time = self.RES_TIME_LIST,
                    reduced_res_time = self.RES_TIME_LIST,
                    network_res_time = self.RES_TIME_LIST,
                    optimal_res_time = self.RES_TIME_LIST,
                    PSR_condition = self.PSR_condition,
                    save_path = inverse_path + '/compare_nn_PSR.png',
                    n_col = 6,
                )
                
                # PSRex PART
                relative_error = np.mean(np.abs((np.log2(cantera_psrex_data) - np.log2(self.true_extinction_time)) / np.log2(self.true_extinction_time))) * 100
                self.InverseLogger.info(f"log2 Relative Error is {relative_error * 100} %")
                # log2 scale
                final_psrex = optim_psrex_net(torch.tensor(A_init, dtype = torch.float32)).detach().numpy()
                self.InverseLogger.info("Compare First PSRex:" + "\n" + f"True:{self.true_extinction_time}; " + "\n" +
                                        f"Cantera:{cantera_psrex_data};" + f"Final:{2 ** final_psrex};")
                self.InverseLogger.info("-" * 90)
                compare_nn_train3(
                        np.log2(self.true_extinction_time),
                        np.log2(cantera_psrex_data),
                        np.log2(self.reduced_extinction_time),
                        final_psrex, 
                        labels = [r'$Optimal$', r'$Reduced$', r'$DNN\_Optimal$'],
                        markers = ['+', '+', 'o'],
                        colors = ['blue', 'red', 'blue'],
                        title = f'PSR  Relative Error: {relative_error:.2f} %',
                        save_path = inverse_path + '/compare_nn_PSRex.png',
                        wc = self.PSR_condition
                    )
                
                # LFS part
                relative_error = np.mean(np.abs((cantera_lfs_data - self.true_lfs_data) / self.true_lfs_data)) * 100
                self.InverseLogger.info(f"LFS Relative Error is {relative_error} %")  
                # log scale
                true_lfs_data = self.true_lfs_data; reduced_lfs_data = self.reduced_lfs_data
                final_lfs = optim_net.forward_Net2(torch.tensor(A_init, dtype = torch.float32)).detach().numpy()
                # self.InverseLogger.info(f"Average Diff between final_A and A0 is {np.mean(np.abs(A_init - self.reduced_mech_A0))}; While the min and max is {np.min(np.abs(A_init - self.A0))} and {np.max(np.abs(A_init - self.A0))}")
                self.InverseLogger.info("Compare First LFS:" + "\n" + f"True:{true_lfs_data}; " + "\n" + f"Reduced:{reduced_lfs_data};\n" + 
                                        f"Cantera:{cantera_lfs_data};" + f"Final:{final_lfs};")
                self.InverseLogger.info("-" * 90)
                compare_nn_train3(
                        true_lfs_data,
                        cantera_lfs_data,
                        reduced_lfs_data,
                        final_lfs, 
                        labels = [r'$Optimal$', r'$Reduced$', r'$DNN\_Optimal$'],
                        markers = ['+', '+', 'o'],
                        colors = ['blue', 'red', 'blue'],
                        title = f'LFS Relative Error: {relative_error:.2f} %',
                        save_path = inverse_path + '/compare_nn_LFS.png',
                        wc = self.LFS_condition,
                        xlims = (-0.3, 0.3)
                    )                
                
                # PSRconcentration part
                non_zero_index = self.true_psr_concentration_data > 0
                relative_error = np.mean(np.abs((cantera_psr_concentration_data[non_zero_index] - self.true_psr_concentration_data[non_zero_index]) / self.true_psr_concentration_data[non_zero_index])) * 100
                self.InverseLogger.info(f"PSR concentration Relative Error is {relative_error} %")  
                # no-log scale, similar to LFS
                true_psr_concentration_data = self.true_psr_concentration_data; reduced_psr_concentration_data = self.reduced_psr_concentration_data;
                final_psr_concentration = optim_net.forward_Net3(torch.tensor(A_init, dtype = torch.float32)).detach().numpy()
                self.InverseLogger.info("Compare First PSR concentration:" + "\n" + f"True:{true_psr_concentration_data}; " + "\n" + f"Reduced:{reduced_psr_concentration_data};\n" +
                                        f"Cantera:{cantera_psr_concentration_data};" + f"Final:{final_psr_concentration};")
                self.InverseLogger.info("-" * 90)
                if index <= 5:
                    compare_PSR_concentration(
                        true_psr_concentration_data,
                        cantera_psr_concentration_data,
                        reduced_psr_concentration_data,
                        self.PSR_concentration_condition,
                        self.PSR_concentration_species,
                        final_psr_concentration, 
                        save_path = inverse_path + '/compare_nn_PSR_concentration.png',
                    )
                    compare_PSR_concentration(
                        true_psr_concentration_data,
                        cantera_psr_concentration_data,
                        reduced_psr_concentration_data,
                        self.PSR_concentration_condition,
                        self.PSR_concentration_species,
                        save_path = inverse_path + '/compare_nn_PSR_concentration2.png',
                    )
                else:
                    compare_nn_train3(
                        true_psr_concentration_data,
                        cantera_psr_concentration_data,
                        reduced_psr_concentration_data,
                        final_psr_concentration, 
                        labels = [r'$Optimal$', r'$Reduced$', r'$DNN\_Optimal$'],
                        markers = ['+', '+', 'o'],
                        colors = ['blue', 'red', 'blue'],
                        title = f'PSR Concentration Relative Error: {relative_error:.2f} %',
                        save_path = inverse_path + '/compare_nn_PSR_concentration.png',
                        wc = self.PSR_concentration_condition,
                        xlims = (-0.01, 0.01)
                    )
                    
                # 保存 IDT 的相关数据
                np.savez(
                        inverse_path + "/IDT_data.npz",
                        true_idt_data = true_idt_data,
                        reduced_idt_data = reduced_idt_data,
                        cantera_idt_data = cantera_idt_data,
                        dnn_idt_data = np.array(final_idt),
                        true_psr_data = self.true_psr_data,
                        reduced_psr_data = self.reduced_psr_data,
                        cantera_psr_data = cantera_psr_data,
                        true_psrex_data = self.true_extinction_time,
                        cantera_psrex_data = cantera_psrex_data,
                        dnn_psrex_data = np.array(final_psrex),
                        true_lfs_data = true_lfs_data,
                        reduced_lfs_data = reduced_lfs_data,
                        cantera_lfs_data = cantera_lfs_data,
                        dnn_lfs_data = np.array(final_lfs),
                        
                        true_psr_concentration_data = true_psr_concentration_data,
                        reduced_psr_concentration_data = reduced_psr_concentration_data,
                        cantera_psr_concentration_data = cantera_psr_concentration_data,
                        dnn_psr_concentration_data = np.array(final_psr_concentration),
                        
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
            all_idt_data, all_T_data, all_psr_data, Alist_data, all_psr_extinction_data, all_lfs_data = [], [], [], [], [], []
            all_psr_concentration_data = []
            for target_file in files:
                target_file = os.path.join(save_path, target_file)
                if os.path.getsize(target_file) >= 5:
                    tmp = np.load(target_file)
                    if len(tmp['Alist']) == 0: break
                    try:
                        all_idt_data.append(tmp['IDT'].tolist());   all_T_data.append(tmp['T'].tolist())
                        all_psr_data.append(tmp['PSR_T'].tolist());   Alist_data.append(tmp['Alist'].tolist()); 
                        all_psr_extinction_data.append(tmp['PSR_extinction'].tolist())
                        all_lfs_data.append(tmp['LFS'].tolist()); all_psr_concentration_data.append(tmp['PSR_concentration'].tolist())
                    except:
                        print(target_file, traceback.format_exc())
                    if len(Alist_data) % 100 == 0:
                        logger.info(f"Cost {time.time() - t0:.1f}, Gather Data Process has finished {len(Alist_data)/filenums * 100:.2f} %")
            np.savez(save_file_name,
                            all_idt_data = all_idt_data, all_T_data = all_T_data, 
                            all_psr_data = all_psr_data, Alist = Alist_data, 
                            all_psr_extinction_data = all_psr_extinction_data,
                            all_lfs_data = all_lfs_data, all_psr_concentration_data = all_psr_concentration_data)
            logger.info("apart_data Saved!")
            if rm_tree:
                # 不使用 rm -rf 的原因是因为 rm -rf 删除大批量文件的效率太低
                # 使用 rsync 命令删除文件夹
                logger.info(f"removing the tmp files in {save_path}")
                mkdirplus("./data/APART_data/blank_dir")
                subprocess.run(f"rsync --delete-before -d -a ./data/APART_data/blank_dir/ {save_path}/", shell = True)
            return all_idt_data, all_T_data, all_psr_data, Alist_data, all_psr_extinction_data, all_lfs_data, all_psr_concentration_data
        else:
            self.GenAPARTDataLogger.warning("gather_apart_data function is out-of-commision")


    def gather_apart_data_PSRco(self, save_path = "./data/APART_data/tmp", save_file_name = None,  rm_tree = True, 
                        cover = True, logger = None, **kwargs):
        """
        重写特化于当前脚本的 gather apart data; 不继承原有的函数
        """
        t0 = time.time()
        save_file_name = self.apart_data_path if save_file_name is None else save_file_name
        files = [file for file in os.listdir(save_path) if file.find('.npz') != -1]
        filenums = len(files); logger = self.TrainAnetLogger if logger is None else logger
        if cover or not os.path.exists(save_file_name) or os.path.getsize(save_file_name) /1024/1024 <= 1:
            all_idt_data, all_T_data, all_psr_data, Alist_data, all_psr_extinction_data = [], [], [], [], []
            all_psr_concentration_data = []
            for target_file in files:
                target_file = os.path.join(save_path, target_file)
                if os.path.getsize(target_file) >= 5:
                    tmp = np.load(target_file)
                    if len(tmp['Alist']) == 0: break
                    all_idt_data.append(tmp['IDT'].tolist());   all_T_data.append(tmp['T'].tolist())
                    all_psr_data.append(tmp['PSR_T'].tolist());   Alist_data.append(tmp['Alist'].tolist()); 
                    all_psr_extinction_data.append(tmp['PSR_extinction'].tolist())
                    all_psr_concentration_data.append(tmp['PSR_concentration'].tolist())
                    if len(Alist_data) % 100 == 0:
                        logger.info(f"Cost {time.time() - t0:.1f}, Gather Data Process has finished {len(Alist_data)/filenums * 100:.2f} %")
            np.savez(save_file_name,
                            all_idt_data = all_idt_data, all_T_data = all_T_data, 
                            all_psr_data = all_psr_data, Alist = Alist_data, 
                            all_psr_extinction_data = all_psr_extinction_data,
                            all_psr_concentration_data = all_psr_concentration_data
                            )
            logger.info("apart_data Saved!")
            if rm_tree:
                # 不使用 rm -rf 的原因是因为 rm -rf 删除大批量文件的效率太低
                # 使用 rsync 命令删除文件夹
                logger.info(f"removing the tmp files in {save_path}")
                mkdirplus("./data/APART_data/blank_dir")
                subprocess.run(f"rsync --delete-before -d -a ./data/APART_data/blank_dir/ {save_path}/", shell = True)
            return all_idt_data, all_T_data, all_psr_data, Alist_data, all_psr_extinction_data, all_psr_concentration_data
        else:
            self.GenAPARTDataLogger.warning("gather_apart_data function is out-of-commision")


    def SortALISTStat(self, apart_data_path = None, father_sample_path = None):
        """
        用于分析每步采样后得到的 apart_data 中数据的分布情况，分为 

        1.  Alist 与 A0 之间的距离，Alist 之间的距离，Alist 在每个维度上的方差，Alist 在每个维度上分布的图像
        2.  IDT 在每个工况上的分布情况
        3.  PSR 在每个工况和每个特定的 res_time 上的分布情况; PSR 与 IDT 的相关性
        4.  Mole 在每个工况上的分布情况; Mole 与 IDT 的相关性
        5.  LFS 在每个工况上的分布情况; LFS 与 IDT 的相关性
        """
        np.set_printoptions(precision = 2, suppress = True)
        apart_data_path = self.apart_data_path if apart_data_path is None else apart_data_path
        logger = Log(f"./log/SortALISTStat_circ={self.circ}.log"); mkdirplus("./analysis")
        # 读取数据
        apart_data = np.load(apart_data_path)
        Alist = apart_data['Alist']; IDT = apart_data['all_idt_data']; PSR = apart_data['all_psr_data'];
        LFS = apart_data['all_lfs_data']
        # 样本点与原点 A0 的平均距离
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
        ax[0].set_ylim(-3.5, 3.5)
        ax[1].set_title("Father Sample Distribution")
        ax[1].set_xticks(np.arange(1, len(self.reduced_mech_A0) + 1))
        ax[1].set_xticklabels([f"{i}" for i in range(1, len(self.reduced_mech_A0) + 1)])
        ax[1].set_xlabel("Dimension")
        fig.savefig(f"./analysis/Alist-A0_circ={self.circ}.png")
        plt.close(fig)
        logger.info("Alist - A0 的分布情况已保存至 ./analysis/Alist-A0.png")


        # 调用 sample_distribution 函数绘制 IDT 与 PSR 的分布情况
        ## 加载上一步中的最优样本点
        previous_best_chem = f"./data/APART_data/reduced_data/previous_best_chem_circ={self.circ}.yaml"
        previous_best_chem_IDT, _ = yaml2idt(
            previous_best_chem, mode = self.IDT_mode, 
            IDT_condition = self.IDT_condition, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
            cut_time = self.APART_args.get("True_IDT_Cut_Time", 50)
        )
        previous_best_chem_PSR = yaml2psr(
            previous_best_chem, PSR_condition = self.PSR_condition, RES_TIME_LIST = self.RES_TIME_LIST,
            fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer, error_tol = 0.
        )   
        previous_best_chem_PSRex = yaml2psr_extinction_time(
            previous_best_chem, PSR_condition = self.PSR_condition, init_res_time= self.init_res_time,
            fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer, exp_factor = self.PSRex_decay_rate
        )
        previous_best_chem_LFS = yaml2FS_Mcondition(
            previous_best_chem, FS_condition = self.LFS_condition, fuel = self.LFS_fuel, oxidizer = self.LFS_oxidizer,
            cpu_process = os.cpu_count() - 1
        )
        ## 调用 sample_distribution 函数绘制 IDT 与 PSR 的分布情况
        from Apart_Package.APART_plot.APART_plot import sample_distribution
        asamples = np.load(os.path.dirname(apart_data_path) + f"/Asamples_{self.circ}.npy")
        network = load_best_dnn(Network_PlainTripleHead_PSRconcentration, self.model_current_json, device = 'cpu')
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
        # 使用 sample_distribution_PSR 来展示 LFS 的结果
        LFS_func = lambda x: network.forward_Net2(torch.tensor(x, dtype = torch.float32)).detach().numpy()
        sample_distribution_IDT(
                                LFS,
                                self.true_lfs_data,
                                self.reduced_lfs_data,
                                IDT_func = LFS_func,
                                asamples = asamples,
                                marker_idt=previous_best_chem_LFS,
                                IDT_condition = self.LFS_condition,
                                xlim = (-0.5, 0.5),
                                save_path = f"./analysis/SampleDistribution_LFS_circ={self.circ}.png"
                                )
        sample_distribution_PSR(
                                PSR,
                                self.true_psr_data,
                                self.reduced_psr_data,
                                asamples = asamples,
                                marker_psr=previous_best_chem_PSR,
                                PSR_condition = self.PSR_condition,
                                RES_TIME_LIST = self.RES_TIME_LIST,
                                save_path = f"./analysis/SampleDistribution_PSR_circ={self.circ}.png"
                                )
        network = load_best_dnn(Network_PlainSingleHead, self.model_current_json, device = 'cpu', prefix = "PSR_extinction")
        PSRex_func = lambda x: network(torch.tensor(x, dtype = torch.float32)).detach().numpy()
        sample_distribution_PSRex(
                                np.log2(apart_data['all_psr_extinction_data']),
                                np.log2(self.true_extinction_time),
                                np.log2(self.reduced_extinction_time),
                                marker_psrex=np.log2(previous_best_chem_PSRex),
                                PSRex_func = PSRex_func, asamples = asamples,
                                PSRex_condition = self.PSR_condition,
                                save_path = f"./analysis/SampleDistribution_PSRex_circ={self.circ}.png"
        )
        logger.info("IDT, PSR, PSRex, LFS 的分布情况已保存至 ./analysis/ 中")
        return dist_A0, dist_center, std_sample


    # def PSRex_SensitivityDesent(self, start_samples = None, target_PSR_conditions = None, iter_nums = 10, delta = 0.01,  
    #                         eve_sample_size = 50, max_step_size = 0.2, cpu_nums = None, init_res_time = None, father_sample_size = None,
    #                           **kwargs):
    #     """
    #     适合于 self.reduced_mech PSR 点火失败的情况
    #     从 start_samples 中计算 PSRex 的敏感度，然后进行多次循环下降过程，多次循环中尝试寻找能够使得所有对应工况的 PSR 均点火的样本点
    #     获得合适的样本点后使用多进程计算它们的 PSRex 值并选择其中最小的样本点

    #     Args:
    #         start_samples: 二维数组，进行敏感度下降的初始样本点
    #         target_PSR_conditions: 二维数组，无法成功点火的 PSR 工况
    #         iter_nums: 敏感度下降的迭代次数
    #         delta: 计算敏感度使用的步长
    #         eve_sample_size: 每次迭代中，每个父样本点生成的子样本点的数量
    #         max_step_size: 每次迭代中，敏感度下降中每个父样本点生成的子样本点的最大步长
    #         cpu_nums: 多进程计算的 cpu 数量
    #     Returns:
    #         Alist: 二维数组，最终的样本点
    #         all_idt_data, all_T_data, all_psr_data, all_psr_extinction_data: 与 Alist 对应的 idt, PSR, PSRex 数据
    #     """
    #     np.set_printoptions(precision = 6, suppress = True)
    #     dirpath = mkdirplus("./data/SensitivityDesent")
    #     target_PSR_conditions = self.PSR_condition if target_PSR_conditions is None else target_PSR_conditions
    #     cpu_nums = cpu_nums if cpu_nums is not None else cpu_count() - 1
    #     father_samples = start_samples if start_samples is not None else [self.A0]
    #     logger = Log("./log/SensitivityDesent.log")
    #     init_res_time = np.amax(self.init_res_time) if init_res_time is None else init_res_time
    #     father_sample_size = len(father_samples) if father_sample_size is None else father_sample_size

    #     benchmark_PSRex = yaml2psr_extinction_time(
    #         self.detail_mech, target_PSR_conditions, 
    #         fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer,
    #         psr_tol = 0, init_res_time = init_res_time, exp_factor = self.PSRex_decay_rate
    #     )
    #     DDesent_samples = []
    #     for iter in range(iter_nums):
    #         Desent_samples = []
    #         for i, sample in enumerate(father_samples):
    #             tmp_path = dirpath + f"/{i}th_father_sample.yaml"
    #             Adict2yaml(self.reduced_mech, tmp_path, Alist = sample, eq_dict = self.eq_dict)
    #             # 计算 IDT 和 PSR 的敏感性
    #             PSRex_sensitivity, base_PSRex = yaml2PSRex_sensitivity(
    #                 tmp_path, target_PSR_conditions, delta = delta, 
    #                 fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer,
    #                 psr_tol = 0, init_res_time = init_res_time, exp_factor = self.PSRex_decay_rate,
    #                 need_base_PSRex = True
    #             )
    #             # 针对 IDT 和 PSR 的 loss 进行敏感性下降。我们考虑的损失函数是： 
    #             # \sum_{pp} (IDT_{pp} - IDTtrue_{pp})^2 + \sum_{pp} (PSR_{pp} - PSRtrue_{pp})^2; 因此求导后获得
    #             # \sum_{pp} 2 * (IDT_{pp} - IDTtrue_{pp}) * PSRex_sensitivity_{pp} + \sum_{pp} 2 * (PSR_{pp} - PSRtrue_{pp}) * PSR_sensitivity_{pp}
    #             ## 根据 self.eq_dcit 重排序 PSRex_sensitivity 转化为 numpy 数组
    #             PSRex_sensitivity_list = eq_dict_broadcast2Alist(PSRex_sensitivity, self.eq_dict)
    #             # 计算损失函数的梯度
    #             grad = 2 * np.sum((base_PSRex - benchmark_PSRex) * PSRex_sensitivity_list, axis = 1)
    #             # 选择合适的步长进行下降，保证下降范围在 0.1 内
    #             step = max_step_size / np.max(np.abs(grad))
    #             # 进行下降
    #             desent_sample = sample - step * grad
    #             # 在 sample 和 desent_sample 两点连成的直线上均匀取 50 个点
    #             desent_samples = []
    #             for j in range(eve_sample_size):
    #                 desent_samples.append((sample + j / 10 * (desent_sample - sample)).tolist())
    #             Desent_samples.append(desent_samples)
    #         DDesent_samples.append(Desent_samples)
    #         father_samples = np.array(Desent_samples).reshape(-1, len(sample))
    #         np.random.shuffle(father_samples)
    #         father_samples = father_samples[:father_sample_size]
    #         logger.info(f"iter {iter} finished, sampled {len(DDesent_samples) * len(DDesent_samples[0])} points.")
    #         logger.info(f"The example of base_PSRex is {base_PSRex}, the benchmark_PSRex is {benchmark_PSRex}, desent step is {step * grad}.")
    #     # 将 DDsent_samples 转化为二维数组, 其中一个维度是 sample 的维度
    #     DDesent_samples = np.array(DDesent_samples).reshape(-1, len(sample))
    #     # 使用多进程计算所有的 PSRex 值
    #     with ProcessPoolExecutor(max_workers = cpu_nums) as exec:
    #         for index, sample in enumerate(DDesent_samples):
    #             exec.submit(
    #                     GenOneDataIDT_LFS_PSRextinction, 
    #                     index = index,
    #                     IDT_condition = self.IDT_condition,
    #                     PSR_condition = self.PSR_condition,
    #                     RES_TIME_LIST = self.RES_TIME_LIST,
    #                     init_res_time = self.init_res_time,
    #                     Alist = sample,
    #                     eq_dict = self.eq_dict,
    #                     LFS_fuel = self.LFS_fuel,
    #                     LFS_oxidizer = self.LFS_oxidizer,
    #                     IDT_fuel = self.IDT_fuel,
    #                     IDT_oxidizer = self.IDT_oxidizer,
    #                     PSR_fuel = self.PSR_fuel,
    #                     PSR_oxidizer = self.PSR_oxidizer,
    #                     reduced_mech = self.reduced_mech,
    #                     IDT_mode = self.IDT_mode,
    #                     # 将 idt_arrays 采取如下设置: 其每个分量是 true_idt_data 和 reduced_idt_data 对应分量中大的那个
    #                     idt_arrays = np.maximum(self.true_idt_data, self.reduced_idt_data),
    #                     cut_time_alpha = 10,
    #                     error_tol = 0, 
    #                     exp_factor = self.PSRex_decay_rate,
    #                     save_path = "./data/SensitivityDesent",
    #                     my_logger = logger,
    #                     **kwargs
    #                     )
    #     # 使用 gather_data 函数将数据收集起来
    #     self.gather_apart_data(
    #         save_path = "./data/SensitivityDesent",
    #         save_file_name = "./data/SensitivityDesentData.npz",
    #         rm_tree = True,
    #         logger = logger
    #     )
    #     # 加载 "./data/SensitivityDesentData.npz" 
    #     data = np.load("./data/SensitivityDesentData.npz"); Alist = data['Alist']; all_idt_data = data['all_idt_data']
    #     all_psr_data = data['all_psr_data']; all_psr_extinction_data = data['all_psr_extinction_data']
    #     # 将 data['all_psr_extinction_data'] - self.true_psr_extinction_data 的绝对值计算无穷范数，得到的是一个一维数组
    #     diff_psrex = np.linalg.norm(all_psr_extinction_data - self.true_extinction_time, ord = np.inf, axis = 1)
    #     # 将 diff_psrex 升序排序后选取前 10 个 Alist
    #     index = np.argsort(diff_psrex)[:10]
    #     Alist = Alist[index]; all_idt_data = all_idt_data[index]; all_psr_data = all_psr_data[index]; all_psr_extinction_data = all_psr_extinction_data[index]
    #     # 将这 10 个 Alist 保存到 "./data/SensitivityDesentData.npz" 中
    #     np.savez("./data/SensitivityDesentData_filtered.npz", Alist = Alist, all_idt_data = all_idt_data, all_psr_data = all_psr_data, all_psr_extinction_data = all_psr_extinction_data)
    #     return Alist, all_idt_data, all_psr_data, all_psr_extinction_data


def GenOneDataIDT_LFS_PSRex_concentration(IDT_condition: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, LFS_condition: np.ndarray, PSR_concentration_kwargs: dict,
                                          Alist:list, eq_dict:dict, reduced_mech:str, index:int,  my_logger:Log, tmp_chem_path:str = None, fuel:str = None, oxidizer: str = None, 
                                          IDT_mode = 0, remove_chem = True, idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, PSR_EXP_FACTOR = 0.5,
                                          cut_time_alpha = 10, psr_error_tol = 50, init_res_time:np.ndarray = 1, save_path = 'data/APART_data/tmp',
                                          IDT_fuel:str = None, IDT_oxidizer: str = None, PSR_fuel:str = None, PSR_oxidizer: str = None, 
                                          LFS_fuel = None, LFS_oxidizer = None, **kwargs):
    """
    将 IDT 和 PSR 结合起来
    相比 IDT 的函数增加了 PSR_condition 和 RES_TIME_LIST, psr_error_tol 3个参数
    """
    tmp_chem_path = save_path + f'/{index}th.yaml' if tmp_chem_path is None else tmp_chem_path
    save_path = save_path + f'/{index}th.npz'
    Adict2yaml(reduced_mech, tmp_chem_path, eq_dict = eq_dict, Alist = Alist)
    # 将 PSR_condition RES_TIME_LIST 等转化为 array
    PSR_condition = np.array(PSR_condition); RES_TIME_LIST = np.array(RES_TIME_LIST); IDT_condition = np.array(IDT_condition)
    LFS_condition = np.array(LFS_condition); init_res_time = np.array(init_res_time)
    t0 = time.time()
    if PSR_fuel is None: PSR_fuel = fuel
    if PSR_oxidizer is None: PSR_oxidizer = oxidizer
    PSR_T, PSR_extinction = _GenOnePSR_with_extinction(PSR_condition, RES_TIME_LIST, PSR_fuel, PSR_oxidizer, tmp_chem_path, index, 
                                      my_logger, psr_error_tol = psr_error_tol, PSR_EXP_FACTOR = PSR_EXP_FACTOR, **kwargs)
    t1 = time.time()
    if IDT_fuel is None: IDT_fuel = fuel
    if IDT_oxidizer is None: IDT_oxidizer = oxidizer
    idt_T = _GenOneIDT(IDT_condition, IDT_fuel, IDT_oxidizer, tmp_chem_path,  index,  my_logger, IDT_mode,idt_arrays, cut_time, cut_time_alpha, **kwargs)
    t2 = time.time()
    if LFS_fuel is None: LFS_fuel = fuel
    if LFS_oxidizer is None: LFS_oxidizer = oxidizer
    LFS = _GenOneLFS(LFS_condition, LFS_fuel, LFS_oxidizer, tmp_chem_path,  index,  my_logger, **kwargs)

    t3 = time.time()
    PSR_concentration_condition = np.array(PSR_concentration_kwargs['condition'])
    concentration_species = PSR_concentration_kwargs['species']
    concentration_res_time = PSR_concentration_kwargs['res_time']
    concentraion_fuel, concentration_oxidizer, concentration_diluent = PSR_concentration_kwargs['fuel'], PSR_concentration_kwargs['oxidizer'], PSR_concentration_kwargs['diluent']
    
    PSR_concentration = _GenOnePSR_concentration(
        species=concentration_species,
        condition=PSR_concentration_condition,
        res_time=concentration_res_time,
        fuel=concentraion_fuel, oxidizer=concentration_oxidizer, diluent=concentration_diluent,
        tmp_chem_path=tmp_chem_path, index=index, my_logger=my_logger,
    )
    t4 = time.time()
    if remove_chem: os.remove(tmp_chem_path)
    if isinstance(idt_T, int) or isinstance(PSR_T, int) or isinstance(LFS, int) or isinstance(PSR_concentration, int):
        return idt_T
    else:
        IDT, T = idt_T
        my_logger.info(f"Mechanism {index} : cost {t2 - t1:.2f} + {t1 - t0:.2f} + {t3 - t2:.2f} + {t4 - t3:.2f} s, the first IDT element is {np.log10(IDT[0]):.3e}, the first PSR element is {PSR_T[0:3]}, the first PSR extinction time is {PSR_extinction[0:3]}, " + 
                       f"first LFS element is {LFS[0]:.3e}, the first PSR concentration is {len(PSR_concentration)} Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024:.2e} GB")
        np.savez(save_path, IDT = IDT, T = T, PSR = PSR_T, Alist = Alist, PSR_extinction = PSR_extinction, LFS = LFS, PSR_concentration = PSR_concentration)
        return None
    
