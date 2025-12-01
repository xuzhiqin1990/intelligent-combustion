# -*- coding:utf-8 -*-

import os, sys, time, shutil, psutil, GPUtil, traceback, subprocess, warnings
# 在程序运行之前且导入 torch 之前先确定是否使用 GPU
try:
    device = GPUtil.getFirstAvailable(maxMemory=0.5, maxLoad=0.5)
    print("Avaliable GPU is ", device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
except:
    pass

import numpy as np, pandas as pd, seaborn as sns, cantera as ct, torch.nn as nn
import matplotlib.pyplot as plt
from func_timeout import func_set_timeout
from torch.utils.data import DataLoader, random_split
from torch import optim
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .basic_set import _DeePMO
from APART_base import _GenOneIDT
from APART_plot.APART_plot import compare_nn_train3, sample_distribution_IDT
from utils.cantera_utils import *
from utils.setting_utils import *
from utils.yamlfiles_utils import * 
from .DeePMO_V1_Network import Network_PlainSingleHead, DATASET_SingleHead
from utils.cantera_multiprocess_utils import yaml2FS_Mcondition, yaml2IDT_sensitivity_Multiprocess


class DeePMO_IDT(_DeePMO):
    """
    创建于 20230801; 为了替换原有的 APART132 而重写; 直接继承 APART_base 类并在之后的时间里将 DeePMO_IDT 部分也继承到 APART_base 类
    在本模块中只关系 IDT 的计算; 因此将删除除了 IDT 之外的所有优化 QoI 的部分
    同时将涉及到实验数据的部分同样重写如果需要可以在后续增加一些实验机理的相关指示函数
    """
    def __init__(self, circ = 0, setup_file: str = './settings/setup.yaml',  SetAdjustableReactions_mode:int = None,
                 previous_best_chem:str = None, GenASampleRange = None, GenASampleRange_mode = None, **kwargs) -> None:
        
        # 定义目前采样 DNN 筛选循环次数
        super().__init__(circ = circ, setup_file = setup_file, SetAdjustableReactions_mode = SetAdjustableReactions_mode, **kwargs)
        GenASampleRange_mode = self.APART_args.get('GenASampleRange_mode', None) if GenASampleRange_mode is None else GenASampleRange_mode
        
        self.reduced_mech_A0, self.reduced_mech_eq_dict = yaml_key2A(self.reduced_mech)

        if GenASampleRange is None:
            GenASampleRange = True if GenASampleRange_mode is not None else False
        # 计算最大热释放率
        if GenASampleRange:
            if self.circ == 0: 
                previous_best_chem = self.reduced_mech
                previous_best_chem_IDT = self.reduced_idt_data
            else:
                previous_best_chem, previous_best_chem_IDT,  _ = self.SortALIST(
                    apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                    experiment_time = 1,
                    SortALIST_T_threshold_ratio = self.APART_args.get('SortALIST_T_threshold_ratio', None),
                    return_all=True,
                )
                previous_best_chem = np.squeeze(previous_best_chem)
                previous_eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ - 1}.json")
                previous_best_chem = Adict2yaml(self.reduced_mech, f"./data/APART_data/reduced_data/previous_best_chem_circ={self.circ}.yaml", previous_eq_dict, previous_best_chem)
            # previous_best_chem = f"./inverse_skip/circ={self.circ}/0/optim_chem.yaml" if previous_best_chem is None else previous_best_chem
            self.GenASampleRange(mode = GenASampleRange_mode, target_chem = previous_best_chem, **kwargs)
            self.GenASampleThreshold( 
                best_chem_IDT = previous_best_chem_IDT, 
                **kwargs
            )
        else:
            # 在不调用 GenAsampleRange 的情况下, 需要根据之前生成的 eq_dict 更新 self.eq_dict 和 self.A0
            self.eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json")
            self.A0 = eq_dict2Alist(self.eq_dict)
            

        self.WriteCurrentAPART_args(
            IDT_weight = self.IDT_weight,
            GenASampleRange_mode = GenASampleRange_mode,
            GenASampleRange = GenASampleRange,
            # **currentAPART_args
        )


    def LoadPreviousSetupIDT_condition(self, circ, json_name = None):
        """
        加载之前 CIRC 中的设置到 APART_args，因为设置可能发生了变化使得 setup.yaml 文件中的一些设定变的过时
        加载后将直接覆盖 __init__ 中设置的 self.APART_args 中的 IDT_condition
        直接覆盖 self.IDT_condition
        """
        if json_name is None: 
            if circ == 0:
                json_name = f"./model/model_pth/settings_circ={circ}.json"
            else:
                json_name = f"./model/model_pth/settings_circ={circ - 1}.json"
        json_data = read_json_data(json_name)
        self.APART_args.update(IDT_condition = json_data['IDT_condition'])
        self.IDT_condition = np.array(self.APART_args['IDT_condition'])
        self.GenAPARTDataLogger.info(f"LoadPreviousSetupIDT_condition FINISHED! IDT_condition_length = {len(self.IDT_condition)}")


    def LoadCurrentSetupIDT_condition(self, circ, json_name = None):
        """
        加载目前 CIRC 中的设置到 APART_args
        加载后将直接覆盖 __init__ 中设置的 self.APART_args
        直接覆盖 self.IDT_condition
        """
        if json_name is None: 
            json_name = f"./model/model_pth/settings_circ={circ}.json"
        json_data = read_json_data(json_name)
        self.APART_args.update(json_data)
        self.IDT_condition = np.array(self.APART_args['IDT_condition'])


    def __OldSetAdjustableReactions(self, mode = 0, reserved_equations = None, save_jsonpath = None, logger = None, **kwargs):
        """
        设置允许调整的化学反应; 在 basic set 中调用
        存在以下设置方法
            0. 根据 kwargs 中参数键，自动化设置 mode
            1. 通过给定反应物列表，设置所有列表中物质作为反应物的反应为可调整反应
            2. 在 1 的基础上，增加列表中物质作为生成物的反应(可逆反应)
            3. 通过给定反应列表，设置所有列表中反应为可调整反应
            4. 通过灵敏度分析的结果，对不同 QoI 灵敏度归一化后求和，
                按照灵敏度和大小排序后设置前一定比例的反应为可调整反应
            
        params:
            mode: 设置模式，1-4
            reserved_equations: 需要单独保存的反应列表，mode = 1, 2, 4
            save_jsonpath: 保存的 json 文件路径，mode = 1, 2, 4
            kwargs: 
                reactors: mode = 1 需要指定的参数; 或者在 APART_args 中以 rea_keywords 指定
                reactors: mode = 2 需要指定的参数
                reactions: mode = 3 需要指定的参数
                mode = 4 需要指定的参数:
                    target_chem: 计算灵敏度的目标化学反应
                    select_ratio: 选择多少比例的反应用于调整
                    IDT_sensitivity, default: None
        
        requirement for APART_args:
            rea_keywords: mode = 1, 2
            reactions: mode = 3
            SetAdjustableReactions_select_ratio: mode = 4
        add:
            self.eq_dict, self.reduced_mech_A0, self.gen_yaml, self.APART_args['eq_dict']
        """
        raise DeprecationWarning("This function is deprecated, please use SetAdjustableReactions instead!")
        if save_jsonpath is None: save_jsonpath = './data/APART_data/reduced_data/SetAdjustableReactions_Sensitivity.json'
        mkdirplus(os.path.dirname(save_jsonpath))
        # 如果 mode 为 0 需要根据 kwargs 中相应的参数是否存在设置 mode
        if mode == 0:
            if 'reactors' in kwargs or 'rea_keywords' in self.APART_args:
                mode = 1
            elif 'reactions' in kwargs or 'reactions' in self.APART_args:
                mode = 3
            elif 'select_ratio' in kwargs or 'SetAdjustableReactions_select_ratio' in self.APART_args:
                mode = 4
            else:
                mode = 2
                warnings.warn('mode is given as 0, but no reactors, reactions or select_ratio in kwargs, use mode = 2 and set reactors as None')
        match mode:
            case 1: # 通过给定反应物列表，设置所有列表中物质作为反应物的反应为可调整反应
                # 从 kwargs pop 出 reactors 或者从 self.APART_args 中读取 rea_keywords
                reactors = self.APART_args.get('rea_keywords', False)
                if not reactors: reactors = kwargs.pop('reactors', None)
                self.GenAPARTDataLogger.info(f"SetAdjustableReactions: mode = 1, reactors = {reactors}")
                _, eq_dict = yaml_key2A(self.reduced_mech, rea_keywords = reactors,)
            case 2: # 在 1 的基础上，增加列表中物质作为生成物的反应(可逆反应)
                reactors = self.APART_args.get('rea_keywords', False)
                if not reactors: reactors = kwargs.pop('reactors', None)
                self.GenAPARTDataLogger.info(f"SetAdjustableReactions: mode = 2, reactors = {reactors}")
                _, tmp_eq_dict1 = yaml_key2A(self.reduced_mech, rea_keywords = reactors,)
                _, tmp_eq_dict2 = yaml_key2A(self.reduced_mech, pro_keywords = reactors)
                eq_dict = dict(tmp_eq_dict1, **tmp_eq_dict2)
            case 3: # 通过给定反应列表，设置所有列表中反应为可调整反应
                ## 从字典 kwargs 中读取 reactions 若不存在报错 KeyError: When mode = 3, reactions must be given!
                reactions = self.APART_args.get('reactions', False)
                if not reactions: reactions = kwargs.pop('reactions', None)
                if reactions is None: raise KeyError("When mode = 3, reactions must be given!")
                _, eq_dict = yaml_eq2A(self.reduced_mech, *reactions)
            case 4:
                ## 从字典 kwargs 中读取 IDT_sensitivity
                target_chem = kwargs.pop('target_chem', self.reduced_mech)
                select_ratio = kwargs.pop('select_ratio', 0.3)
                if not os.path.exists(save_jsonpath): 
                    ## 计算所有反应关于 IDT 的灵敏度
                    IDT_sensitivity = yaml2idt_sensitivity(
                        target_chem,
                        IDT_condition = self.IDT_condition,
                        fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                        mode = self.IDT_mode
                    )
                    # IDT_sensitivity 内所有 value 取绝对值后求平均值，替换原来的位置
                    IDT_sensitivity = {k: np.mean(np.abs(v)) for k, v in IDT_sensitivity.items()}
                    # 所有的 value 标准化: value - min(value) / (max(value) - min(value))
                    IDT_sensitivity = {k: (v - min(IDT_sensitivity.values())) / (max(IDT_sensitivity.values()) - min(IDT_sensitivity.values())) for k, v in IDT_sensitivity.items()}
                    # # 根据 value 降序排序
                    # IDT_sensitivity = dict(sorted(IDT_sensitivity.items(), key = lambda item: item[1], reverse = True))
                    write_json_data(save_jsonpath, IDT_sensitivity)
                else:
                    IDT_sensitivity = read_json_data(save_jsonpath)
                Sensitivity = copy.deepcopy(IDT_sensitivity)
                # 根据 value 降序排序 Sensitivity
                Sensitivity = dict(sorted(Sensitivity.items(), key = lambda item: item[1], reverse = True))
                # 选取 Sensitivity 中 value 最大的 select_ratio * len(Sensitivity) 个 key 作为 target_reactions
                target_reactions = list(Sensitivity.keys())[: int(select_ratio * len(Sensitivity))]
                _, eq_dict = yaml_eq2A(self.reduced_mech, *target_reactions)
        if not reserved_equations is None:
            # 单独获得 reserved_equations 的 eq_dict
            _, reserved_eq_dict = yaml_eq2A(self.reduced_mech, *reserved_equations)
            # 将 reserved_eq_dict 与 eq_dict 合并
            eq_dict.update(reserved_eq_dict)

        Alist = eq_dict2Alist(eq_dict); self.eq_dict = eq_dict    
        self.reduced_mech_A0 = Alist
        self.gen_yaml:Callable = partial(Adict2yaml, eq_dict = self.eq_dict)
        self.APART_args['eq_dict'] = self.eq_dict


    def WriteCurrentAPART_args(self, save_json = None, cover = False, **kwargs):
        """
        保存当前下的 APART_args 中的参数
            kwargs: 存放参数用于 update APART_args 中的 key-value
        rewrite: 
            self.APART_args
        """
        if save_json is None: save_json = self.model_current_json
        self.APART_args.update(kwargs)
        write_json_data(save_json, self.APART_args, cover = cover)

        
    """==============================================================================================================="""
    """                                  Asample  &   Self-Adjustment Strategy                                       """
    """==============================================================================================================="""

    def SetAdjustableReactions(self, save_jsonpath = None, logger = None, **kwargs):
        """
        20241125：新版只按照含碳量来设置可调节反应，应用于 GenASampleRange 中的 C_numbers 模式
        """
        warnings.warn(f'Not ready to use this function for some alignment problems, please use yaml2FS_sensitivity instead')
        

    def GenASampleRange(self, l_alpha = None, r_alpha = None, mode = 'default', target_chem = None, **kwargs):    
        """
        生成采样的范围; 根据 mode 选择采样范围的生成依据. 需要在 init 中就启动
        params:
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
        
        C_numbers = self.APART_args.get('C_numbers', None)
        target_chem = target_chem if target_chem is not None else self.reduced_mech
        _, eq_dict = yaml_key2A(target_chem,)
        match mode:
            case 'default':
                self.eq_dict = eq_dict
                A0 = eq_dict2Alist(self.eq_dict)
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
                reactions_group = reactions_division_by_C_num(self.reduced_mech, C_numbers, eq_dict)
                # 如果 max(C_numbers) 对应的 {C_numbers}_l_alpha 和 _r_alpha 不存在 self.APART_args 中，则使用默认值
                # C_max = max([spec.composition.get('C', 0) for spec in ct.Solution(self.reduced_mech).species()])
                # if f'{C_max}_l_alpha' not in self.APART_args:
                #     self.APART_args[f'{C_max}_l_alpha'] = self.APART_args[f'{max(C_numbers)}_l_alpha']
                #     self.APART_args[f'{C_max}_r_alpha'] = self.APART_args[f'{max(C_numbers)}_r_alpha']
                print(f"anet.reduced_mech: {self.reduced_mech}, C_numbers: {C_numbers}, reactions_group: {reactions_group}")
                for i, num in enumerate(sorted(C_numbers)):
                    num = int(num) if i < len(reactions_group) - 1 else 'max'
                    tmp_l_alpha = self.APART_args[f'C{num}_l_alpha']
                    tmp_r_alpha = self.APART_args[f'C{num}_r_alpha']
                    tmp_l_alpha = tmp_l_alpha[self.circ] if isinstance(tmp_l_alpha, Iterable) else tmp_l_alpha
                    tmp_r_alpha = tmp_r_alpha[self.circ] if isinstance(tmp_r_alpha, Iterable) else tmp_r_alpha
                    if tmp_l_alpha == 0 and tmp_r_alpha == 0:
                        continue
                    tmp_A0, tmp_eq_dict = yaml_eq2A(target_chem, *reactions_group[i], )
                    self.eq_dict.update(tmp_eq_dict)
                    self.alpha_dict.update(
                        {key: (tmp_l_alpha, tmp_r_alpha) for key in tmp_eq_dict.keys()}
                    )
                    self.l_alpha.extend([tmp_l_alpha] * len(tmp_A0))
                    self.r_alpha.extend([tmp_r_alpha] * len(tmp_A0))
                self.reduced_mech_eq_dict = {key: self.reduced_mech_eq_dict[key] for key in self.eq_dict.keys()}
                self.reduced_mech_A0 = eq_dict2Alist(self.reduced_mech_eq_dict)
                self.GenAPARTDataLogger.info(f'Remains {len(self.eq_dict)} reactions in eq_dict')
            case 'locally_IDT_sensitivity':  
                self.l_alpha, self.r_alpha = [], []; self.alpha_dict = {}; self.eq_dict = {}
                IDTsensitivity_json_path = f"./data/APART_data/reduced_data/IDT_sensitivity_circ={self.circ}.json"  

                # 计算 original_eq_dict 中所有反应关于 IDT 的敏感度
                if not os.path.exists(IDTsensitivity_json_path):
                    IDT_sensitivity = yaml2IDT_sensitivity_Multiprocess(
                                target_chem,
                                IDT_condition = self.IDT_condition,
                                fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                                mode = self.IDT_mode,
                                specific_reactions = list(original_eq_dict.keys()),
                                cut_time = 5*self.true_idt_data,
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
                # 根据 IDT_sensitivity 中的 value 对 original_eq_dict 进行分组: value < 1e-4, 1e-4 < value < 1e-2, 1e-2 < value < 1e-1, 1e-1 < value < 1
                reactions_group = [
                        [k for k, v in IDT_sensitivity.items() if 1e-1 <= v < 1 + 1e-3],
                        [k for k, v in IDT_sensitivity.items() if 1e-2 <= v < 1e-1],
                        [k for k, v in IDT_sensitivity.items() if v <= 1e-2],
                    ]
                # 读取 APART_args 中三组 l_alpha 与 r_alpha 数值: 
                # IDTse_last_l_alpha, IDTse_last_r_alpha 用于统一赋值给第一组 reactions_group[-1]
                # IDTse_miden_l_alpha, IDTse_miden_r_alpha 用于给第二组 reactions_group[-2] 作为最大调整范围
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
                l_alpha_dict = {key: self.alpha_dict[key][0] for key in self.eq_dict.keys()}
                r_alpha_dict = {key: self.alpha_dict[key][1] for key in self.eq_dict.keys()}
                self.l_alpha = eq_dict_broadcast2Alist(l_alpha_dict, self.eq_dict)
                self.r_alpha = eq_dict_broadcast2Alist(r_alpha_dict, self.eq_dict)
            case 'Scale_Sensitivity':
                self.eq_dict = eq_dict
                A0 = eq_dict2Alist(eq_dict)
                M = np.amax(np.abs(self.true_idt_data - self.reduced_idt_data))
                print(f"Current CIRC M is {self.true_idt_data}, {self.reduced_idt_data}")
                Delta = self.APART_args.get('sensitivity_scale_coeff', 0.2)
                original_eq_dict = copy.deepcopy(self.eq_dict)
                IDTsensitivity_json_path = f"./data/APART_data/reduced_data/IDT_sensitivity_circ={self.circ}.json"  

                # 计算 original_eq_dict 中所有反应关于 IDT 的敏感度
                if not os.path.exists(IDTsensitivity_json_path):
                    IDT_sensitivity = yaml2IDT_sensitivity_Multiprocess(
                                target_chem,
                                IDT_condition = self.IDT_condition,
                                fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                                mode = self.IDT_mode,
                                specific_reactions = list(original_eq_dict.keys()),
                                cut_time = 5*self.true_idt_data,
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
                distinctive_func = lambda x: np.amin([np.log10(x / M**3 + 1), np.log10(M / x + 1)]) * Delta
                alpha_dict = {k: distinctive_func(v) for k, v in IDT_sensitivity.items()}
                # alpha_dict = dict(sorted(alpha_dict.items(), key=lambda x: x[1], reverse=True))
                # 去除 self.eq_dict 中不存在的 key
                IDT_sensitivity = {k: v for k, v in IDT_sensitivity.items() if k in original_eq_dict.keys()}
                
                # 如果 APART_args 中存在 l_alpha 和 r_alpha 则使用这两者作为采样范围的最大阈值
                if 'l_alpha' in self.APART_args:
                    # alpha = np.abs(self.APART_args['l_alpha'])
                    alpha = np.abs(self.APART_args['l_alpha'][self.circ]) if isinstance(self.APART_args['l_alpha'], Iterable) else abs(self.APART_args['l_alpha'])                    
                    alpha_dict = {k: np.amin([v, alpha]) for k, v in alpha_dict.items()}
                if 'r_alpha' in self.APART_args:
                    # alpha = np.abs(self.APART_args['r_alpha'])
                    alpha = np.abs(self.APART_args['r_alpha'][self.circ]) if isinstance(self.APART_args['r_alpha'], Iterable) else abs(self.APART_args['r_alpha'])
                    alpha_dict = {k: np.amin([v, alpha]) for k, v in alpha_dict.items()}
                 
                    
                self.alpha_dict = {
                    key: [alpha_dict[key] * -1, alpha_dict[key] * 1] for key in alpha_dict.keys()
                }
                l_alpha_dict = {key: self.alpha_dict[key][0] for key in original_eq_dict.keys()}
                r_alpha_dict = {key: self.alpha_dict[key][1] for key in original_eq_dict.keys()}
                self.l_alpha = eq_dict_broadcast2Alist(l_alpha_dict, self.eq_dict)
                self.r_alpha = eq_dict_broadcast2Alist(r_alpha_dict, self.eq_dict)
        self.A0 = eq_dict2Alist(self.eq_dict)  
        self.gen_yaml:Callable = partial(Adict2yaml, eq_dict = self.eq_dict)
        self.APART_args['eq_dict'] = self.eq_dict
        self.l_alpha = np.array(self.l_alpha); self.r_alpha = np.array(self.r_alpha)
        
        # 调用 cal_samples_quality 来修正
        sample_shrink = self.cal_samples_quality(
            Alist_L2_benchmark=self.A0,
            Alist_L2penalty = 0.001
        )
        self.GenAPARTDataLogger.info(f"Shrink the sample range by {sample_shrink}")
        self.l_alpha = self.l_alpha * sample_shrink; self.r_alpha = self.r_alpha * sample_shrink
        
        self.GenAPARTDataLogger.info(f"Current CIRC alphas are {np.unique(self.l_alpha)} ~ {np.unique(self.r_alpha)}")
        write_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={self.circ}.json", self.eq_dict, cover = True)
        write_json_data(f"./data/APART_data/reduced_data/alpha_dict_circ={self.circ}.json", self.alpha_dict, cover = True)
        self.WriteCurrentAPART_args(eq_dict = self.eq_dict, l_alpha = self.l_alpha, r_alpha = self.r_alpha)


    def GenASampleThreshold(self, best_chem_IDT, threshold_expand_factor = None, **kwargs):
        """
        生成采样的阈值; 根据 best_chem_IDT 生成采样阈值. 需要在 init 中就启动
        生成的具体策略为: 
            根据 best_chem_IDT 中的每一项，计算其与真实值的误差 Rlos，乘以系数 threshold_expand_factor 后，
            以此作为 IDT采样阈值；同时，如果 thresholds 是一个列表，将阈值设置为
            Rlos * threshold_expand_factor * thresholds[i] / thresholds[i - 1]
        add
            self.idt_threshold
        """
        idt_threshold = self.APART_args['idt_threshold']
        idt_threshold = np.array(idt_threshold)[self.circ - 1] if isinstance(idt_threshold, Iterable) else idt_threshold
        threshold_expand_factor = self.APART_args.get('threshold_expand_factor', 1.5) if threshold_expand_factor is None else threshold_expand_factor

        idt_Rlos = 10 ** np.amax(np.abs(np.log10(best_chem_IDT) - np.log10(self.true_idt_data)))  
        self.GenAPARTDataLogger.info(f"Current CIRC Rlos are IDT:{idt_Rlos}")
        if self.circ >= 1:
            if isinstance(idt_threshold, Iterable):
                self.idt_threshold = min(idt_Rlos * threshold_expand_factor * idt_threshold[self.circ] / idt_threshold[self.circ - 1], idt_threshold)
            else:
                self.idt_threshold = min(idt_Rlos * threshold_expand_factor, idt_threshold)
        else:
            idt_threshold = np.array(idt_threshold)[0] if isinstance(idt_threshold, Iterable) else idt_threshold
            self.idt_threshold = min(idt_Rlos * threshold_expand_factor, idt_threshold)

        self.GenAPARTDataLogger.info(f"Current CIRC thresholds are fixed as {np.log10(self.idt_threshold)}(np.log10)")
     

    def SortALIST(self, apart_data_path = None, experiment_time = 50, cluster_ratio = False, 
                  IDT_reduced_threshold = None, SortALIST_T_threshold:float = None, SortALIST_T_threshold_ratio:list|float = None,
                  need_idt = False, return_all = False, father_sample_save_path = None, 
                  logger = None, ord = 2, **kwargs) -> np.ndarray:
        """
        从 apart_data.npz 中筛选出来最接近真实 IDT 的采样点，以反问题权重作为 IDT 的筛选权重
        增加一个限制： 筛选出的结果到真实值的距离不能比 Reduced 结果差大于 IDT_reduced_threshold
        1. 以此作为反问题初值; 2. 用于 ASample 筛选
        params:
            apart_data_path: 输入 apart_data.npz
            experiment_time: 最后返回的列表大小
            cluster_ratio: 是否使用聚类模式; 如果为 int 类型，则表示聚类中初始点的数量
            father_sample_save_path: 保存 father_sample 的路径
            SortALIST_T_threshold: 筛选出的结果中，温度误差不能高于 SortALIST_T_threshold
            SortALIST_T_threshold_ratio: 筛选出的结果中，
                        温度误差不能高于 SortALIST_T_threshold_ratio * self.Temperature_Diff; 优先级高于 SortALIST_T_threshold
        """
        if apart_data_path is None: apart_data_path = self.apart_data_path
        if logger is None: logger = self.GenAPARTDataLogger
        apart_data = np.load(apart_data_path)
        apart_data_idt = apart_data['all_idt_data']; Alist = apart_data['Alist']; apart_data_T = apart_data['all_T_data']

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

        # 计算 DIFF_idt
        diff_idt = np.linalg.norm(
            self.IDT_weight * (np.log10(apart_data_idt) - true_idt),
            axis = 1, ord = ord)
        
        diff_idt = diff_idt[if_idt_pass]; Alist = Alist[if_idt_pass, :]; apart_data_idt = apart_data_idt[if_idt_pass, :]
        index = np.argsort(diff_idt); Alist = Alist[index,:]
        apart_data_idt = apart_data_idt[index,:]; apart_data_T = apart_data['all_T_data'][index,:]

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
           np.savez(father_sample_save_path, Alist = Alist[0:experiment_time, :], IDT = apart_data_idt[0:experiment_time, :], T = apart_data_T[0:experiment_time, :]) 
        if need_idt:
            return Alist[0:experiment_time, :], apart_data_idt[0:experiment_time, :]
        if return_all:
            return Alist[0:experiment_time, :], apart_data_idt[0:experiment_time, :], apart_data_T[0:experiment_time, :]
        else:
            return Alist[0:experiment_time, :]
        

    def ASample(self, sample_size = None, coreAlist = None, passing_rate_upper_limit = None, 
                IDT_reduced_threshold = None, shrink_delta = None, cluster_ratio = False, father_sample_save_path = None, 
                start_circ = 0, shrink_ratio = None, average_timeout_time = 0.072, sampling_expand_factor = None, **kwargs):
        """
        重写了 APART.ASample_IDT 函数，主要是为了增加针对不同反应调节采样范围的函数a
        202230730: 增加了通过率上限 passing_rate_upper_limit,
        """
        np.set_printoptions(precision = 2, suppress = True); t0 = time.time()

        # 预设置
        self.GenAPARTDataLogger.info(f"Start The ASample Process; Here we apply three aspect into consideration: IDT: True")
        # 提取采样的左右界限 + 采样阈值

        # 检测类中是否存在 self.idt_threshold
        if not hasattr(self, 'idt_threshold'):
            idt_threshold = self.APART_args['idt_threshold']
            self.idt_threshold = np.array(idt_threshold)[self.circ - 1] if isinstance(idt_threshold, Iterable) else idt_threshold

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

        # 将YAML文件的A值提取并构建均匀采样点
        if self.circ == 0 or self.circ == start_circ:
            self.samples = sample_constant_A(sample_size, self.reduced_mech_A0, self.l_alpha, self.r_alpha)
            if shrink_strategy:
                self.boarder_samples = sample_constant_A(int(sample_size * shrink_ratio), self.reduced_mech_A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha),)
        else:
            # json_data = read_json_data(self.model_previous_json)
            net = load_best_dnn(Network_PlainSingleHead, self.model_previous_json, device = 'cpu')
            self.GenAPARTDataLogger.info(f"idt_threshold: log10({self.idt_threshold}) = {np.log10(self.idt_threshold)}; sample_size: {sample_size}; father sample size: {core_size}")
            self.GenAPARTDataLogger.info(f"shrink_delta: {shrink_delta}; shrink_strategy: {shrink_strategy}")
            self.GenAPARTDataLogger.info(f"IDT_reduced_threshold: {IDT_reduced_threshold}; passing_rate_upper_limit: {passing_rate_upper_limit}")
            self.GenAPARTDataLogger.info("="*100)

            # 读取之前的 apart_data.npz 从中选择前 1% 的最优采样点作为核心; 依然选择 IDT 作为指标，不涉及 

            if coreAlist is None:
                cluster_weight = kwargs.get('cluster_weight', 0.1)
                previous_coreAlist = self.SortALIST(
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
                
            t0 = time.time(); idt_zero_father_sample_times = 0
            self.samples = []; self.boarder_samples = []; tmp_sample_size = int(2 * (sample_size) // core_size); tmp_sample = []  
            while len(self.samples) < sample_size:
                # 每次采样的样本点不能太少，也不能太多；因此创建自适应调节机制
                if len(tmp_sample) >= sample_size * 0.02:
                    tmp_sample_size = int(sampling_expand_factor * (sample_size - len(self.samples)) // core_size)
                else:
                    tmp_sample_size = int(sampling_expand_factor * sample_size // core_size)
                for A0 in coreAlist:
                    self.GenAPARTDataLogger.info(f"tmp_sample_size in this circle is {tmp_sample_size}")
                    try:  
                        tmp_sample, idt_pred_data = SampleAWithNet(net.forward, np.log10(self.true_idt_data), threshold = np.log10(self.idt_threshold), 
                                                    size = tmp_sample_size, A0 = A0, l_alpha = self.l_alpha, passing_rate_upper_limit = np.sqrt(passing_rate_upper_limit),
                                                    r_alpha = self.r_alpha, save_path = None, debug = True, reduced_data = np.log10(self.reduced_idt_data), reduced_threshold = None)
                        if shrink_strategy:
                            tmp_boarder_sample = SampleAWithNet(net.forward, np.log10(self.true_idt_data), threshold = np.log10(self.idt_threshold) + 1, 
                                                    size = int(tmp_sample_size * shrink_ratio), A0 = A0, l_alpha = ((1 + shrink_delta) * self.l_alpha, (1 - shrink_delta) * self.l_alpha), 
                                                    r_alpha = ((1 - shrink_delta) * self.r_alpha, (1 + shrink_delta) * self.r_alpha), 
                                                    reduced_data = np.log10(self.reduced_idt_data), reduced_threshold = None, debug = False)
                        
                        idt_Rlos = np.abs(np.mean(idt_pred_data - np.log10(self.true_idt_data), axis = 0))
                        self.GenAPARTDataLogger.info(f"IDT: On Average, Working Condition Index which not satisfy threshold IS {np.where(idt_Rlos > np.log10(self.idt_threshold))[0]}")
                        self.GenAPARTDataLogger.info(f"IDT: On Average, Working Condition which not satisfy threshold IS {self.IDT_condition[np.where(idt_Rlos > np.log10(self.idt_threshold))[0],:]}")
                        self.GenAPARTDataLogger.info(f"true IDT is {np.log10(self.true_idt_data)}")
                        self.GenAPARTDataLogger.info(f"First element of sample prediction IDT is {idt_pred_data[0]}")
                        self.GenAPARTDataLogger.info(f"Abs Difference between true IDT and sample prediction is {idt_Rlos}")
                        self.GenAPARTDataLogger.info(f"IDT: Remain sample size is {len(tmp_sample)}")
                        self.GenAPARTDataLogger.info("-·"*100)
                        if len(tmp_sample) == 0:
                            raise ValueError("tmp_sample is Empty!")
                    except ValueError or IndexError:
                        traceback.print_exc()
                        self.GenAPARTDataLogger.warning(f"IDT: tmp_sample is Empty in This circle! idt_zero_father_sample_times = {idt_zero_father_sample_times}")
                        self.GenAPARTDataLogger.info("-·"*100)
                        idt_zero_father_sample_times += 1
                        continue

                    self.samples.extend(tmp_sample.tolist())
                    pass_rate = len(tmp_sample) / tmp_sample_size
                    # pass_rate 太高调小 idt_threshold
                    if pass_rate ** 1/2 > passing_rate_upper_limit * 0.9 and pass_rate  ** 1/2 > 0.5:
                        self.idt_threshold = self.idt_threshold * 0.8
                        self.GenAPARTDataLogger.warning(f"IDT + : pass_rate is {pass_rate * 100} %, which is too high! idt_threshold have changed to log10({self.idt_threshold}) = {np.log10(self.idt_threshold)}")
                    
                    # 样本太多立即退出
                    if len(self.samples) >= sample_size * 1.2:
                        self.GenAPARTDataLogger.warning(f"IDT: Stop the Sampling Process, Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                        self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                        break

                # 时间过长退出机制
                if time.time() - t0 > sample_size * average_timeout_time:
                    self.GenAPARTDataLogger.warning("Error: Function Asample has timed out!")
                    break
                self.GenAPARTDataLogger.info(f"In this Iteration, IDT + : Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                # 如果过长时间无法筛选到合适的点，稍微拓宽采样的 threshold
                if idt_zero_father_sample_times > 10 * len(coreAlist):
                    self.idt_threshold = 10 ** self.idt_threshold
                    self.GenAPARTDataLogger.warning(f"IDT: idt_threshold have changed to log10{self.idt_threshold} = {np.log10(self.idt_threshold)}")
                    idt_zero_father_sample_times = 0
        
        self.samples = np.array(self.samples)
        # 截取 samples 前 sample_size 个点
        self.samples = self.samples[0:sample_size, :]
        self.father_samples = coreAlist
        if len(self.samples) == 0:
            raise ValueError("self.samples is Empty! End the whole process! Please check the threshold!")
        np.save(f'./data/APART_data/Asamples_{self.circ}.npy', self.samples)  
        # 更新 idt_threshold
        self.APART_args['idt_threshold'] = self.idt_threshold
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


    def SAAdaptWC4Asample(self, range_phi, range_T, range_P, best_sample = None, delta_range_phi = 0, delta_range_T = 0, 
                     delta_range_P = 0, conserve_wc = None, save_path = None, save_nums = None,
                     **kwargs):
        """
        根据筛选中选择的父样本点来进行自适应调整工况，调整方法是调用“自适应选择工况”函数，所有参数详见
        SFAdjust_IDT_condition 函数中的说明

        暂时只考虑第一个父样本点 best_sample

        adjust: self.IDT_condition
        """ 
        np.set_printoptions(suppress=True)
        self.GenAPARTDataLogger.info("="*100)
        self.GenAPARTDataLogger.info(f"Old IDT condition is {self.IDT_condition}")
        self.GenAPARTDataLogger.info(f"Old True IDT data is {np.log10(self.true_idt_data)}")
        self.GenAPARTDataLogger.info("Start AdaptWC4Asample Progress...")

        best_sample = self.best_sample if best_sample is None else best_sample
        if self.circ == 0:
            self.SFAdjust_IDT_condition(
                     range_phi, range_T, range_P, delta_range_phi = delta_range_phi, delta_range_T = delta_range_T, 
                     delta_range_P = delta_range_P, conserve_wc = conserve_wc, save_path = save_path, save_nums = save_nums,
                     **kwargs
            )
        else:
            coreA_yaml = f"./data/APART_data/best_sample_circ={self.circ}.yaml"
            Adict2yaml(self.reduced_mech, coreA_yaml, self.eq_dict, best_sample)
            self.SFAdjust_IDT_condition(
                     range_phi, range_T, range_P, optim_chem = coreA_yaml, delta_range_phi = delta_range_phi, delta_range_T = delta_range_T, 
                     delta_range_P = delta_range_P, conserve_wc = conserve_wc, save_path = save_path, save_nums = save_nums,
                     **kwargs
            )
            
        # 更新 APART_args['IDT_condition'] 和 json 文件
        self.APART_args['IDT_condition'] = self.IDT_condition
        args = self.APART_args.copy()
        for arg in args:
            if isinstance(args[arg], np.ndarray): args[arg] = args[arg].tolist()
        write_json_data(self.model_current_json, args, cover = True)

        self.GenAPARTDataLogger.info(f"New IDT_condition is {self.IDT_condition}")
        # 更新 self.true_idt_data
        self.LoadTrueData(self.true_data_path + '/true_idt.npz', mode = 'idt', refresh = True)
        self.GenAPARTDataLogger.info(f"New True IDT data is {np.log10(self.true_idt_data)}")
        self.GenAPARTDataLogger.info(f"New Reduced IDT data is {np.log10(self.reduced_idt_data)}")
        self.GenAPARTDataLogger.info("End AdaptWC4Asample Progress.")
        self.GenAPARTDataLogger.info("="*100)


    def GridAdaptWC4Asample(self, grid_T, grid_P, grid_phi, best_sample = None, Tlim:tuple = None, Plim:tuple = None, philim:tuple = None,
                             delta_range_phi = 0, delta_range_T = 0, delta_range_P = 0, conserve_wc = None, save_path = None, 
                             save_nums = None, cpu_nums = 0, **kwargs):
        """
        根据筛选中选择的父样本点来进行自适应调整工况，调整方法是调用“自适应选择工况”函数，所有参数详见
        self.GridAdjust_IDT_condition 函数中的说明

        暂时只考虑第一个父样本点 best_sample

        adjust: self.IDT_condition
        """ 
        np.set_printoptions(suppress=True)
        self.GenAPARTDataLogger.info("="*100)
        self.GenAPARTDataLogger.info(f"Old IDT condition is {self.IDT_condition}")
        self.GenAPARTDataLogger.info(f"Old True IDT data is {np.log10(self.true_idt_data)}")
        self.GenAPARTDataLogger.info(f"Multiprocess Cpus are {cpu_nums}, while cpu_nums = 0 means using all cpus")
        self.GenAPARTDataLogger.info("Start AdaptWC4Asample Progress...")

        best_sample = (self.reduced_mech_A0 if self.circ == 0 else self.best_sample) if best_sample is None else best_sample
        print(f"grid_T = {grid_T}, grid_P = {grid_P}, grid_phi = {grid_phi}")
        grid_P = int(grid_P); grid_T = int(grid_T); grid_phi = int(grid_phi)
        if self.circ == 0:
            self.GridAdjust_IDT_condition(
                     grid_T, grid_P, grid_phi, Tlim = Tlim, Plim = Plim, philim = philim, delta_range_phi = delta_range_phi, delta_range_T = delta_range_T, 
                     delta_range_P = delta_range_P, conserve_wc = conserve_wc, save_path = save_path, save_nums = save_nums,
                     cut_time = self.idt_cut_time, logger = self.GenAPARTDataLogger, cpu_nums = cpu_nums, **kwargs
            )
        else:
            coreA_yaml = f"./data/APART_data/best_sample_circ={self.circ}.yaml"
            Adict2yaml(self.reduced_mech, coreA_yaml, self.eq_dict, best_sample)
            self.GridAdjust_IDT_condition(
                     grid_T, grid_P, grid_phi, optim_chem = coreA_yaml, Tlim = Tlim, Plim = Plim, 
                     philim = philim, delta_range_phi = delta_range_phi, delta_range_T = delta_range_T, 
                     delta_range_P = delta_range_P, conserve_wc = conserve_wc, save_path = save_path, save_nums = save_nums,
                     cut_time = self.idt_cut_time, logger = self.GenAPARTDataLogger, cpu_nums = cpu_nums, **kwargs
            )
            
        # 更新 APART_args['IDT_condition'] 和 json 文件
        self.APART_args['IDT_condition'] = self.IDT_condition
        self.WriteCurrentAPART_args(IDT_condition = self.IDT_condition)

        self.GenAPARTDataLogger.info(f"New IDT_condition with len {len(self.APART_args['IDT_condition'])} is {self.APART_args['IDT_condition']}")
        # 更新 self.true_idt_data
        self.LoadTrueData(self.true_data_path + '/true_idt.npz', mode = 'idt', refresh = True)
        self.GenAPARTDataLogger.info(f"New True IDT data is {np.log10(self.true_idt_data)}")
        self.GenAPARTDataLogger.info(f"New Reduced IDT data is {np.log10(self.reduced_idt_data)}")
        self.GenAPARTDataLogger.info("End AdaptWC4Asample Progress.")
        self.GenAPARTDataLogger.info("="*100)


    def SegmentationNetworkGenerator(self, network_json_list: list, best_sample: np.ndarray, l_alpha_list: list = None,
                                     r_alpha_list:list = None, current_circ = None, **kwargs):
        """
        用于 Asample 中的分段网络函数生成器，最后生成一个 Callable 对象 segmentation_network
        segementation_network 的作用是将输入的 A 值判断其应该使用哪一个网络，加载该网络并返回 A 的预测值
        params:
            network_json_list: list, 每一个元素是一个网络的 json 文件路径
            best_sample: np.ndarray, 单个最优样本点
            l_alpha_list: list, 每一个元素是一个网络的左边界范围值
            r_alpha_list: list, 每一个元素是一个网络的右边界范围值
            current_circ: int, 当前的循环次数
        return:
            segmentation_network: Callable, 输入 A 值，返回预测值
        """
        network_nums = len(network_json_list)
        current_circ = self.circ if current_circ is None else current_circ
        if l_alpha_list is None: l_alpha_list = self.APART_args['l_alpha'][current_circ - network_nums:current_circ]
        if r_alpha_list is None: r_alpha_list = self.APART_args['r_alpha'][current_circ - network_nums:current_circ]
        network_list = [
            load_best_dnn(Network_PlainSingleHead, network_json_list[i], device = 'cpu') for i in range(network_nums)
        ]
        def segmentation_network(A: float):
            """
            输入 A 值，返回预测值
            """
            result = None
            for i in range(network_nums):
                # 判断 A - best_sample 的每个分量都在哪个网络的范围内
                if (l_alpha_list[i] <= A - best_sample[i] <= r_alpha_list[i]).all():
                    result = network_list[i](torch.tensor(A, dtype = torch.float32))
                    break
            if result is None:
                result = network_list[-1](torch.tensor(A, dtype = torch.float32))
            return result.detach().numpy()
        return segmentation_network


    def gather_apart_data(self, save_path = "./data/APART_data/tmp", save_file_name = None,  rm_tree = True, 
                          cover = True, logger = None, **kwargs):
        """
        从 save_path 中的所有 npz 文件中提取 IDT, T, Alist 数据，汇总到 save_file_name 中
        预计被 APART_base.GatherAPARTData 函数替代
        params:
            save_path: str, 保存的文件夹路径
            save_file_name: str, 保存的文件名
            rm_tree: bool, 是否删除 save_path 中的所有文件
            cover: bool, 是否覆盖 save_file_name 文件
            logger: Log, 日志记录器
        return:
            all_idt_data: np.array, 汇总的 IDT 数据
            all_T_data: np.array, 汇总的 T 数据
            Alist_data: np.array, 汇总的 Alist 数据
        """
        t0 = time.time()
        save_file_name = self.apart_data_path if save_file_name is None else save_file_name
        files = [file for file in os.listdir(save_path) if file.find('.npz') != -1]
        filenums = len(files); logger = self.TrainAnetLogger if logger is None else logger
        if cover or not os.path.exists(save_file_name) or os.path.getsize(save_file_name) /1024/1024 <= 1:
            all_idt_data, all_T_data, Alist_data = [], [], [] # 汇总数据
            for target_file in files:
                target_file = os.path.join(save_path, target_file)
                if os.path.getsize(target_file) >= 5:
                    tmp = np.load(target_file)
                    if len(tmp['Alist']) == 0: break
                    all_idt_data.append(tmp['IDT'].tolist());   all_T_data.append(tmp['T'].tolist())
                    Alist_data.append(tmp['Alist'].tolist()); 
                    if len(Alist_data) % 100 == 0:
                        logger.info(f"Cost {time.time() - t0:.1f}, Gather Data Process has finished {len(Alist_data)/filenums * 100:.2f} %")

            # 保存数据
            np.savez(save_file_name,
                        all_idt_data = all_idt_data, all_T_data = all_T_data, Alist = Alist_data)
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
            

    """==============================================================================================================="""
    """                                               GenDataFunc                                                     """
    """==============================================================================================================="""
    

    # def GenIDT_POOL(self, samples:np.ndarray = None, idt_cut_time_alpha = 1.5, save_path = "./data/APART_data/tmp",
    #                    **kwargs):
    #     """
    #     使用多进程 POOL 运行 GenOneDataIDT
    #     args:
    #         samples: 默认为 None, 指定其他则会替换

    #     kwargs:     
    #         cpu_process: 默认为 APART_args['cpu_process']
    #         ignore_error_path: 保存出错样本点的路径, 若为 None 不保存

    #     return:
    #         None, 但是保存了一个文件 f'./data/APART_data/apart_data_circ={self.circ}.npz'
    #         这个文件里面的 key 为： aall_idt_data, all_T_data,  all_Alist
    #         每个 key 对应一个 list

    #     """
    #     cpu_process = kwargs.get('cpu_process', self.APART_args['cpu_process'])
    #     mkdirplus(save_path)
    #     ignore_error_path = kwargs.get("ignore_error_path", None)

    #     RES = [] # 收集出错样本
    #     def callback(status):
    #         if not status is None:
    #             RES.append(status)
    #     def error_callback(error):
    #         # 发生故障的 call back 函数
    #         exstr = traceback.format_exc()
    #         print(f"Error occur! the error is {error}")
    #         print(f"Detail: {exstr}")

    #     self.GenAPARTDataLogger.info(f"Start the GenIDT_POOL at circ {self.circ}")
        
    #     p = Pool(cpu_process)
    #     for index in range(np.size(samples, 0)):
    #         p.apply_async(func = GenOneDataIDT, 
    #                         kwds = dict(
    #                         index = index,
    #                         IDT_condition = self.IDT_condition,
    #                         Alist = samples[index],
    #                         eq_dict = self.eq_dict,
    #                         fuel = self.IDT_fuel,
    #                         oxidizer = self.IDT_oxidizer,
    #                         reduced_mech = self.reduced_mech,
    #                         my_logger = self.GenAPARTDataLogger,
    #                         IDT_mode = self.IDT_mode,
    #                         # 将 idt_arrays 采取如下设置: 其每个分量是 true_idt_data 和 reduced_idt_data 对应分量中大的那个
    #                         idt_arrays = np.maximum(self.true_idt_data, self.reduced_idt_data),
    #                         cut_time_alpha = idt_cut_time_alpha,
    #                         **kwargs
    #                         ),
    #                       callback = callback, 
    #                       error_callback = error_callback,
    #                       )
    #     p.close();  p.join()

        
    #     if not ignore_error_path is None:
    #         np.save(ignore_error_path, np.array(RES))   


    def GenDataFuture(self, samples:np.ndarray = None, idt_cut_time_alpha = 1.5, save_path = "./data/APART_data/tmp",
                       ignore_error_path = None, cpu_process = None, start_sample_index = 0, **kwargs):
        """
        使用 future 模块的 ProcessPoolExecutor 生成数据; 
        start_sample_index: 开始的样本点的编号; 适用于重复采样的情况
        save_path: 保存普通样本点的路径
        ignore_error_path: 保存出错样本点的路径, 若为 None 不保存
        return:
            生成失败的样本点个数
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
                            GenOneDataIDT,
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
    

    """==============================================================================================================="""
    """                                               DNN                                                             """
    """==============================================================================================================="""
    

    def DeePMO_train(self, concat_pre = False,  rm_tree = True, 
                           shrink_strategy = None, extract_strategy = None, 
                           rate = 0.8, device = None, load_file_name = None, 
                           one_data_save_path = "./data/APART_data/tmp", 
                           net_data_dirpath = "./data/APART_data/ANET_data", lock_json = False, **kwargs):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                
            else:
                device = torch.device("cpu")
        self.TrainAnetLogger.info(f"DeePMO_train: Device is {device}")
        self.GenAPARTDataLogger.info(f"Device is {device}")
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
        self._DeePMO_train(device = device)
        self.APART_args['IDT_condition'] = self.APART_args['IDT_condition'].tolist()
        
        # 保存DNN的超参数到JSON文件中
        self.WriteCurrentAPART_args(cover = True)
        if lock_json: 
            subprocess.run(f"chmod 444 {self.model_current_json}", shell = True)
        tmp_data = np.load(f"{self.model_loss_path}/APART_loss_his_circ={self.circ}.npz")
        train_his1 = tmp_data['train_his'];test_his1 = tmp_data['test_his']; stop_index = tmp_data['stop_index']
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.semilogy(train_his1, lw=1, label='IDTtrain')
        ax.semilogy(test_his1, 'r', lw=1.2, label='IDTtest')
        ax.axvline(stop_index, label = 'early stopping', color = 'green', )
        ax.set_xlabel('epoch'); 
        ax.set_ylabel('loss (log scale)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/loss_his_circ={self.circ}.png')
        plt.close(fig)

        self.TrainAnetLogger.info(f"Finished Train DNN! Total cost {time.time() - t0:.2f} s")


    def _TrainDataProcess(self, load_file_name = None,
                              concat_pre = False,  rm_tree = True, 
                              shrink_strategy = False, extract_strategy = False,
                              rate = 0.8, device = 'cuda', one_data_save_path = "./data/APART_data/tmp", 
                              net_data_dirpath = "./data/APART_data/ANET_data", **kwargs) -> None:
        """
        用于 DeePMO_train 中的数据处理部分
        params:
            load_file_name: str | list, 默认为 None, 若为 None 则使用 self.apart_data_path
            concat_pre: bool, 默认为 False, 若为 True 则将之前的数据集和当前的数据集进行拼接
            rm_tree: bool, 默认为 True, 若为 True 则删除中间文件
            shrink_strategy: bool, 默认为 False, 若为 True 则使用缩减策略
            extract_strategy: bool, 默认为 False, 若为 True 则使用抽取策略
            rate: float, 默认为 0.8, 训练集的比例
            device: str, 默认为 'cuda', 训练的设备
            one_data_save_path: str, 默认为 "./data/APART_data/tmp", 用于存储单个数据集的路径
            net_data_dirpath: str, 默认为 "./data/APART_data/ANET_data", 用于存储网络数据集的路径
            kwargs:
                extract_rate: float, 默认为 0.05, 抽取策略的抽取比例
        return:
            None
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
            all_T_data = Data['all_T_data']
            # 判断是否加载之前的 data:
        else:
            all_idt_data, all_T_data, Alist_data = [], [], []
            for file in load_file_name:   
                Data = np.load(file)
                Alist_data.extend(Data['Alist']); all_idt_data.extend(Data['all_idt_data']); 
                all_T_data.extend(Data['all_T_data'])
        
        # 进入抽取模式，抽取之前数据集中表现最优秀的前 5% 样本点
        if extract_strategy and self.circ != 0:
            extract_rate = kwargs.get('extract_rate', 0.05)
            self.TrainAnetLogger.info(f'Extracting DNN Data From Previous Best Sample..., extract_rate = {extract_rate}')
            for circ in range(self.circ):
                apart_data_path = os.path.dirname(self.apart_data_path) + f"/apart_data_circ={circ}.npz"
                if os.path.exists(apart_data_path) and os.path.getsize(apart_data_path) /1024 /1024 > 1:
                    Alist, IDT, T = self.SortALIST(
                        apart_data_path = apart_data_path,
                        experiment_time = extract_rate,
                        return_all = True,
                        logger = self.TrainAnetLogger
                    )
                    all_idt_data.extend(IDT); all_T_data.extend(T); Alist_data.extend(Alist)
                else:
                    self.TrainAnetLogger.warning(f"Extracting: Can't find {apart_data_path} at circ = {circ}, skip")
        
        # 转化为 ndarray 加快转化 tensor 速度 + IDT  对数处理 处理
        all_idt_data, all_T_data, Alist_data  = \
             np.array(all_idt_data), np.array(all_T_data), np.array(Alist_data)
        all_idt_data = np.log10(all_idt_data)
        # 如果 Alist_data 是 1 维数组，提示错误：检查 Alist 的长度是否一样
        if len(Alist_data.shape) == 1: 
            raise ValueError(f"Alist_data is 1D array, check the length of Alist_data is same or not")

        self.TrainAnetLogger.info(f"The size of all_idt_data is {all_idt_data.shape}, all_T_data is {all_T_data.shape}, Alist_data is {Alist_data.shape}")

        # 只保留 sample size 个数据点
        expected_sample_size = self.APART_args['sample_size'] if isinstance(self.APART_args['sample_size'], int) else self.APART_args['sample_size'][self.circ]
        if all_idt_data.shape[0] > expected_sample_size:
            all_idt_data, all_T_data, Alist_data = \
                all_idt_data[:expected_sample_size], all_T_data[:expected_sample_size], \
                Alist_data[:expected_sample_size]

        assert all_idt_data.shape[0] > 0 and all_T_data.shape[0] > 0 and Alist_data.shape[0] > 0, \
            f"Error! all_idt_data.shape = {all_idt_data.shape}, all_T_data.shape = {all_T_data.shape}, Alist_data.shape = {Alist_data.shape}"
        
        assert all_idt_data.shape[0] == all_T_data.shape[0] == Alist_data.shape[0], \
            f"Error The shape at axis 0 is not same! all_idt_data.shape = {all_idt_data.shape}, all_T_data.shape = {all_T_data.shape}, Alist_data.shape = {Alist_data.shape}"
        # T data 做 zscore 标准化
        all_T_data, T_mean, T_std = zscore(all_T_data)
        self.APART_args.update({'T_mean': T_mean.tolist(), "T_std": T_std.tolist()})

        self.APART_args['input_dim'], self.APART_args['output_dim'] = Alist_data.shape[1], all_idt_data.shape[1]
        self.APART_args['train_size']  = int(Alist_data.shape[0] * rate)
        self.APART_args['test_size'] = Alist_data.shape[0] - self.APART_args['train_size']

        self.TrainAnetLogger.info(f"The size of train set is {self.APART_args['train_size']}; test set is {self.APART_args['test_size']}")

        dataset = DATASET_SingleHead(
            data_A = torch.tensor(Alist_data, dtype = torch.float32),
            data_QoI = torch.tensor(all_idt_data, dtype = torch.float32),
        )
        train_data, test_data = random_split(dataset
            , lengths = [self.APART_args['train_size'], self.APART_args['test_size']])

        self.train_loader = DataLoader(train_data, 
                                       shuffle = True, 
                                       batch_size = self.APART_args['batch_size'],
                                       drop_last = True)
        
        # 单独写出来 test 的数据
        test_loader = DataLoader(test_data, 
                                shuffle = False, 
                                batch_size = len(test_data),)

        for x_test, idt_test in test_loader:
            self.test_loader = (x_test.to(device), idt_test.to(device))
        
        if not net_data_dirpath is None:
            np.savez(
                net_data_dirpath + f"/ANET_Dataset_train={self.APART_args['train_size']}_circ={self.circ}.npz",
                train_data_x = dataset[train_data.indices][0].cpu().numpy(),
                train_data_y = dataset[train_data.indices][1].cpu().numpy(),
                test_data_x =  dataset[test_data.indices][0].cpu().numpy(),
                test_data_y =  dataset[test_data.indices][1].cpu().numpy(),
            )


    def _DeePMO_train(self, device = None, **kwargs):
        """
        专用于关于 IDT 网络的训练，采取的训练模式是 APART 1.3.1 的 dataloader; 模型是整合在一起的 Network_PlainSingleHead
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        model_pth_path  = self.model_path
        model_loss_path = self.model_loss_path

        t0 = time.time()
        input_dim, output_dim = self.APART_args['input_dim'], self.APART_args['output_dim']
        tmp_save_path = mkdirplus(f'{model_pth_path}/tmp')
        self.TrainAnetLogger.info(f'A -> IDT training has started...; circ = {self.circ}')
        ANET = Network_PlainSingleHead(input_dim, self.APART_args['hidden_units'], output_dim).to(device)
        if self.APART_args['optimizer'] == 'Adam': optimizer = optim.Adam(ANET.parameters(), lr = self.APART_args['learning_rate'], weight_decay = 0.1)
        if self.APART_args['optimizer'] == 'SGD':  optimizer = optim.SGD(ANET.parameters(), lr = self.APART_args['learning_rate'], momentum = 0.1, weight_decay = 0.1)
        # 第一次保存初始模型
        state = {'model':ANET.state_dict()}   
        torch.save(state, f'{model_pth_path}/Network_PlainSingleHead_initialized_circ={self.circ}.pth')  # 保存DNN模型
        
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.APART_args['lr_decay_step'], gamma = self.APART_args['lr_decay_rate'])

        epoch_index, train_his1, test_his1= [], [], []
        for epoch in range(self.APART_args['epoch']):
            # 预测
            with torch.no_grad():
                x_dnn_test, idt_test = self.test_loader
                idt_pred = ANET.forward(x_dnn_test)
                loss1 = criterion(idt_pred, idt_test)
                test_loss = float(loss1.cpu().detach())
            # 训练
            idt_loss  = 0
            for batch_num, (x_train_batch, idt_train_batch) in enumerate(self.train_loader):  # 按照 batch 进行训练
                x_train_batch, idt_train_batch = \
                    x_train_batch.to(device), idt_train_batch.to(device)
                idt_pred = ANET.forward(x_train_batch)
                train_loss_batch = criterion(idt_pred, idt_train_batch)

                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()
                idt_loss += float(train_loss_batch.cpu().detach())
            
            scheduler.step()
            batch_num = len(self.train_loader)
            idt_loss /=  batch_num
            train_his1.append(idt_loss); test_his1.append(test_loss)
            epoch_index.append(epoch);           

            if epoch % 5 == 0:
                GPUtil.showUtilization()
                self.TrainAnetLogger.info(f"epoch: {epoch}\t train loss: {idt_loss:.3e},"+
                                            f"test loss: {test_his1[-1]:.3e},"+
                                            f"time cost: {int(time.time()-t0):.2e} s   lr:{optimizer.param_groups[0]['lr']:.2e}")
            if (epoch == 0) or ((epoch - 25) % 50 == 0):
                state = {'model':ANET.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}   
                torch.save(state, f'{model_pth_path}/tmp/ANET_epoch_{epoch}.pth')  # 保存DNN模型   
        
        train_his1, test_his1 = np.array(train_his1), np.array(test_his1)
        
        # 构建 early stopping 必需的文件夹, 注意这里的 early stopping 是以 IDT 作为基准的，因为相比其他两个指标 IDT 更加重要
        if not os.path.exists(f'{model_pth_path}/early_stopping'): # 创建临时保存网络参数的文件夹
            os.mkdir(f'{model_pth_path}/early_stopping')
        # early stopping 平均 earlystopping_step 的误差求最小
        earlystopping_step = min(self.APART_args['epoch'], 50)

        test_loss_sum = np.sum(test_his1.reshape(-1, earlystopping_step, 1)[...,0], axis = 1)
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
       
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.semilogy(train_his1, lw=1, label='IDTtrain')
        ax.semilogy(test_his1, 'r', lw=1.2, label='IDTtest')
        ax.axvline(stop_index, label = 'early stopping', color = 'green', )
        ax.set_xlabel('epoch'); 
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{self.model_loss_path}/loss_his_circ={self.circ}.png')
        plt.close(fig)

        self.TrainAnetLogger.info(f"Finished Train DNN! Total cost {time.time() - t0:.2f} s")


    def SkipSolveInverse(self, father_sample:str = None, save_dirpath = f'./inverse_skip', 
                             csv_path = None, device = 'cpu', IDT_reduced_threshold = None, raw_data = False, 
                             experiment_time = 15, **kwargs):
        """
        自动跳过 Inverse 部分直接使用最优样本就可以拿到结果，可以直接放在训练步骤之后使用
        """
        np.set_printoptions(suppress=True, precision=3)

        save_folder = mkdirplus(save_dirpath)
        # 加载最优的样本
        if not father_sample is None and os.path.exists(father_sample):
            tmp_father_sample = np.load(father_sample)
            inverse_alist = tmp_father_sample['Alist']
        else:
            inverse_alist = self.SortALIST(self.apart_data_path, experiment_time = experiment_time,
                                    IDT_reduced_threshold = IDT_reduced_threshold,)
        # 加载网络
        optim_net = load_best_dnn(Network_PlainSingleHead, self.model_current_json, device = device)
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
                Alist_data = apart_data['Alist']; idt_data = apart_data['all_idt_data']
                # 查找 Ainit 的 index
                index = np.where(np.all(Alist_data == A_init, axis = 1))[0][0] 
                cantera_idt_data = idt_data[index]
                
                # 简化机理指标 vs 真实机理指标的绘图
                # IDT part
                relative_error = np.mean(np.abs((cantera_idt_data - self.true_idt_data) / self.true_idt_data)) * 100
                log_abs_error = np.mean(np.abs(np.log10(cantera_idt_data) - np.log10(self.true_idt_data)))
                original_log_abs_error = np.mean(np.abs(np.log10(self.true_idt_data) - np.log10(self.reduced_idt_data)))
                self.InverseLogger.info(f"Relative Error is {relative_error} %, Log Abs Error is {log_abs_error:.2f}, Original Log Abs Error is {original_log_abs_error:.2f}")
                # log scale
                true_idt_data = np.log10(self.true_idt_data); cantera_idt_data = np.log10(cantera_idt_data); 
                reduced_idt_data = np.log10(self.reduced_idt_data)
                final_idt = optim_net.forward(torch.tensor(A_init, dtype = torch.float32)).detach().numpy()

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
        
                # 保存 IDT 的相关数据
                np.savez(
                        inverse_path + "/IDT_data.npz",
                        true_idt_data = true_idt_data,
                        reduced_idt_data = reduced_idt_data,
                        cantera_idt_data = cantera_idt_data,
                        dnn_idt_data = np.array(final_idt),
                        Alist = A_init
                        )

                self.InverseLogger.info(f'plot compare idt picture done! time cost:{time.time() - t1} seconds')
            except Exception:
                exstr = traceback.format_exc()
                self.InverseLogger.info(f'!!ERROR:{exstr}')

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
        previous_best_chem_IDT, _ = yaml2idt(
            previous_best_chem, mode = self.IDT_mode, 
            IDT_condition = self.IDT_condition, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
            cut_time = self.idt_cut_time, idt_defined_T_diff = self.idt_defined_T_diff, time_multiple = self.idt_defined_time_multiple
        )
        asamples = np.load(os.path.dirname(apart_data_path) + f"/Asamples_{self.circ}.npy")
        network = load_best_dnn(Network_PlainSingleHead, self.model_current_json, device = 'cpu', )
        IDT_func = lambda x: network.forward(torch.tensor(x, dtype = torch.float32)).detach().numpy()
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

        return dist_A0, dist_center, std_sample


    def CheckDIDT(self,):
        """
        检查真实机理和简化机理的双点火情况，返回两者再所有工况下的时间-温度曲线(同 ValidationIDT) 并显示抓取的双点火位置
        返回工况中的双点火工况和双点火延迟时间
        """
        np.set_printoptions(suppress = True)
        true_didt, _, _ = yaml2didt(self.detail_mech, self.IDT_condition, self.IDT_fuel, self.IDT_oxidizer, didt_default = 0)
        reduced_didt, _, _ = yaml2didt(self.reduced_mech, self.IDT_condition, self.IDT_fuel, self.IDT_oxidizer, didt_default = 0)
        true_didt_condition = self.IDT_condition[np.nonzero(true_didt)]
        reduced_didt_condition = self.IDT_condition[np.nonzero(reduced_didt)]
        if (len(np.nonzero(true_didt)) != np.nonzero(reduced_didt)) or (not np.all(np.equal(np.nonzero(true_didt), np.nonzero(reduced_didt)))):
            warnings.warn("Detail and Reduced are NOT the same!")
            warnings.warn(f"Detail:{true_didt_condition}; Reduced: {reduced_didt_condition}")
        else:
            self.GenAPARTDataLogger.info(f"DIDT condition is {true_didt_condition}")
        return true_didt_condition


    def CheckIDTCurve(self, save_dirpath = "./analysis/CheckIDTCurve", cut_time = None):
        """
        检查时间温度曲线以保证确实在当前工况下可以成功点火。如果不能成功点火，我们需要适当地调整 cut_time 或者 工况
        
        save_dirpath: 保存下来的图片所放的文件夹
        """
        mkdirplus(save_dirpath); cut_time = self.APART_args['idt_cut_time'] if cut_time is None else cut_time

        for condition in self.IDT_condition:
            phi, T, P = condition
            detail_gas = ct.Solution(self.detail_mech); reduced_gas = ct.Solution(self.reduced_mech)
            detail_gas.TP = T, P * ct.one_atm; reduced_gas.TP = T, P * ct.one_atm
            detail_gas.set_equivalence_ratio(phi, self.IDT_fuel, self.IDT_oxidizer); reduced_gas.set_equivalence_ratio(phi, self.IDT_fuel, self.IDT_oxidizer)
            detail_time, detail_curve, _, _ = solve_idt(
                detail_gas,
                get_curve = True,
                cut_time = 2 * cut_time, 
                mode = 0
            )
            reduced_time, reduced_curve, _, _ = solve_idt(
                reduced_gas,
                get_curve = True,
                cut_time = 2 * cut_time, 
                mode = 0
            )
            fig, ax = plt.subplots(1, 1)
            ax.plot(detail_time, detail_curve, c = 'blue', alpha = 0.8) 
            ax.plot(reduced_time, reduced_curve, c = 'orange', alpha = 0.8) 
            ax.axhline(y = T + 400, c = 'r', ls = '--')
            ax.axvline(x = min(cut_time, np.amax(reduced_time)), c = 'purple', ls = '--')
            ax.set_title(f'T={T}K P={P} atm phi={phi}')
            ax.set_xlabel("time (s)")
            ax.set_ylabel("Temperature (K)")
            fig.savefig(save_dirpath + f'/CheckIDTCurve_T={T}_P={P}_phi={phi}.png')


    def IDT_SensitivityDesent(self, start_samples = None, target_IDT_conditions = None, iter_nums = 10, delta = 0.01,  
                            eve_sample_size = 50, max_step_size = 0.2, cpu_nums = None, father_sample_size = None,
                              **kwargs):
        """
        适合于 self.reduced_mech IDT 点火失败的情况
        从 start_samples 中计算 IDT 的敏感度，然后进行多次循环下降过程，多次循环中尝试寻找能够使得所有对应工况的 IDT 均点火的样本点
        获得合适的样本点后使用多进程计算它们的 IDT 值并选择其中最小的样本点

        Args:
            start_samples: 二维数组，进行敏感度下降的初始样本点
            target_IDT_conditions: 二维数组，无法成功点火的 IDT 工况
            iter_nums: 敏感度下降的迭代次数
            delta: 计算敏感度使用的步长
            eve_sample_size: 每次迭代中，每个父样本点生成的子样本点的数量
            max_step_size: 每次迭代中，敏感度下降中每个父样本点生成的子样本点的最大步长
            cpu_nums: 多进程计算的 cpu 数量
        Returns:
            Alist: 二维数组，最终的样本点
            all_idt_data, all_T_data, all_psr_data, all_psr_extinction_data: 与 Alist 对应的 idt, IDT, IDT 数据
        """
        np.set_printoptions(precision = 6, suppress = True)
        dirpath = mkdirplus("./data/SensitivityDesent")
        target_IDT_conditions = self.IDT_condition if target_IDT_conditions is None else target_IDT_conditions
        cpu_nums = cpu_nums if cpu_nums is not None else os.cpu_count() - 1
        father_samples = start_samples if start_samples is not None else [self.A0]
        logger = Log("./log/SensitivityDesent.log")
        father_sample_size = len(father_samples) if father_sample_size is None else father_sample_size

        benchmark_IDT = yaml2idt(
            self.detail_mech, IDT_condition = target_IDT_conditions, 
            fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
            IDT_mode = self.IDT_mode, cut_time = self.APART_args['idt_cut_time']
        )
        DDesent_samples = []
        for iter in range(iter_nums):
            Desent_samples = []
            for i, sample in enumerate(father_samples):
                tmp_path = dirpath + f"/{i}th_father_sample.yaml"
                Adict2yaml(self.reduced_mech, tmp_path, Alist = sample, eq_dict = self.eq_dict)
                # 计算 IDT 和 IDT 的敏感性
                IDT_sensitivity, base_IDT = yaml2idt_sensitivity(
                    tmp_path, delta = delta, 
                    IDT_condition = target_IDT_conditions, 
                    fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                    IDT_mode = self.IDT_mode, cut_time = self.APART_args['idt_cut_time'],
                    need_base_IDT = True
                )
                # 针对 IDT 和 IDT 的 loss 进行敏感性下降。我们考虑的损失函数是： 
                # \sum_{pp} (IDT_{pp} - IDTtrue_{pp})^2 + \sum_{pp} (IDT_{pp} - IDTtrue_{pp})^2; 因此求导后获得
                # \sum_{pp} 2 * (IDT_{pp} - IDTtrue_{pp}) * IDT_sensitivity_{pp} + \sum_{pp} 2 * (IDT_{pp} - IDTtrue_{pp}) * IDT_sensitivity_{pp}
                ## 根据 self.eq_dcit 重排序 IDT_sensitivity 转化为 numpy 数组
                IDT_sensitivity_list = eq_dict_broadcast2Alist(IDT_sensitivity, self.eq_dict)
                # 计算损失函数的梯度
                grad = 2 * np.sum((base_IDT - benchmark_IDT) * IDT_sensitivity_list, axis = 1)
                # 选择合适的步长进行下降，保证下降范围在 0.1 内
                step = max_step_size / np.max(np.abs(grad))
                # 进行下降
                desent_sample = sample - step * grad
                # 在 sample 和 desent_sample 两点连成的直线上均匀取 50 个点
                desent_samples = []
                for j in range(eve_sample_size):
                    desent_samples.append((sample + j / 10 * (desent_sample - sample)).tolist())
                Desent_samples.append(desent_samples)
            DDesent_samples.append(Desent_samples)
            father_samples = np.array(Desent_samples).reshape(-1, len(sample))
            np.random.shuffle(father_samples)
            father_samples = father_samples[:father_sample_size]
            logger.info(f"iter {iter} finished, sampled {len(DDesent_samples) * len(DDesent_samples[0])} points.")
            logger.info(f"The example of base_IDT is {base_IDT}, the benchmark_IDT is {benchmark_IDT}, desent step is {step * grad}.")
        # 将 DDsent_samples 转化为二维数组, 其中一个维度是 sample 的维度
        DDesent_samples = np.array(DDesent_samples).reshape(-1, len(sample))
        # 使用多进程计算所有的 IDT 值
        with ProcessPoolExecutor(max_workers = cpu_nums) as exec:
            for index, sample in enumerate(DDesent_samples):
                exec.submit(
                        GenOneDataIDT, 
                        index = index,
                        IDT_condition = self.IDT_condition,
                        Alist = sample,
                        eq_dict = self.eq_dict,
                        fuel = self.IDT_fuel,
                        oxidizer = self.IDT_oxidizer,
                        reduced_mech = self.reduced_mech,
                        IDT_mode = self.IDT_mode,
                        # 将 idt_arrays 采取如下设置: 其每个分量是 true_idt_data 和 reduced_idt_data 对应分量中大的那个
                        idt_arrays = np.maximum(self.true_idt_data, self.reduced_idt_data),
                        cut_time_alpha = 10,
                        save_path = "./data/SensitivityDesent",
                        my_logger = logger,
                        **kwargs
                        )
        # 使用 gather_data 函数将数据收集起来
        self.gather_apart_data(
            save_path = "./data/SensitivityDesent",
            save_file_name = "./data/SensitivityDesentData.npz",
            rm_tree = True,
            logger = logger
        )
        # 加载 "./data/SensitivityDesentData.npz" 
        data = np.load("./data/SensitivityDesentData.npz"); Alist = data['Alist']; all_idt_data = data['all_idt_data']
        # 将 data['all_psr_extinction_data'] - self.true_psr_extinction_data 的绝对值计算无穷范数，得到的是一个一维数组
        diff_idt = np.linalg.norm(np.log10(all_idt_data) - np.log10(self.true_idt_data), ord = 2, axis = 1)
        # 将 diff_psrex 升序排序后选取前 10 个 Alist
        index = np.argsort(diff_idt)[:10]
        Alist = Alist[index]; all_idt_data = all_idt_data[index]
        # 将这 10 个 Alist 保存到 "./data/SensitivityDesentData.npz" 中
        np.savez("./data/SensitivityDesentData_filtered.npz", Alist = Alist, all_idt_data = all_idt_data)
        return Alist, all_idt_data


"""================================================================================================"""
"""                                        GenOneDataFunc                                          """
"""================================================================================================"""


@func_set_timeout(300)
def GenOneDataIDT(IDT_condition: np.ndarray, Alist:list, eq_dict:dict, 
                   reduced_mech:str, index:int,  my_logger:Log, tmp_chem_path:str = None, fuel:str = None, oxidizer: str = None, 
                   IDT_mode = 0, remove_chem = True, idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                   cut_time_alpha = 10, save_path = 'data/APART_data/tmp', IDT_fuel = None, IDT_oxidizer = None, **kwargs):
    """
    只生成 IDT 的数据; 完全继承 _GenOneDataIDT 的参数设置
    """
    tmp_chem_path = save_path + f'/{index}th.yaml' if tmp_chem_path is None else tmp_chem_path
    save_path = save_path + f'/{index}th.npz'
    t0 = time.time()
    Adict2yaml(reduced_mech, tmp_chem_path, eq_dict = eq_dict, Alist = Alist)
    IDT_condition = np.array(IDT_condition)
    if IDT_fuel is None or IDT_oxidizer is None:
        IDT_fuel = fuel; IDT_oxidizer = oxidizer
    idt_T = _GenOneIDT(IDT_condition, IDT_fuel, IDT_oxidizer, tmp_chem_path,  index,  my_logger, 
                       IDT_mode, idt_arrays, cut_time, cut_time_alpha, **kwargs)
    if remove_chem: os.remove(tmp_chem_path)
    if isinstance(idt_T, int):
        my_logger.warning(f'mechanism {index}:Error, Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024:.2e} GB")')
        return idt_T
    else:
        IDT, T = idt_T
        my_logger.info(f'mechanism {index} : cost {time.time()-t0:.2f} s, the first IDT element is {np.log10(IDT[0]):.3e},  Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024:.2e} GB")')
        np.savez(save_path, IDT = IDT, T = T,  Alist = Alist)
        return None