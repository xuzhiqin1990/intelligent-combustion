# -*- coding:utf-8 -*-
import os, time, traceback, subprocess
import numpy as np, pandas as pd, seaborn as sns, cantera as ct

from func_timeout import func_set_timeout
from multiprocessing import Pool
import psutil

from Apart_Package.utils.cantera_utils import *
from Apart_Package.utils.setting_utils import *
from Apart_Package.utils.yamlfiles_utils import *
from Apart_Package.utils.cantera_multiprocess_utils import yaml2idt_Mcondition, yaml2FS_Mcondition
from Apart_Package.utils.cantera_PSR_definations import *
from Apart_Package.utils.Expdata_utils import *

class APART_base():
    """
    弃用 DeePMR_base_class, 创建新替代品 APART_base. 目的是实现详细机理和简化机理的初步计算过程，用于适配
    APART.py

    需要实现的功能有:

    首先在底层增加 detail_mech 和 reduced_mech 两个概念

    其次划分 IDT 和 PSR 两个模块，可以做到两个模块分离，便于之后增加指标

    utils:
        1. mechanism_info 能够返回基本的机理信息; 反应数、组分数、燃料、氧化剂等等
        2. set_FO 设置燃料和氧化剂
        3. set_IDT_condition 能够获得/设置除了网格格点外的工况情况 == set_TPphi
        4. set_PSR_condition == set_PSR_condition
        5. GenTrueData 生成相应 IDT_condition 和 PSR_condition 条件下的数据
        6. LoadTrueData 加载指定路径下的真实值；如不存在则生成

    functions:
        1. cal_res_time 计算真实的 PSR 允许工况并从中挑选合适的 PSR 工况
        2. ValidationIDT 实现在计算之前的真实机理简化机理比较，为自适应工况采样做准备

    """
    def __init__(self, detail_mech:str = None, reduced_mech:str = None, setup_file = None,
                 setup = {}, cond_file = None, **kwargs):
        """
        设置 detail_mech 和 reduced_mech
        接受其他的输入：
            kwargs:
                setup_file: 输入一个setup.yaml文件，需要存在一个 key: APART_base 其中
                包括以下的内容
                    detail_mech， (文件)
                    reduced_mech， (文件)
                    fuel, oxidizer,
                    IDT_mode: 注明使用哪个 IDT 的定义，False 为不使用 IDT 模块
                    PSR_mode: False 为不使用 PSR 模块
                    Mole_mode: False 为不使用 Mole 模块
                    LFS_mode: False 为不使用 LFS 模块
                    IDT_condition, (文件)
                    PSR_condition, (文件)
                    IDT_T, IDT_P, IDT_phi,
                    PSR_T, PSR_P, PSR_phi
                    PSR 专用: 
                        cond_file: 储存 PSR condition 的位置; 如果启用了 PSR 模块则不能为 None
                        PSR_reduced_base：按照 reduced_mech 来计算 PSR res time
                        PSR_INI_RES_TIME: PSR res time 计算的初始时间
                        PSR_RES_TIME_LENGTH: PSR res time 计算的长度; 有两种输入方式:
                            1. 一个整数，表示 RES_TIME_LIST 的长度; 在实际中会将 true_ref_psr_data.npz 中的 RES_TIME_LIST 取出 res_time_length 分位点
                            2. str: "ThreeStage" 表示 RES_TIME_LIST 的长度为 
                            3. "FullLowerHalf" 表示在 ThreeStage 基础上增加 RES_TIME_LIST 的后半部分
                        PSR_EXP_FACTOR: PSR res time 计算的指数因子
                若不存在相应的 key 则设置为 None
                setup: dict, 包括所有 setup 的内容，请使用 **setup 输入
        add:
            self.setup; detailed_mech; reduced_mech; fuel; oxidizer; 等等
        """
        ct.suppress_thermo_warnings()
        # detail_mech 和 reduced_mech 两个参数的优先级更高
        self.detail_mech = detail_mech; self.reduced_mech = reduced_mech
        if not setup_file is None:
            setup = get_yaml_data(setup_file, "APART_base")
        if not setup is {}:
            setup.update(**kwargs); self.setup = setup
            if self.detail_mech is None: self.detail_mech = setup.get("detail_mech", None)
            if self.reduced_mech is None: self.reduced_mech = setup.get("reduced_mech", None)
            self.fuel = setup.get("fuel", None); self.oxidizer = setup.get("oxidizer", None)
            self.IDT_mode = setup.get("IDT_mode", True)
            # self.PSR_mode = setup.get("PSR_mode", False)
            # self.Mole_mode = setup.get("Mole_mode", False)
            # self.LFS_mode = setup.get("LFS_mode", False)
            # self.PSR_concentration_mode = setup.get("PSR_concentration_mode", False)
            if not self.IDT_mode is False:
                self.idt_defined_T_diff = setup.get("IDT_defined_T_diff", 400)
                self.idt_defined_time_multiple = setup.get("IDT_defined_time_multiple", 2)
                self.idt_defined_species = self.APART_args.get("idt_defined_species", 'OH')
                self.IDT_fuel = kwargs.get("IDT_fuel", self.fuel)
                self.IDT_oxidizer = kwargs.get("IDT_oxidizer", self.oxidizer)
            if 'IDT_condition' in setup.keys(): 
                self.set_IDT_condition(setup['IDT_condition'])
            if 'PSR_condition' in setup.keys(): 
                PSR_reduced_base = setup.get("PSR_reduced_base", False)
                ini_res_time = setup.get("PSR_INI_RES_TIME", 1)
                res_time_length = setup.get("PSR_RES_TIME_LENGTH", 'ThreeStage')
                exp_factor = 2 ** setup.get("PSR_EXP_FACTOR", 0.5)
                psr_error_tol = setup.get("PSR_ERROR_TOL", 100)
                self.PSR_fuel = kwargs.get("PSR_fuel", self.fuel)
                self.PSR_oxidizer = kwargs.get("PSR_oxidizer", self.oxidizer)
                self.set_PSR_condition(res_time_length = res_time_length, cond_file = cond_file, PSR_condition = setup['PSR_condition'], 
                                       reduced_base = PSR_reduced_base, ini_res_time = ini_res_time, exp_factor = exp_factor, psr_error_tol = psr_error_tol)
            if 'IDT_T' in setup.keys(): 
                self.set_IDT_condition(IDT_T = setup['IDT_T'], IDT_P = setup['IDT_P'], IDT_phi = setup['IDT_phi'])
            if 'PSR_T' in setup.keys(): 
                mkdirplus(os.path.dirname(cond_file))
                PSR_reduced_base = setup.get("PSR_reduced_base", False)
                ini_res_time = setup.get("PSR_INI_RES_TIME", 1)
                res_time_length = setup.get("PSR_RES_TIME_LENGTH", 'ThreeStage')
                exp_factor =  2 ** setup.get("PSR_EXP_FACTOR", 1/2)
                psr_error_tol = setup.get("PSR_ERROR_TOL", 100)
                self.PSR_fuel = kwargs.get("PSR_fuel", self.fuel)
                self.PSR_oxidizer = kwargs.get("PSR_oxidizer", self.oxidizer)
                self.set_PSR_condition(res_time_length = res_time_length, cond_file = cond_file, PSR_T = setup['PSR_T'], 
                                       PSR_P = setup['PSR_P'], PSR_phi = setup['PSR_phi'], reduced_base = PSR_reduced_base,
                                       ini_res_time = ini_res_time, exp_factor = exp_factor, psr_error_tol = psr_error_tol)
            if 'LFS_T' in setup.keys(): 
                self.LFS_fuel = kwargs.get("LFS_fuel", self.fuel)
                self.LFS_oxidizer = kwargs.get("LFS_oxidizer", self.oxidizer)
                self.set_LFS_condition(LFS_T = setup['LFS_T'], LFS_P = setup['LFS_P'], LFS_phi = setup['LFS_phi'])
            if 'HRR_T' in setup.keys(): 
                self.IDT_fuel = kwargs.get("HRR_fuel", self.fuel)
                self.IDT_oxidizer = kwargs.get("HRR_oxidizer", self.oxidizer)
                self.set_HRR_condition(HRR_T = setup['HRR_T'], HRR_P = setup['HRR_P'], HRR_phi = setup['HRR_phi'])
            # 20250624
            self.PSR_concentration_species = setup.get('PSR_concentration_species', None)
            if 'PSR_concentration_T' in setup.keys():
                self.PSR_concentration_T = setup['PSR_concentration_T']
                self.PSR_concentration_P = setup['PSR_concentration_P']
                self.PSR_concentration_phi = setup['PSR_concentration_phi']
                self.PSR_concentration_res_time = setup.get('PSR_concentration_res_time', 1)
                self.PSR_concentration_fuel = setup.get('PSR_concentration_fuel', self.PSR_fuel)
                self.PSR_concentration_oxidizer = setup.get('PSR_concentration_oxidizer', self.PSR_oxidizer)
                if 'PSR_concentration_diluent' in setup.keys():
                    self.PSR_concentration_diluent = setup['PSR_concentration_diluent']
                    assert isinstance(self.PSR_concentration_diluent, dict), \
                        f"PSR_concentration_diluent should be a dict, e.g. 'N2': 0.21, 'Ar': 0.79, but got {self.PSR_concentration_diluent}"
                    self.PSR_concentration_diluent = {
                        'diluent': list(self.PSR_concentration_diluent.keys())[0],
                        'fraction': {
                            'diluent': list(self.PSR_concentration_diluent.values())[0]
                        }
                    }
                else:
                    self.PSR_concentration_diluent = {}
                self.PSR_concentration_condition = np.array(
                    [[phi, T, P] for phi in self.PSR_concentration_phi for T in self.PSR_concentration_T for P in self.PSR_concentration_P]
                )
                self.PSR_concentration_kwargs = dict(
                                fuel = self.PSR_concentration_fuel,
                                oxidizer = self.PSR_concentration_oxidizer,
                                diluent = self.PSR_concentration_diluent,
                                species = self.PSR_concentration_species,
                                res_time = self.PSR_concentration_res_time,
                                condition = self.PSR_concentration_condition.tolist(),
                )
        else:
            raise ValueError("setup_file or setup is required to initialize APART_base class")
        
        if 'max_uncertainty_range' in setup.keys():
            self.max_uncertainty_range = setup['max_uncertainty_range']
            if isinstance(self.max_uncertainty_range, str) and os.path.exists(self.max_uncertainty_range) and self.max_uncertainty_range.endswith('.npy'):
                self.max_uncertainty_range = np.load(self.max_uncertainty_range, allow_pickle=True).tolist()
        else:
            self.max_uncertainty_range = None
    def __setitem__(self, key, value):
        self.__dict__[key] = value # 允许类进行字典式索引
    

    def __getitem__(self, key):
        return self.__dict__[key] # 允许类进行字典式索引


    def set_IDT_condition(self, cond_file = None, IDT_T = None, IDT_P = None, IDT_phi = None):
        """
        设置 IDT 的初始温度压强当量比采样点
        params:
            cond_file: 输入的 condition file, 需要满足如下格式
                为一 npy 文件中含有一个矩阵，每一行对应一个工况，[[phi, T, P], ...]
            IDT_T, IDT_P, IDT_phi: 如果需要网格采样工况设置这个，缺一不可
        add:
            self.IDT_condition，可以直接修改
        """
        match cond_file:
            case None:
                assert not (IDT_T is None or IDT_P is None or IDT_phi is None)
                self.IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )
            case _:
                self.IDT_condition = np.load(cond_file)

    
    def set_PSR_condition(self, res_time_length = None, cond_file = None, PSR_condition = None,
                          PSR_T = None, PSR_P = None, PSR_phi = None, reduced_base = False, 
                          ini_res_time = 1, exp_factor = 2 ** (1/2), psr_error_tol = 100, **kwargs):
        """
        设置 PSR 的初始温度压强当量比采样点

        如果存在 cond_file 会直接读取其中的所有内容，否则会生成 true_ref_psr_data.npz
        Args:
            cond_file: 应输入 true_ref_psr_data.npz; 其中包括两个 key: condition & RES_TIME_LIST
            PSR_T, PSR_P, PSR_phi: 如果需要网格采样工况设置这个，缺一不可; 若输入这个将自动调用 cal_res_time 计算 RES_TIME_LIST
            reduced_base: 用于计算 RES_TIME_LIST 的基础机理
            ini_res_time: RES_TIME_LIST 中的初始时间
        add:
            self.PSR_condition: 一个二维数组, [[phi, T, P], ...]
            self.RES_TIME_LIST: 一个二维数组, [res_time_list1, ...] 其包含了 solve_true_psr 中 RES_TIME_LIST 中的起始/中值/末尾三个数值
        """
        # PSR 专用的 condition 以免和 IDT 混淆
        if not cond_file is None and os.path.exists(cond_file):
            data = np.load(cond_file, allow_pickle = True)
            # 检测文件的 condition 和 定义的是否相同，不同会按照文件更改 condition
            self.PSR_condition = data['condition']
            self.RES_TIME_LIST = data['RES_TIME_LIST']       
        else:       
            if PSR_condition is None:
                assert not (PSR_T is None or PSR_P is None or PSR_phi is None)     
                self.PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
            else:
                self.PSR_condition = PSR_condition
            self.cal_res_time(self.PSR_condition, res_time_length, cond_file, reduced_base = reduced_base, 
                              ini_res_time = ini_res_time, exp_factor = exp_factor, psr_error_tol = psr_error_tol, **kwargs)

                    
    def set_mole_condition(self, species, cond_file = None, mole_T = None, mole_P = None, mole_phi = None):
        """
        设置 mole 的初始温度压强当量比采样点
        params:
            species
            cond_file: 输入的 condition file, 需要满足如下格式
                为一 npy 文件中含有一个矩阵，每一行对应一个工况，[[phi, T, P], ...]
            mole_T, mole_P, mole_phi: 如果需要网格采样工况设置这个，缺一不可
        add:
            self.mole_condition，可以直接修改
        """
        self.mole_species = species
        match cond_file:
            case None:
                assert not (mole_T is None or mole_P is None or mole_phi is None)
                self.mole_condition = np.array(
                    [[phi, T, P] for phi in mole_phi for T in mole_T for P in mole_P]
                )
            case _:
                self.mole_condition = np.load(cond_file)


    def set_LFS_condition(self, cond_file = None, LFS_T = None, LFS_P = None, LFS_phi = None):
        """
        生成 LFS 的真实数据和简化数据; 需要注意的是 LFS 的计算速度极慢，因此最优选择是提前计算好
        params:
            cond_file: 输入的 condition file, 需要满足如下格式
            为一 npy 文件中含有一个矩阵，每一行对应一个工况，[[phi, T, P], ...]
            LFS_T, LFS_P, LFS_phi: 如果需要网格采样工况设置这个，缺一不可
        add:
            self.LFS_condition
        """
        match cond_file:
            case None:
                assert not (LFS_T is None or LFS_P is None or LFS_phi is None)
                self.LFS_condition = np.array(
                    [[phi, T, P] for phi in LFS_phi for T in LFS_T for P in LFS_P]
                )
            case _:
                self.LFS_condition = np.load(cond_file)


    def set_HRR_condition(self, cond_file = None, HRR_T = None, HRR_P = None, HRR_phi = None):
        """
        生成 HRR 的真实数据和简化数据
        params:
            cond_file: 输入的 condition file, 需要满足如下格式
            为一 npy 文件中含有一个矩阵，每一行对应一个工况，[[phi, T, P], ...]
            HRR_T, HRR_P, HRR_phi: 如果需要网格采样工况设置这个，缺一不可
        add:
            self.HRR_condition
        """
        match cond_file:
            case None:
                assert not (HRR_T is None or HRR_P is None or HRR_phi is None)
                self.HRR_condition = np.array(
                    [[phi, T, P] for phi in HRR_phi for T in HRR_T for P in HRR_P]
                )
            case _:
                self.HRR_condition = np.load(cond_file)


    def cal_res_time(self, PSR_condition, res_time_length = 'ThreeStage',
                     cond_file:str = None, save_dirpath = "./data/true_data", 
                     reduced_base = False, ini_res_time = 1, exp_factor = 2 ** (1/2),  **kwargs):
        """
        生成 PSR 列表，注意，一个工况就对应着 PSR 的一个列表. 此处 RES_TIME_LIST 采用的是 append 方式合并，如此可以得到二维的数据
        Args:
            PSR_condition: 输入一个二维 condition 矩阵
            res_time_length: 生成的 RES_TIME_LIST 的长度; 有两种输入方式: 
                1. 一个整数，表示 RES_TIME_LIST 的长度; 在实际中会将 true_ref_psr_data.npz 中的 RES_TIME_LIST 取出 res_time_length 分位点
                2. str: "ThreeStage" 表示 RES_TIME_LIST 的长度为 3; "FullLowerHalf" 表示在 ThreeStage 基础上增加 RES_TIME_LIST 的后半部分
            cond_file: 保存 PSR 的 condition file 的位置
            reduced_base: 因为可能存在 detail_mech PSR 很完美点火但是 reduced_mech 在某些条件下没法点火的情况，因此增加了以 reduced_mech 为基准
                            的 PSR restime 求解
            ini_res_time: solve_psr_true 中 初始的 res_time
        add:
            self.RES_TIME_LIST
            self.PSR_TLIST
        """
        def division_RES_TIME_LIST(res_time, res_time_length, **kwargs):
            # 根据 res_time_length 的不同，生成不同的 RES_TIME_LIST
            if isinstance(res_time_length, int):
                # res_time_length = min(res_time_length, *[len(RES_TIME_LIST[i]) for i in range(len(RES_TIME_LIST))])
                assert res_time_length <= len(res_time), f"res_time_length = {res_time_length} is larger than len(res_time) = {len(res_time)}"
                new_res_time = [
                        np.array_split(res_time, res_time_length - 1)[i][0] for i in range(res_time_length - 1)
                    ]
                new_res_time.append(res_time[-1])
            elif res_time_length == 'ThreeStage':
                length_res_time = len(res_time); 
                new_res_time = [res_time[0], res_time[int(length_res_time / 2)], res_time[-1]]
            elif res_time_length == 'FullLowerHalf':
                length_res_time = kwargs.get('tile_length', len(res_time)); 
                new_res_time = [res_time[0]]; new_res_time.extend(res_time[-length_res_time:])
            return new_res_time
        
        if cond_file == None: cond_file = save_dirpath + '/ref_psr_data.npz'; mkdirplus(save_dirpath)
        true_ref_file = save_dirpath + '/true_ref_psr_data_all.npz'
        if not os.path.exists(true_ref_file) or os.path.getsize(true_ref_file) <= 100:
            gas = ct.Solution(self.reduced_mech) if reduced_base else ct.Solution(self.detail_mech)
            print(f"here at cal_res_time, {PSR_condition=}")
            RES_TIME_LIST, PSR_T_LIST = [], []
            for condition_item in PSR_condition:
                phi, T, p  = condition_item
                gas.TP = T, p * ct.one_atm
                gas.set_equivalence_ratio(phi, self.PSR_fuel, self.PSR_oxidizer)
                tmp_RES_TIME_LIST, tmp_PSR_T_LIST = solve_psr_true(gas, ini_res_time, exp_factor = exp_factor)
                RES_TIME_LIST.append(tmp_RES_TIME_LIST); PSR_T_LIST.append(tmp_PSR_T_LIST)
            if not save_dirpath is None:
                np.savez(true_ref_file, condition = PSR_condition, 
                         RES_TIME_LIST = np.array(RES_TIME_LIST, dtype=object),
                         PSR_T_LIST = np.array(PSR_T_LIST, dtype=object))
        else:
            true_ref_file_data = np.load(true_ref_file, allow_pickle = True)
            RES_TIME_LIST = true_ref_file_data['RES_TIME_LIST']
            PSR_T_LIST = true_ref_file_data['PSR_T_LIST']

        gas = ct.Solution(self.reduced_mech) if reduced_base else ct.Solution(self.detail_mech)
        NEW_RES_TIME = []; NEW_T_LIST = []; tile_length = 0
        print("here at cal_res_time, start to solve_psr")
        # 保证最后的结果是矩阵而非不同长度列表集合
        if isinstance(res_time_length, int):
            res_time_length = min(res_time_length, *[len(RES_TIME_LIST[i]) for i in range(len(RES_TIME_LIST))])
        elif res_time_length == 'FullLowerHalf':
            tile_length = min([len(RES_TIME_LIST[i]) for i in range(len(RES_TIME_LIST))])
            tile_length = int(tile_length / 2)
        print("here at tile_length:", tile_length)
        
        # 用存储的res_time模拟，如果烧不起来，则把最小的res_time去掉
        for i in range(len(PSR_condition)):
            phi, T, p = PSR_condition[i][0], PSR_condition[i][1], PSR_condition[i][2]
            res_time_list = RES_TIME_LIST[i]
            new_res_time = division_RES_TIME_LIST(res_time_list, res_time_length, tile_length = tile_length)
            print(f"here at res_time_length = {res_time_length},")
            psr_T = solve_psr(gas, new_res_time, T, p, phi, self.PSR_fuel, self.PSR_oxidizer, error_tol=0.)
            psr_T = np.array(psr_T)
            
            fix_count = 0
            while np.any(psr_T - T <= 10):
                fix_count += 1
                print(f"here at res_time_length = {res_time_length}, PSR condition: {PSR_condition[i]} is not suitable, Resolving...")
                # 剔除满足 np.any(np.array(psr_T) - T <= 10) 的点
                res_time_list = res_time_list[:len(res_time_list) - fix_count]
                # 将 res_time_list 补全为最后一个值
                res_time_list = np.concatenate([res_time_list, np.ones(fix_count) * np.amin(res_time_list)], axis = None)
                new_res_time = division_RES_TIME_LIST(res_time_list, res_time_length, tile_length = tile_length)
                psr_T = solve_psr(gas, new_res_time, T, p, phi, self.PSR_fuel, self.PSR_oxidizer, error_tol=0.)

            NEW_RES_TIME.append(new_res_time)
            NEW_T_LIST.extend(psr_T)
        self.RES_TIME_LIST = np.array(NEW_RES_TIME); self.PSR_TLIST = np.array(NEW_T_LIST)

        np.savez(cond_file, condition = PSR_condition, 
                    RES_TIME_LIST = NEW_RES_TIME, T_LIST = NEW_T_LIST)

    
    def GenTrueData(self, mode:str = 'idt', idt_cut_time = 1, csv_path = None, save_path = None, **kwargs):
        """
        生成所有的真实和简化机理的 IDT 或者 PSR 数据
        params:
            mode: {'idt', 'psr', 'fs',} 接受这4种的输入
            idt_cut_time: 计算真实机理和简化机理的 IDT cut_time
            kwargs:
                save_path: 保存的地址
                kwargs 中包含所有 solve idt fast 的参数；详见 solve idt fast
                kwargs 中包含所有 solve flame 的参数，详见 solve flame
        add:
            self.true_idt_data; reduced_idt_data; *_T_data; *_psr_data
        """
        self.true_idt_data: np.ndarray
        self.reduced_idt_data: np.ndarray
        self.true_T_data: np.ndarray
        self.reduced_T_data: np.ndarray
        self.true_psr_data: np.ndarray
        self.reduced_psr_data: np.ndarray     
        self.true_mole_data: np.ndarray
        self.reduced_mole_data: np.ndarray      
        self.true_hrr_data: np.ndarray
        self.reduced_hrr_data: np.ndarray
        self.true_lfs_data: np.ndarray
        self.reduced_lfs_data: np.ndarray
        self.true_psr_concentration_data: np.ndarray
        self.reduced_psr_concentration_data: np.ndarray       
        t0 = time.time()
        match mode:
            case 'idt':
                if csv_path is None:
                    self.true_idt_data, self.true_T_data = yaml2idt(
                        self.detail_mech, mode = self.IDT_mode, cut_time = idt_cut_time, IDT_condition = self.IDT_condition,
                        fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer, idt_defined_T_diff = self.idt_defined_T_diff, 
                        time_multiple = self.idt_defined_time_multiple, **kwargs
                    )
                else:
                    self.getIDTconditionAndTrueDataFromCSV(csv_path)   
                self.reduced_idt_data, self.reduced_T_data = yaml2idt(
                    self.reduced_mech, mode = self.IDT_mode, cut_time = idt_cut_time, IDT_condition = self.IDT_condition,
                    fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer, idt_defined_T_diff = self.idt_defined_T_diff, 
                    time_multiple = self.idt_defined_time_multiple, **kwargs
                )
                if csv_path is not None: self.true_T_data = self.reduced_T_data
                    # _, self.true_T_data = yaml2idt(
                    #     self.detail_mech, mode = self.IDT_mode, cut_time = idt_cut_time, IDT_condition = self.IDT_condition,
                    #     fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer, idt_defined_T_diff = self.idt_defined_T_diff, 
                    #     time_multiple = self.idt_defined_time_multiple, **kwargs
                    # )
                    

                print(f"IDT GenTrueData 消耗{time.time() - t0}; shape of true_idt_data: {self.true_idt_data.shape}; shape of reduced_idt_data: {self.reduced_idt_data.shape}; shape of IDT_condition: {self.IDT_condition.shape}")
                if not save_path is None: np.savez(save_path, true_idt_data = self.true_idt_data,
                                    reduced_idt_data = self.reduced_idt_data, true_T_data = self.true_T_data,
                                    reduced_T_data = self.reduced_T_data, condition = self.IDT_condition)
            case 'psr':
                for ind, mech in zip(['true', 'reduced'], [self.detail_mech, self.reduced_mech]):
                    PSR_T = yaml2psr(
                        mech, PSR_condition = self.PSR_condition, RES_TIME_LIST = self.RES_TIME_LIST, error_tol = 0., 
                        fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer, **kwargs
                    )                   
                    print(f"{self.PSR_condition=}")
                    print(f"PSR GenTrueData 消耗{time.time() - t0}")
                    self[f"{ind}_psr_data"] = PSR_T
                if not save_path is None: np.savez(save_path, true_psr_data = self.true_psr_data,
                                    reduced_psr_data = self.reduced_psr_data, RES_TIME_LIST = self.RES_TIME_LIST, condition = self.PSR_condition)
            case 'hrr':
                for ind, mech in zip(['true', 'reduced'], [self.detail_mech, self.reduced_mech]):
                    _, hrr, _ = yaml2idt_hrr(
                        mech, self.HRR_condition, cut_time = idt_cut_time * 10, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                        time_multiple = 10
                    )
                    print(f"hrr GenTrueData 消耗{time.time() - t0}")
                    self[f"{ind}_hrr_data"] = np.array(hrr)
                if not save_path is None: np.savez(save_path, true_hrr_data = self.true_hrr_data, reduced_hrr_data = self.reduced_hrr_data, condition = self.HRR_condition)
            case 'lfs':
                if not csv_path is None:
                    self.getLFSconditionAndTrueDataFromCSV(csv_path)
                else:
                    print("Warning: LFS data is very slow to generate, please prepare it in advance; Start to generate LFS data...")
                    LFS = yaml2FS_Mcondition(
                        self.detail_mech, self.LFS_condition, fuel = self.LFS_fuel, oxidizer = self.LFS_oxidizer, 
                        cpu_process = os.cpu_count() - 1, **kwargs
                    )
                    self.true_lfs_data = np.array(LFS)
                    true_lfs_data_path = os.path.dirname(save_path) + '/true_lfs_data.npz'
                    np.savez(true_lfs_data_path, true_lfs_data = self.true_lfs_data, condition = self.LFS_condition)
                LFS = yaml2FS_Mcondition(
                        self.reduced_mech, self.LFS_condition, fuel = self.LFS_fuel, oxidizer = self.LFS_oxidizer,
                        cpu_process = os.cpu_count() - 1, **kwargs
                )
                self.reduced_lfs_data = np.array(LFS)
                np.savez(save_path, true_lfs_data = self.true_lfs_data, reduced_lfs_data = self.reduced_lfs_data, condition = self.LFS_condition)
                print(f"LFS GenTrueData 消耗{time.time() - t0};")
            case 'PSR_concentration':
                # for ind, mech in zip(['true', 'reduced'], [self.detail_mech, self.reduced_mech]):
                #     print(f"mech: {mech}, PSR_condition: {self.PSR_concentration_condition}, PSR_res_time: {self.PSR_concentration_res_time}, \
                #         fuel: {self.PSR_concentration_fuel}, oxidizer: {self.PSR_concentration_oxidizer}, \
                #         diluent: {self.PSR_concentration_diluent}, species: {self.PSR_concentration_species}")
                #     # exit()
                if csv_path is not None:
                    self.getPSR_concentration_conditionAndTrueDataFromCSV(csv_path)
                else:
                    mech = self.detail_mech
                    PSR_concentration_data = yaml2PSR_concentration(
                        mech, PSR_condition = self.PSR_concentration_condition, concentration_res_time = self.PSR_concentration_res_time,
                        fuel = self.PSR_concentration_fuel, oxidizer = self.PSR_concentration_oxidizer,
                        diluent = self.PSR_concentration_diluent, concentration_species = self.PSR_concentration_species, **kwargs
                    )
                    
                    print(f"PSR concentration GenTrueData detail 消耗{time.time() - t0}")
                    self.true_psr_concentration_data = PSR_concentration_data

                self.reduced_psr_concentration_data = yaml2PSR_concentration(
                    self.reduced_mech, PSR_condition = self.PSR_concentration_condition, concentration_res_time = self.PSR_concentration_res_time,
                    fuel = self.PSR_concentration_fuel, oxidizer = self.PSR_concentration_oxidizer,
                    diluent = self.PSR_concentration_diluent, concentration_species = self.PSR_concentration_species, **kwargs
                )
                
                print(f"PSR concentration GenTrueData reduced 消耗{time.time() - t0}")

                if not save_path is None: np.savez(save_path, true_psr_concentration_data = self.true_psr_concentration_data,
                                    reduced_psr_concentration_data = self.reduced_psr_concentration_data, 
                                    condition = self.PSR_concentration_condition, res_time = self.PSR_concentration_res_time)
     
                
    # def GenTrueLFSData(self, save_path = ".", csv_path = None, true_lfs_data_path = None, **kwargs):
    #     """
    #     生成真实机理和简化机理的 LFS 数据；但是 LFS 数据生成极慢，建议提前准备好
    #     add:
    #         self.true_lfs_data; reduced_lfs_data
    #     """
    #     self.true_lfs_data: np.ndarray
    #     self.reduced_lfs_data: np.ndarray
    #     t0 = time.time()
    #     if not true_lfs_data_path is None and os.path.exists(true_lfs_data_path):
    #         Data = np.load(true_lfs_data_path)
    #         self.true_lfs_data = Data['true_lfs_data']
    #         # self.LFS_condition = Data['condition']
    #     elif not csv_path is None:
    #         self.getLFSconditionAndTrueDataFromCSV(csv_path)
    #     else:
    #         print("Warning: LFS data is very slow to generate, please prepare it in advance; Start to generate LFS data...")
    #         LFS = yaml2FS_Mcondition(
    #             self.detail_mech, self.LFS_condition, fuel = self.LFS_fuel, oxidizer = self.LFS_oxidizer, 
    #             cpu_process = os.cpu_count() - 1, **kwargs
    #         )
    #         self.true_lfs_data = np.array(LFS)
    #         true_lfs_data_path = os.path.dirname(save_path) + '/true_lfs_data.npz'
    #         np.savez(true_lfs_data_path, true_lfs_data = self.true_lfs_data, condition = self.LFS_condition)
    #     LFS = yaml2FS_Mcondition(
    #             self.reduced_mech, self.LFS_condition, fuel = self.LFS_fuel, oxidizer = self.LFS_oxidizer,
    #             cpu_process = os.cpu_count() - 1, **kwargs
    #     )
    #     self.reduced_lfs_data = np.array(LFS)
    #     np.savez(save_path, true_lfs_data = self.true_lfs_data, reduced_lfs_data = self.reduced_lfs_data, condition = self.LFS_condition)
    #     print(f"LFS GenTrueData 消耗{time.time() - t0};")


    def LoadTrueData(self, save_path:str = None, mode = 'idt', csv_path = None, refresh = False, **kwargs):
        """
        LoadTrueData 加载真实/简化数据或生成数据并保存

        params:
            save_path: 真实数据的文件名
            mode: 同 GenTrueData 的 mode
            refresh: 是否在相应路径下重新生成真实数据
        add:
            (self.true_idt_data, self.true_T_data) or (self.true_psr_data) or (self.true_mole_data) or (self.true_lfs_data)
            self.Temperture_Diff
        """
        if mode == 'idt':
            if save_path is None or not os.path.exists(save_path) or refresh:
                self.GenTrueData(mode = mode, save_path = save_path, csv_path = csv_path, **kwargs)
            Data = np.load(save_path)
            self.true_idt_data, self.true_T_data = Data['true_idt_data'], Data['true_T_data']
            self.reduced_idt_data, self.reduced_T_data = Data['reduced_idt_data'], Data['reduced_T_data']
            self.Temperature_Diff = np.abs(self.true_T_data - self.reduced_T_data)
            if not csv_path is None:
                self.getIDTconditionAndTrueDataFromCSV(csv_path)
            else:
                # 补充 self.true_idt_uncertainty
                self.true_idt_uncertainty = np.ones_like(self.true_idt_data)
        elif mode == 'psr':
            if save_path is None or not os.path.exists(save_path) or refresh:
                self.GenTrueData(mode = mode, save_path = save_path, csv_path = csv_path, **kwargs)
            Data = np.load(save_path)
            self.true_psr_data, self.reduced_psr_data = Data['true_psr_data'], Data['reduced_psr_data']
        elif mode == 'mole':
            if save_path is None or not os.path.exists(save_path) or refresh:
                self.GenTrueData(mode = mode, save_path = save_path, csv_path = csv_path, **kwargs)
            Data = np.load(save_path)
            self.true_mole_data, self.reduced_mole_data = Data['true_mole_data'], Data['reduced_mole_data']
        elif mode == 'lfs':
            if save_path is None or not os.path.exists(save_path) or refresh:
                
                self.GenTrueData(mode = mode, save_path = save_path, csv_path = csv_path, **kwargs)
            Data = np.load(save_path)
            self.true_lfs_data, self.reduced_lfs_data = Data['true_lfs_data'], Data['reduced_lfs_data']
            if csv_path is not None:
                self.getLFSconditionAndTrueDataFromCSV(csv_path)
            else:
                # 补充 self.true_lfs_uncertainty
                self.true_lfs_uncertainty = np.ones_like(self.true_lfs_data)
        elif mode == 'hrr':
            if save_path is None or not os.path.exists(save_path) or refresh:
                self.GenTrueData(mode = mode, save_path = save_path, csv_path = csv_path, **kwargs)
            Data = np.load(save_path)
            self.true_hrr_data, self.reduced_hrr_data = Data['true_hrr_data'], Data['reduced_hrr_data']
        elif mode == 'PSR_concentration':
            if save_path is None or not os.path.exists(save_path) or refresh:
                self.GenTrueData(mode = mode, save_path = save_path, csv_path = csv_path, **kwargs)
            Data = np.load(save_path)
            self.true_psr_concentration_data, self.reduced_psr_concentration_data = Data['true_psr_concentration_data'], Data['reduced_psr_concentration_data']
            if csv_path is not None:
                self.getPSR_concentration_conditionAndTrueDataFromCSV(csv_path)
                
                
    # def LoadTrueLFSData(self, true_data_path:str = ".", refresh = False, **kwargs):
    #     """
    #     LoadTrueLFSData 加载真实/简化 LFS 数据或生成 LFS 数据并保存
    #     由于详细机理的 LFS 数据生成极慢，建议提前准备好，且将简化机理和真实机理两个数据放在不同的文件内

    #     params:
    #         true_data_path: 真实数据的文件名
    #         refresh: 是否在相应路径下重新生成真实数据
    #     add:
    #         (self.true_lfs_data, self.reduced_lfs_data)
    #     """
    #     reduced_data_path = true_data_path + '/reduced_lfs_data.npz'
    #     true_data_path = true_data_path + '/true_lfs_data.npz'
    #     if not os.path.exists(true_data_path):
    #         self.GenTrueLFSData(save_path = true_data_path, true_data_flag = True, reduced_data_flag = False, **kwargs)
    #     if not os.path.exists(reduced_data_path) or refresh:
    #         self.GenTrueLFSData(save_path = reduced_data_path, true_data_flag = False, reduced_data_flag = True, **kwargs)
    #     self.true_lfs_data = np.load(true_data_path)['true_lfs_data']
    #     self.reduced_lfs_data = np.load(reduced_data_path)['reduced_lfs_data']


    """================================================================================================"""
    """                                         Validation                                             """
    """================================================================================================"""
    
    def SFAdjust_IDT_condition(self, range_phi, range_T, range_P, optim_chem = None,  delta_range_phi = 0, delta_range_T = 0, 
                     delta_range_P = 0, conserve_wc = None, save_path = None, save_nums = None,
                     **kwargs):
        """
        根据在一个给定范围的工况下 detail_idt 和 reduced_idt 的差距进行自适应的采样
        采样模式为：
            1. 根据 abs(detail - reduce) 对所有待筛选工况进行排序
            2. 将排序的第一个工况确定下来，并在一个给定范围 delta 内删除列表中与该工况相近的其他工况
            3. 再选取再删除，直到达到预计好的数目
            提前给定的一些工况将会被保留下来
        params:
            optim_chem: 基于 optim_chem 进行 IDT 的调整; 如果不设置 optim_chem 会基于 reduced_chem 进行调整
            range_phi, T, P: 用于生成所有的网格状待筛选工况
            delta_range_phi, T, P: 筛选范围 delta，默认为 0
            conserve_wc: 预计要保留的工况，推荐保留筛选边界, 列需为三列
            save_path: 保存 IDT_condition 的路径
            save_nums: 最后经过 delta 筛选后保存前 save_nums 个工况
            kwargs:
                detail_mech: 详细机理路径；不指定将直接调用 self.detail_gas
                reduced_mech: 详细机理路径；不指定将直接调用 self.reduced_gas

        add:
            self.IDT_condition
        """
        np.set_printoptions(suppress = True)
        detail_mech = kwargs.get("detail_mech", self.detail_mech)
        reduced_mech = kwargs.get("reduced_mech", self.reduced_mech)
        optim_chem = reduced_mech if optim_chem is None else optim_chem
        
        IDT_conditions = [[phi, T, P] for phi in range_phi for T in range_T for P in range_P]
        if not conserve_wc is None:
            IDT_conditions = np.r_[conserve_wc, IDT_conditions]
        IDTs = []
        for condition in IDT_conditions:
            phi, T, P = condition; gas1 = ct.Solution(detail_mech); gas2 = ct.Solution(optim_chem)
            gas1.TP = T, P * ct.one_atm; gas2.TP = T, P * ct.one_atm
            gas1.set_equivalence_ratio(phi, self.IDT_fuel, self.IDT_oxidizer); gas2.set_equivalence_ratio(phi, self.IDT_fuel, self.IDT_oxidizer)
            idt1, temperature1 = solve_idt(gas1, mode = self.IDT_mode); idt2, temperature2 = solve_idt(gas2, mode = self.IDT_mode)
            IDTs.append(-abs(idt1 - idt2))
        IDT_conditions = np.array(IDT_conditions)[np.argsort(IDTs), :]
        
        k = 0
        while k < len(IDT_conditions):
            condition = IDT_conditions[k,:]
            j = k
            while j < len(IDT_conditions):
                if abs(IDT_conditions[j,0] - condition[0]) < delta_range_phi or abs(IDT_conditions[j,1] - condition[1]) < delta_range_T\
                    or abs(IDT_conditions[j,2] - condition[2]) < delta_range_P:
                    IDT_conditions = np.delete(IDT_conditions, j, axis = 0)
                j += 1
            if k >= len(IDT_conditions):
                break
            print(f"after considering condition {condition}, we have {IDT_conditions=}")
            k += 1
        if not conserve_wc is None:
            IDT_conditions = np.r_[conserve_wc, IDT_conditions]
            IDT_conditions = np.unique(IDT_conditions, axis = 0)
        if not save_nums is None:
            IDT_conditions = IDT_conditions[:save_nums, :]
        self.IDT_condition = IDT_conditions
        if not save_path is None:
            np.savez(save_path, IDT_conditions = IDT_conditions, range_phi = range_phi, range_T = range_T, range_P = range_P,
                            IDTs = -np.reshape(IDTs, (len(range_phi), len(range_T), len(range_P))))


    def GridAdjust_IDT_condition(self, grid_T:int, grid_P:int, grid_phi:int, optim_chem:str = None, detail_mech = None, 
                                 Tlim:tuple = None, Plim:tuple = None, philim:tuple = None, IDT_nums:int = None, conserve_wc = None,
                                 delta_range_phi = 0, delta_range_T = 0, delta_range_P = 0, save_path = "./data/APART_data/reduced_data", logger = None,
                                 cpu_nums = 0, cut_time = 10, **kwargs):
        """
        构造网格用以调整 self.IDT_condition, 具体实现过程是计算上一代的样本点在所有网格上的 IDT 与详细机理 IDT 的差距
        然后选择差距最大的网格点, 根据 delta_range_x 作为距离限制删除后, 作为新的 self.IDT_condition..
        params:
            optim_chem: 基于 optim_chem 进行 IDT 的调整; 如果不设置 optim_chem 会基于 reduced_chem 进行调整
            grid_T, P, phi: 网格数量
            Tlim, Plim, philim: 限制网格的范围 形如: (Tmin, Tmax)
            IDT_nums: 保留的 IDT 条件数目
            delta_range_phi, T, P: 筛选范围 delta，默认为 0
            conserve_wc: 预计要保留的工况，推荐保留筛选边界, 列需为三列
            save_path: 保存 详细机理在网格点上的 IDT 的路径; key: IDT_grid, true_idt_data, optim_idt_data
            cpu_nums: 并行计算的 cpu 数量; 默认为 0, 即不并行
            cut_time: 计算 IDT 的最大时间
        add:
            self.IDT_condition            
        """
        np.set_printoptions(suppress = True)
        if detail_mech is None: detail_mech = self.detail_mech
        if optim_chem is None: optim_chem = self.reduced_mech
        if IDT_nums is None: IDT_nums = self.IDT_condition.shape[0]
        if logger is None: logger = Log("./GridAdjust_IDT_condition.log")
        # 若 save_path 不为 None 且 save_path 对应的文件存在, 则直接读取
        if Tlim is None: Tlim = (self.IDT_condition[:,1].min(), self.IDT_condition[:,1].max())
        if Plim is None: Plim = (self.IDT_condition[:,2].min(), self.IDT_condition[:,2].max())
        if philim is None: philim = (self.IDT_condition[:,0].min(), self.IDT_condition[:,0].max())
        grid_T = np.linspace(Tlim[0], Tlim[1], grid_T); grid_P = np.linspace(Plim[0], Plim[1], grid_P); grid_phi = np.linspace(philim[0], philim[1], grid_phi); 
        IDT_grid = np.array([[phi, T, P] for T in grid_T for P in grid_P for phi in grid_phi])
        if cpu_nums == 0:
            true_idt_data, _ = yaml2idt(detail_mech,  IDT_condition = IDT_grid, mode = self.IDT_mode, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer, cut_time = cut_time, **kwargs)
            optim_idt_data, _ = yaml2idt(optim_chem, IDT_condition = IDT_grid, fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer, mode = self.IDT_mode, cut_time = cut_time, **kwargs)
        else:
            # 使用yaml2idt_Mcondition函数    
            true_idt_data = yaml2idt_Mcondition(detail_mech, self.IDT_mode, IDT_condition = IDT_grid, cut_time = cut_time,
                                                fuel = self.IDT_IDT_fuel, oxidizer = self.oxidizer, save_dirpath = save_path, **kwargs)
            optim_idt_data = yaml2idt_Mcondition(optim_chem, self.IDT_mode, IDT_condition = IDT_grid, cut_time = cut_time,
                                                fuel = self.IDT_fuel, oxidizer = self.oxidizer, save_dirpath = save_path, **kwargs)
        diff_idt = -np.abs(true_idt_data - optim_idt_data)

        # 根据 diff_idt 找到对应的最小 IDT_nums 个网格点
        IDT_grid = IDT_grid[np.argsort(diff_idt), :]
        
        # 根据 delta_range_x 筛选网格点；删除距离过近的网格点
        k = 0
        while k < len(IDT_grid):
            condition = IDT_grid[k,:]
            j = k
            while j < len(IDT_grid):
                if abs(IDT_grid[j,0] - condition[0]) < delta_range_phi or abs(IDT_grid[j,1] - condition[1]) < delta_range_T\
                    or abs(IDT_grid[j,2] - condition[2]) < delta_range_P:
                    IDT_grid = np.delete(IDT_grid, j, axis = 0)
                j += 1
                logger.info(f"The {j}th IDT_grid is deleted, The memory cost of the whole process is {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")
            if k >= len(IDT_grid):
                break
            logger.info(f"after considering condition {condition}, we have {IDT_grid=}")
            logger.info("="*50)
            k += 1
        
        # 截取前 IDT_nums 个网格点, 如果网格点数量不足则全部保存
        if IDT_nums < len(IDT_grid):
            IDT_grid = IDT_grid[:IDT_nums, :]

        # 保留 conserve_wc
        if not conserve_wc is None: 
            IDT_grid = np.r_[conserve_wc, IDT_grid]
            IDT_grid = np.unique(IDT_grid, axis = 0)

        self.IDT_condition = IDT_grid
        if not save_path is None:
            np.savez(save_path, IDT_grid = IDT_grid, true_idt_data = true_idt_data, optim_idt_data = optim_idt_data)
    
    
    def getIDTconditionAndTrueDataFromCSV(self, csv_path:str,):
        """
        从 csv 文件中获取 IDT_condition, IDT_fuel, IDT_oxidizer, true_idt_data
        若 csv 文件中存在 mode 则获取 mode
        add:
            self.IDT_condition, self.IDT_fuel, self.IDT_oxidizer, self.true_idt_data, self.IDT_mode, self.true_idt_uncertainty
        """
        self.IDT_condition: np.ndarray
        self.IDT_fuel: str | List[str]
        self.IDT_oxidizer: str | List[str]
        self.IDT_mode: str
        self.true_idt_data: np.ndarray
        self.true_idt_uncertainty: np.ndarray = None
        print("Load IDT data from:", csv_path)
        self.IDT_condition, self.IDT_fuel, self.IDT_oxidizer, IDT_mode, self.true_idt_data, self.true_idt_uncertainty = LoadIDTExpData(
            csv_path, raw_data = False, uncertainty='absolute'
        )
        if IDT_mode is not None: 
            self.IDT_mode = IDT_mode
            print('IDT_mode is updated by the csv file:', csv_path)
        assert self.IDT_mode is not None, "IDT_mode is None, please check the csv file or set IDT_mode manually"
  
    
    def getLFSconditionAndTrueDataFromCSV(self, csv_path:str,):
        """
        同 getIDTconditionAndTrueDataFromCSV
        add:
            self.LFS_condition, self.LFS_fuel, self.LFS_oxidizer, self.true_lfs_data, self.true_lfs_uncertainty 
        """
        self.LFS_condition: np.ndarray
        self.LFS_fuel: str | List[str]
        self.LFS_oxidizer: str | List[str]
        self.true_lfs_data: np.ndarray
        self.true_lfs_uncertainty: np.ndarray = None
        
        print("Load LFS data from:", csv_path)
        self.LFS_condition, self.LFS_fuel, self.LFS_oxidizer, self.true_lfs_data, self.true_lfs_uncertainty = LoadIDTExpDataWithoutMode(
            csv_path, raw_data = False, condition_prefix = 'LFS', uncertainty='absolute'
        )
   
        
    def getPSR_concentration_conditionAndTrueDataFromCSV(self, csv_path:str,):
        """
        同 getIDTconditionAndTrueDataFromCSV
        注意对机理文件格式的要求：
        1. 需要在 column 里面指定好 species 的名称：例如 species 为 ['CO', 'CO2'] 则需要有 'CO_PSR' 和 'CO2_PSR' 两列
        """
        self.PSR_concentration_condition: np.ndarray
        self.PSR_concentration_fuel: str | List[str]
        self.PSR_concentration_oxidizer: str | List[str]
        self.PSR_concentration_diluent: dict
        self.PSR_concentration_species: list
        self.true_psr_concentration_data: np.ndarray
        
        print("Load PSR concentration data from:", csv_path)
        self.PSR_concentration_condition, self.PSR_concentration_fuel, self.PSR_concentration_oxidizer, self.PSR_concentration_res_time, self.true_psr_concentration_data = LoadPSR_concentrationExpData(
            csv_path, raw_data = False,
            concentration_species=self.PSR_concentration_species,
            return_df = False
        )
        self.PSR_concentration_diluent = {} # 暂时设置为空集，不允许加载实验数据时候用这个
        self.PSR_concentration_kwargs = dict(
                        fuel = self.PSR_concentration_fuel,
                        oxidizer = self.PSR_concentration_oxidizer,
                        diluent = self.PSR_concentration_diluent,
                        species = self.PSR_concentration_species,
                        res_time = self.PSR_concentration_res_time.tolist(),
                        condition = self.PSR_concentration_condition.tolist(),
        )

def gather_apart_data_partition(partition_index, file_group, save_file_name, npz_keys, logger, rm_tree = False):
    """
    gather_apart_data_partition 为 gather_apart_data 的子函数，用于多进程
    params:
        file_group: 一个文件组，即一个进程处理的文件
        save_path: 需要 gather 的单条文件位置
        npz_keys: 需要 gather 的 npz 文件中的 keys
        logger: Log 文件的路径
    """
    data_dict = {f"all_{key}_data": [] for key in npz_keys}
    for ind, file in enumerate(file_group):
        # 获取文件名并删除文件后缀
        file_name = os.path.basename(file)
        try:
            data = np.load(file)
            for key in npz_keys:
                # new_key = key if key != 'PSR_T' else 'PSR'  # 将 PSR_T 改为 PSR
                data_dict[f"all_{key}_data"].append(data[key])
            if ind % 100 == 0: logger.info(f"single process: {file_name} file is gathered; Finished {ind/len(file_group):.2f}")
        except:
            Error = traceback.format_exc()
            logger.warning(f"single process: {file_name} file is not gathered; Reason: {Error}")
        finally:
            if rm_tree: os.remove(file)
    np.savez(save_file_name, **data_dict)
    logger.info(f"partition {partition_index}: {save_file_name} is saved. The key of data_dict is {list(data_dict.keys())}")

import warnings
def gatherAPARTdata(npz_keys = ['IDT', 'T', 'PSR_T'], apart_data_keys = ['idt', 't', 'psr'], 
                    save_path = "./data/APART_data/tmp", save_file_name = None, cpu_nums = None,
                    rm_tree = True, cover = True, logger = None, **kwargs):
    """
    使用多进程 multiprocess 来 gather data 以加速整体的速度
    基本原理为，先将保存单条数据的 tmp 文件夹中的文件按照 cpu_nums 划分为多个组，然后每个组使用一个进程来处理
    处理结束后使用0号 CPU 将每个组 gather 得到的 npz 再合并为一个 npz 文件
    params:
        npz_keys: 需要 gather 的 npz 文件中的 keys
        apart_data_keys: 需要保存的 APART_data 文件中的 keys
        save_path: 需要 gather 的单条文件位置
        save_file_name: 保存 gather 后的 npz 文件的文件名
        cpu_nums: 使用的 cpu 数量; 默认为 None, 即使用全部的 cpu
        rm_tree: 是否删除 tmp 文件夹; 默认为 True
        cover: 是否覆盖已存在的 npz 文件; 默认为 True
        logger: Log 文件的路径
    """
    if logger is None: logger = Log("./gatherAPARTdata.log")
    # Alist 一定是要收集的
    if 'Alist' not in npz_keys: npz_keys.append('Alist')
    if 'Alist' not in apart_data_keys: apart_data_keys.append('Alist')
    # 若 PSR_T 存在于 npz_keys 中，则将其改为 PSR
    if 'PSR_T' in npz_keys: npz_keys[npz_keys.index('PSR_T')] = 'PSR'
    if 'psr_T' in apart_data_keys: apart_data_keys[apart_data_keys.index('psr_T')] = 'psr'
    if cpu_nums is None: cpu_nums = max(psutil.cpu_count() - 1, 1)
    if save_file_name is None: 
        circ = kwargs.get('circ', "Not_Given")
        save_file_name = os.path.dirname(save_path) + f"/apart_data_circ={circ}.npz"
        
    data_dict = {f"all_{key}_data": [] for key in apart_data_keys}
    
    files = [file for file in os.listdir(save_path) if file.find('.npz') != -1 and os.path.exists(os.path.join(save_path, file)) and os.path.getsize(os.path.join(save_path, file)) != 0]
    if len(files) != 0:
        if cpu_nums == 1:
            for ind, file in enumerate(files):
                try:
                    tmp_file_path = os.path.join(save_path, file)
                    data = np.load(tmp_file_path)
                    for i, key in enumerate(npz_keys):
                        if key in data.files:
                            data_dict[f"all_{apart_data_keys[i]}_data"].append(data[key])
                        
                    if ind % 100 == 0: logger.info(f"single process: {ind}th file is gathered; Finished {ind/len(files):.2f}")
                except:
                    Error = traceback.format_exc()
                    logger.warning(f"single process: {ind}th file is not gathered; Reason: {Error}")
                finally:
                    if rm_tree: os.remove(tmp_file_path)
            # 将 data_dict 中的 all_Alist_data 改为 Alist
            data_dict['Alist'] = data_dict.pop('all_Alist_data')
            np.savez(save_file_name, **data_dict)
            logger.info("gatherAPARTdata finished, apart_data Saved!")
            return None
        else:
            files = np.array(
                [os.path.join(save_path, file) for file in files]
            )
            file_groups = np.array_split(files, cpu_nums)
            # 使用多进程进行处理
            pool = Pool(cpu_nums)
            for partition_index, file_group in enumerate(file_groups):
                tmp_save_file_name = save_file_name.replace('.npz', f"_partition_index={partition_index}.npz")
                pool.apply_async(gather_apart_data_partition, 
                                 args = (partition_index, file_group, tmp_save_file_name, npz_keys, logger),
                                 kwds = {"rm_tree": rm_tree})
            pool.close()
            pool.join()
    # 使用 0 号 CPU 进行 gather
    for partition_index in range(cpu_nums):
        sucessful_gather = True
        tmp_save_file_name = save_file_name.replace('.npz', f"_partition_index={partition_index}.npz")
        if not os.path.exists(tmp_save_file_name):
            logger.warning(f"partition {partition_index} file {tmp_save_file_name} not found, skip this partition")
            continue
        tmp_data = np.load(tmp_save_file_name)
        for i, key in enumerate(npz_keys):
            if f"all_{key}_data" in list(tmp_data.files):
                data_dict[f"all_{apart_data_keys[i]}_data"].extend(tmp_data[f"all_{key}_data"])
            elif f"all_{key.lower()}_data" in list(tmp_data.files):
                data_dict[f"all_{apart_data_keys[i]}_data"].extend(tmp_data[f"all_{key.lower()}_data"])
            else:
                warnings.warn(f"key all_{key}_data not found in {tmp_save_file_name}, available keys: {tmp_data.files}")
                sucessful_gather = False
                break
        # if not sucessful_gather:
        #     for key in tmp_data.files:
        #         new_key = key.lower() if key != 'T' else 'T'; data_dict[new_key].extend(tmp_data[key])
        os.remove(tmp_save_file_name)
    # 将 data_dict 中的 all_Alist_data 改为 Alist
    data_dict['Alist'] = data_dict.pop('all_Alist_data')
    np.savez(save_file_name, **data_dict)
    logger.info("gatherAPARTdata finished, apart_data Saved!")
    if rm_tree:
        # 不使用 rm -rf 的原因是因为 rm -rf 删除大批量文件的效率太低
        # 使用 rsync 命令删除文件夹
        logger.info(f"removing the tmp files in {save_path}"); time0 = time.time()
        mkdirplus("./data/APART_data/blank_dir")
        subprocess.run(f"rsync --delete-before -d -a ./data/APART_data/blank_dir/ {save_path}/", shell = True)
        logger.info(f"removing the tmp files in {save_path} finished; Please check. Cost {time.time() - time0:.2f} s")
    
    return None
    # else:
    #     logger.warning("gatherAPARTdata function is out-of-commision")


@func_set_timeout(300)
def _GenOneIDT(IDT_condition: np.ndarray, fuel:str, oxidizer: str, tmp_chem_path:str, index:int, my_logger:Log, 
               IDT_mode = 0, idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, cut_time_alpha = 10, **kwargs):
        """
        使用 yaml2idt 生成单条 IDT 数据
        params:
            Alist: 输入的 A 列表
            index: 生成数据的序号
            IDT_condition; 
            idt_arrays 或者 cut_time
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
            my_logger: 返回信息的 log 文件位置
            kwargs:
                remove_chem: 是否将临时文件删除; 默认为 TRUE
                cut_time_alpha: idt_array 前的倍数，默认为 10
            return:
                若报错 return index 的值
                若不报错返回 IDT, Temperature
                取消了在此处保存 npz 文件的步骤
        """
        os.environ['OMP_NUM_THREADS'] = '1'
        if not cut_time is None: idt_arrays = cut_time
        idt_arrays = idt_arrays * np.ones(len(IDT_condition)) if not isinstance(idt_arrays, Iterable) else idt_arrays

        try: # 生成简化机理在目标状态点上点火延迟时间
            t0 = time.time()
            IDT, Temperature = yaml2idt(
                    tmp_chem_path, mode = IDT_mode, cut_time = cut_time_alpha * idt_arrays, IDT_condition = IDT_condition,
                    fuel = fuel, oxidizer = oxidizer, **kwargs
                    ) 
            
        except Exception as r:
            # 返回 Error 的 Index
            fmt = traceback.format_exc()
            my_logger.info(f'_GenOneIDT: mechanism {index} cantera simulation error; error reason:{r}, cost {time.time()-t0:.2f} s')
            my_logger.info(fmt)
            return index
        else:
            return np.array(IDT).flatten(), np.array(Temperature)


@func_set_timeout(300)
def _GenOneDIDT(IDT_condition: np.ndarray, fuel:str, oxidizer: str, tmp_chem_path:str, index:int, my_logger:Log, 
               idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, cut_time_alpha = 10, **kwargs):
        """
        使用 yaml2idt 生成单条 IDT 数据
        params:
            Alist: 输入的 A 列表
            index: 生成数据的序号
            IDT_condition; 
            idt_arrays 或者 cut_time
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
            my_logger: 返回信息的 log 文件位置
            kwargs:
                remove_chem: 是否将临时文件删除; 默认为 TRUE
                cut_time_alpha: idt_array 前的倍数，默认为 10
            return:
                若报错 return index 的值
                若不报错返回 IDT, Temperature
                取消了在此处保存 npz 文件的步骤
        """
        os.environ['OMP_NUM_THREADS'] = '1'
        if not cut_time is None: idt_arrays = cut_time
        idt_arrays = idt_arrays * np.ones(len(IDT_condition)) if not isinstance(idt_arrays, Iterable) else idt_arrays

        try: # 生成简化机理在目标状态点上点火延迟时间
            t0 = time.time()
            DIDT, Temperature = yaml2didt(
                    tmp_chem_path, cut_time = cut_time_alpha * idt_arrays, DIDT_condition = IDT_condition,
                    fuel = fuel, oxidizer = oxidizer, **kwargs
                    ) 
            
        except Exception as r:
            # 返回 Error 的 Index
            fmt = traceback.format_exc()
            my_logger.info(f'_GenOneIDT: mechanism {index} cantera simulation error; error reason:{r}, cost {time.time()-t0:.2f} s')
            my_logger.info(fmt)
            return index
        else:
            return np.array(DIDT).flatten(), np.array(Temperature)


@func_set_timeout(100)
def _GenOnePSR(PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, fuel:str, oxidizer: str, tmp_chem_path:str, index:int, 
                my_logger:Log, psr_error_tol = 50, **kwargs):
        """
        使用 yaml2psr 生成单条 PSR 数据
        params:
            index: 生成数据的序号
            args: 请输入所有的参数 (APART_args), 此处使用的是 PSR_condition
                本函数使用到的参数如下：
                PSR_condition; RES_TIME_LIST
                fuel; oxidizer
                eq_dict; reduced_mech
                tmp_chem
            Alist: 输入的 A 列表
            my_logger: 返回信息的 log 文件位置
            remove_chem: 是否将临时文件删除; 默认为 TRUE
            psr_error_tol: solve psr 中的 error tol，默认为 50
            return:
                若报错 return index 的值
                若不报错返回 IDT, Temperature
                取消了在此处保存 npz 文件的步骤
        """
        os.environ['OMP_NUM_THREADS'] = '1'
        try: # 生成简化机理在目标状态点上PSR
            t0 = time.time()    
            PSR_T = yaml2psr(
                        tmp_chem_path, PSR_condition = PSR_condition, RES_TIME_LIST = RES_TIME_LIST, error_tol = psr_error_tol, 
                        fuel = fuel, oxidizer = oxidizer, **kwargs
                    )   
        except Exception:
            # 返回 Error 的 Index
            exstr = traceback.format_exc()
            my_logger.info(f'_GenOnePSR: mechanism {index}: IDT cantera simulation error; error reason:{exstr}, cost {time.time()-t0:.2f} s')
            return index
        else:
            return PSR_T


@func_set_timeout(300)
def _GenOneMole(species, mole_condition: np.ndarray, fuel:str, oxidizer: str, tmp_chem_path:str, index:int, my_logger:Log,
                       cut_time_array: np.ndarray = 1, cut_time_alpha = 10, cut_time = None, **kwargs):
        """
        使用 yaml2mole_time 生成单条 mole 数据
        params:
            index: 生成数据的序号
            mole_condition; 
            cut_time_array
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
            Alist: 输入的 A 列表
            my_logger: 返回信息的 log 文件位置
            remove_chem: 是否将临时文件删除; 默认为 TRUE
            cut_time_alpha: idt_array 前的倍数，默认为 10
            return:
                若报错 return index 的值
                若不报错返回 mole, Temperature
                取消了在此处保存 npz 文件的步骤
        """
        os.environ['OMP_NUM_THREADS'] = '1'
        if not cut_time is None: cut_time_array = cut_time
        cut_time_array = cut_time_array * np.ones(len(mole_condition)) if not isinstance(cut_time_array, Iterable) else cut_time_array
        try: # 生成简化机理在目标状态点上点火延迟时间
            t0 = time.time()   
            mole = yaml2mole_time(
                tmp_chem_path, species = species, mode = 'get_max', cut_time = cut_time_alpha * cut_time_array,
                        fuel = fuel, oxidizer = oxidizer, **kwargs
            )
        except Exception as r:
            # 返回 Error 的 Index
            fmt = traceback.format_exc()
            my_logger.info(f'_GenOneMole: mechanism {index} PSR cantera simulation error; error reason:{r}, cost {time.time()-t0:.2f} s')
            my_logger.info(fmt)
            return index
        else:
            return mole


def _GenOneLFS(LFS_condition: np.ndarray, fuel:str, oxidizer: str, tmp_chem_path:str, index:int,  my_logger:Log, **kwargs):
        """
        使用 yaml2FS 生成单条 LFS 数据;
        params:
            Alist: 输入的 A 列表
            index: 生成数据的序号
            LFS_condition; 
            idt_arrays 或者 cut_time
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
            my_logger: 返回信息的 log 文件位置
            kwargs: 其他参数应用在 solve_flame_speed 内
            return:
                若报错 return index 的值
                若不报错返回 LFS, Temperature
                取消了在此处保存 npz 文件的步骤
        """
        os.environ['OMP_NUM_THREADS'] = '1'
        try:
            t0 = time.time() 
            LFS = yaml2FS(
                tmp_chem_path,
                LFS_condition,
                fuel = fuel, oxidizer = oxidizer,
                **kwargs
            )
            
        except Exception as r:
            # 返回 Error 的 Index
            fmt = traceback.format_exc()
            my_logger.info(f'_GenOneLFS: mechanism {index} LFS cantera simulation error; error reason:{r}, cost {time.time()-t0:.2f} s')
            my_logger.info(fmt)
            return index
        else:
            return LFS


@func_set_timeout(300)
def _GenOnePSR_plus_PSRex(PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, fuel:str, oxidizer: str, tmp_chem_path:str, 
                          index:int,  my_logger:Log, psr_error_tol = 50, **kwargs):
    """
    生成单条 PSR + PSRex 数据; 使用 yaml2PSR_plus_PSRex 函数;
    由于函数本身的性质，要求 RES_TIME_LIST 具有很强的连续性，RES_TIME_LIST 的建议长度为 30 以上以保证 PSRex 的计算正确度
    params:
        index: 生成数据的序号
        args: 请输入所有的参数 (APART_args), 此处使用的是 PSR_condition
            本函数使用到的参数如下：
            PSR_condition; RES_TIME_LIST
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
        Alist: 输入的 A 列表
        my_logger: 返回信息的 log 文件位置
        remove_chem: 是否将临时文件删除; 默认为 TRUE
        psr_error_tol: solve psr 中的 error tol，默认为 50
        return:
            若报错 return index 的值
            若不报错返回 IDT, Temperature
            取消了在此处保存 npz 文件的步骤
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    # 为了保证 yaml2PSR_plus_PSRex 的一致性，需要调整 init_res_time
    RES_TIME_LIST = RES_TIME_LIST.tolist()
    try: # 生成简化机理在目标状态点上PSR
        t0 = time.time()    
        PSR_T, PSR_extinction = yaml2PSR_plus_PSRex(
            tmp_chem_path, RES_TIME_LIST, PSR_condition, psr_error_tol = psr_error_tol, fuel = fuel, oxidizer = oxidizer, **kwargs
            )
    except Exception as r:
        # 返回 Error 的 Index
        my_logger.info(f'_GenOnePSR_plus_PSRex: mechanism {index} PSRex cantera simulation error; error reason:{r}, cost {time.time()-t0:.2f} s')
        return index
    else:
        return PSR_T, PSR_extinction


@func_set_timeout(300)
def _GenOnePSR_with_extinction(PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, 
                              fuel:str, oxidizer: str, tmp_chem_path:str, index:int,  my_logger:Log, 
                              psr_error_tol = 50, PSR_EXP_FACTOR = 0.5, **kwargs):
    """
    生成单条 PSR + PSRex 数据; 使用 yaml2PSR_plus_PSRex 函数;
    由于函数本身的性质，要求 RES_TIME_LIST 具有很强的连续性，RES_TIME_LIST 的建议长度为 30 以上以保证 PSRex 的计算正确度
    params:
        index: 生成数据的序号
        args: 请输入所有的参数 (APART_args), 此处使用的是 PSR_condition
            本函数使用到的参数如下：
            PSR_condition; RES_TIME_LIST
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
        Alist: 输入的 A 列表
        my_logger: 返回信息的 log 文件位置
        remove_chem: 是否将临时文件删除; 默认为 TRUE
        psr_error_tol: solve psr 中的 error tol，默认为 50
        return:
            若报错 return index 的值
            若不报错返回 IDT, Temperature
            取消了在此处保存 npz 文件的步骤
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    exp_factor = 2 ** PSR_EXP_FACTOR
    try: # 生成简化机理在目标状态点上PSR
        t0 = time.time()    
        PSR_T, PSR_extinction = yaml2psr_with_extinction_time(
            tmp_chem_path, RES_TIME_LIST, PSR_condition, psr_error_tol = psr_error_tol,
            fuel = fuel, oxidizer = oxidizer, exp_factor = exp_factor, **kwargs
            )
    except Exception as r:
        # 返回 Error 的 Index
        my_logger.info(f'GenOnePSR_with_extinction: mechanism {index} PSRex cantera simulation error; error reason:{r}, cost {time.time()-t0:.2f} s')
        return index
    else:
        return PSR_T, PSR_extinction


@func_set_timeout(300)
def _GenOnePSR_concentration(tmp_chem_path:str, species:list, condition: np.ndarray, res_time: np.ndarray, fuel:str, oxidizer: str, diluent, index:int,  my_logger:Log, **kwargs):
    """
    使用静态方法生成单条 PSR 稳态状态下的物质浓度分数数据; 
    params:
        concentration_species: 需要计算浓度分数的物质列表
        index: 生成数据的序号
        args: 请输入所有的参数 (APART_args), 此处使用的是 PSR_condition
            本函数使用到的参数如下：
            PSR_condition; concentration_res_time
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
        Alist: 输入的 A 列表
        my_logger: 返回信息的 log 文件位置
        remove_chem: 是否将临时文件删除; 默认为 TRUE
        psr_error_tol: solve psr 中的 error tol，默认为 50
        return:
            若报错 return index 的值
            若不报错返回 IDT, Temperature
            取消了在此处保存 npz 文件的步骤
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    try: # 生成简化机理在目标状态点上点火延迟时间
        t0 = time.time()   
        PSR_concentration = yaml2PSR_concentration(
            tmp_chem_path, species, condition, res_time, fuel = fuel, oxidizer = oxidizer, diluent = diluent, 
            return_list = True, return_condition=False,**kwargs
        )
    except Exception as r:
        # 返回 Error 的 Index
        error = traceback.format_exc()
        my_logger.info(f'GenOnePSR_with_concentration: mechanism {index} PSR/PSRex cantera simulation error; error reason:{error}, cost {time.time()-t0:.2f} s')
        return index
    else:
        return PSR_concentration


@func_set_timeout(320)
def _GenOneIDT_HRR(IDT_condition: np.ndarray,fuel:str, oxidizer: str, tmp_chem_path:str, index:int, my_logger:Log, 
                       IDT_mode = 0, idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                       cut_time_alpha = 10, idt_defined_T_diff = 400, time_multiple = 2, **kwargs):
        """
        使用静态方法生成单条 IDT 数据; 与 APART 内不同的在于使用了 Adict2yaml 函数
        params:
            Alist: 输入的 A 列表
            index: 生成数据的序号
            IDT_condition; 
            idt_arrays 或者 cut_time
            fuel; oxidizer
            eq_dict; reduced_mech
            tmp_chem
            my_logger: 返回信息的 log 文件位置
            kwargs:
                remove_chem: 是否将临时文件删除; 默认为 TRUE
                cut_time_alpha: idt_array 前的倍数，默认为 10
            return:
                若报错 return index 的值
                若不报错返回 IDT, Temperature
                取消了在此处保存 npz 文件的步骤
        """
        os.environ['OMP_NUM_THREADS'] = '1'
        if not cut_time is None: idt_arrays = cut_time
        idt_arrays = idt_arrays * np.ones(len(IDT_condition)) if not isinstance(idt_arrays, Iterable) else idt_arrays
        try: # 生成简化机理在目标状态点上点火延迟时间
            t0 = time.time()
            IDT, Maxhrr, Temperature = yaml2idt_hrr(
                    tmp_chem_path, cut_time = cut_time_alpha * idt_arrays, IDT_condition = IDT_condition,
                    fuel = fuel, oxidizer = oxidizer, IDT_mode = IDT_mode, idt_defined_T_diff = idt_defined_T_diff, time_multiple = time_multiple, **kwargs
                    ) 
        except Exception as r:
            # 返回 Error 的 Index
            fmt = traceback.format_exc()
            my_logger.info(f'_GenOneIDT_HRR: mechanism {index} cantera simulation error; error reason:{r}, cost {time.time()-t0:.2f} s')
            my_logger.info(fmt)
            return index
        else:
            return np.array(IDT).flatten(), np.array(Maxhrr), np.array(Temperature)
