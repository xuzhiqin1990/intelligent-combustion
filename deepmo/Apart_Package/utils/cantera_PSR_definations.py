# -*- coding:utf-8 -*-
import cantera as ct
import numpy as np
import warnings, json, traceback
from .setting_utils import get_yaml_data
from .cantera_utils import solve_psr
from collections.abc import Iterable

"""========================================================================================"""
"""=================================== PSR Definition ========================================="""
"""========================================================================================"""

def solve_psr_extinction_time(gas, PSR_condition, fuel, oxidizer, psr_tol = 10, init_res_time:float = 1,
                            exp_factor:int | float = 2, need_PSR_T:bool = False, RES_TIME_LIST:list = None) -> float:
    """
    求解 PSR 的熄火极限时间
    若给定 exp_factor:
        采取的方式是不断 / exp_factor 地缩短反应器的反应时间，直到熄火为止
        注意这里的 init_res_time 是指的第一次的反应时间，后面的反应时间都是在上一次的基础上除以 exp_factor，因此不要设置太过大的值
        合理的设置方式是设置为简化机理能点火的中值开始尝试，如果熄火了，就除以 exp_factor，如果没有熄火，就增加一倍，直到找到熄火的反应时间
    若给定 RES_TIME_LIST:
        采取的方式是直接使用 RES_TIME_LIST 中的反应时间，如果熄火了，就返回该反应时间，如果没有熄火，就返回最后一个反应时间
        如果想要获取 RES_TIME_LIST 之外的反应时间，请使用 solve_psr 和 solve_psr_extinction_time 结合。这是因为经过尝试发现，
        如果我们最后的 RES_TIME = t 不能点火, 即使在 2t 时间能够点火，使用 cantera 逐渐增加点火时间，也会出现在 t 依然被判定熄火的情况
    params:
        gas: cantera 的气体对象
        PSR_condition: PSR 的工况，包括 phi, T, p
        fuel: 燃料名称
        oxidizer: 氧化剂名称
        psr_tol: 熄火的温度容差
        init_res_time: 初始的反应时间
        exp_factor: 缩短反应时间的因子
        need_PSR_T: 是否需要返回 PSR 的温度 PSR_Tlist
        RES_TIME_LIST: 一维列表; 可以指定 RES_TIME_LIST，如果指定了 RES_TIME_LIST 则不会自动计算，会直接筛选满足的 PSR_Tlist
    return:
        residence_time: 熄火的反应时间 or (residence_time, PSR_Tlist, RES_TIME_LIST)
    """
    phi, T, p = PSR_condition
    gas.TP = T, p * ct.one_atm; gas.set_equivalence_ratio(phi, fuel, oxidizer)
    # inlet -> inlet_mfc -> combustor -> outlet_mfc -> exhaust
    inlet = ct.Reservoir(gas)               # 进气口预混箱
    gas.equilibrate('HP')
    combustor = ct.IdealGasReactor(gas, volume = 1.0)
    exhaust = ct.Reservoir(gas)
    def mdot(t):
        return combustor.mass/residence_time
    inlet_mfc = ct.MassFlowController(upstream = inlet, downstream = combustor, mdot = mdot)
    outlet_mfc = ct.PressureController(upstream = combustor, downstream = exhaust, master=inlet_mfc, K=0.01)
    sim = ct.ReactorNet([combustor])

    PSR_Tlist = []
    if RES_TIME_LIST is None:
        residence_time = init_res_time
        RES_TIME_LIST = [init_res_time]
        sim.set_initial_time(0)
        sim.advance_to_steady_state()
        PSR_Tlist.append(combustor.T)
        # 初始化 extinction_time
        while combustor.T > T + psr_tol:
            residence_time /= exp_factor
            sim.set_initial_time(0)
            try:
                sim.advance_to_steady_state()
                PSR_Tlist.append(combustor.T)
                RES_TIME_LIST.append(residence_time)
            except:
                residence_time = residence_time * exp_factor
                break
        else:
            residence_time *= exp_factor
    else:
        RES_TIME_LIST = np.array(RES_TIME_LIST).tolist()
        residence_time = RES_TIME_LIST.pop(0); previous_residence_time = residence_time
        sim.set_initial_time(0)
        sim.advance_to_steady_state()
        PSR_Tlist.append(combustor.T)
        # 初始化 extinction_time
        while combustor.T > T + psr_tol and RES_TIME_LIST != []:
            residence_time = RES_TIME_LIST[0]
            sim.set_initial_time(0)
            try:
                sim.advance_to_steady_state()   
            except:
                traceback.print_exc()
                residence_time = residence_time * exp_factor 
                break   
            if combustor.T < T + psr_tol:
                residence_time = previous_residence_time
                break
            else:
                previous_residence_time = RES_TIME_LIST.pop(0)
                PSR_Tlist.append(combustor.T)
        if RES_TIME_LIST != []: 
            warnings.warn(f'RES_TIME_LIST 未完全使用，剩余部分为 {RES_TIME_LIST}, residence_time is {residence_time}, 当前反应器温度为 {combustor.T}')
            PSR_Tlist.extend([combustor.T] * len(RES_TIME_LIST))
        else:
            while combustor.T > T + psr_tol:
                residence_time /= exp_factor
                sim.set_initial_time(0)
                try:
                    sim.advance_to_steady_state()
                except:
                    residence_time = residence_time * exp_factor
                    break
            else:
                residence_time *= exp_factor
    # 0230728: 目前发现如果以最小值为 res_time，可能出现 PSR 在 PSRex 点不上火的情况，因此这里采用了一个简单的方法增加他的熄火极限值
    # Date 0804: 删除
    if need_PSR_T:
        return residence_time, PSR_Tlist, RES_TIME_LIST
    return residence_time


def yaml2psr_extinction_time(chem_file, PSR_condition:np.ndarray = None, PSR_T = None, PSR_P = None, PSR_phi = None,
                             fuel:str|list = None, oxidizer:str|list = None, psr_tol = 10, init_res_time:float | list = 1.0, 
                             setup_file = None, exp_factor = 2, **kwargs):
    """
    直接根据 yaml 计算 PSR 的熄火极限时间
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        PSR_T, PSR_P, PSR_phi = chem_args['PSR_T'], chem_args['PSR_P'], chem_args['PSR_phi']
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif PSR_condition is None:
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
    # 单值化快速处理
    gas = ct.Solution(chem_file)
    if PSR_condition.ndim == 1:
        extinction_time = solve_psr_extinction_time(
            gas, PSR_condition, fuel, oxidizer, psr_tol, init_res_time, exp_factor
        )
    else:
        # 若 init_res_time 是 scalar 则转化为和 PSR condition 相同维度的数组
        if isinstance(init_res_time, float): init_res_time = np.ones(len(PSR_condition)) * init_res_time
        # fuel 和 oxidizer 也相同处理
        fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
        extinction_time = np.array(
            [solve_psr_extinction_time(gas, PSR_condition[i], fuel[i], oxidizer[i], 
                psr_tol, init_res_time[i], exp_factor = exp_factor) for i in range(len(PSR_condition))]
        )
    return extinction_time


def yaml2psr_with_extinction_time(chem_file, RES_TIME_LIST, PSR_condition:np.ndarray = None, 
                                  PSR_T = None, PSR_P = None, PSR_phi = None, fuel = None, oxidizer = None, 
                                  psr_tol = 10, init_res_time = None, setup_file = None, exp_factor = 2, **kwargs):
    """
    一起计算 PSR 对应的温度和熄火极限时间，适合计算精确点火极限; 
    注意这里的计算过程是先计算 PSR 的温度，然后将 tmp_res_time[np.where(psr - T < psr_tol) 设为再计算熄火极限时间
    与 yaml2PSR_plus_PSRex 不同，适用于较短 RES_TIME_LIST 的情况并给定 exp_factor 的值
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        PSR_T, PSR_P, PSR_phi = chem_args['PSR_T'], chem_args['PSR_P'], chem_args['PSR_phi']
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif PSR_condition is None:
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
    
    assert RES_TIME_LIST is not None, '请设置 RES_TIME_LIST'
    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'

    fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
    oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
    PSR, PSR_extinction_time = [], []; gas = ct.Solution(chem_file)

    for tmp_res_time, condition, tmp_fuel, tmp_oxidizer in zip(RES_TIME_LIST, PSR_condition, fuel, oxidizer):
        phi, T, P = condition
        psr = solve_psr(gas, tmp_res_time, T = T, P = P, phi = phi, fuel = tmp_fuel, oxidizer = tmp_oxidizer, **kwargs)
        # init_res_time 定义为 psr[index] - T > psr_tol 的最大 index 对应的 tmp_res_time
        try:
            init_res_time = tmp_res_time[np.where(psr - T >= psr_tol)[0][-1]]
        except:
            init_res_time = tmp_res_time[0]
        psr_extinction_time = solve_psr_extinction_time(gas, condition, tmp_fuel, tmp_oxidizer, 
                                                        psr_tol, init_res_time, exp_factor = exp_factor)
        PSR.extend(psr)
        PSR_extinction_time.append(psr_extinction_time)
        

    return PSR, PSR_extinction_time


def yaml2PSR_plus_PSRex(chem_file, RES_TIME_LIST, PSR_condition:np.ndarray = None, 
                        PSR_T = None, PSR_P = None, PSR_phi = None, fuel = None, oxidizer = None, 
                        psr_error_tol = 10, init_res_time = None, setup_file = None, exp_factor = 2, **kwargs):
    """
    通过给定 RES_TIME_LIST 的方式一起计算 PSR 对应的温度和熄火极限时间，不能计算精确熄火极限; 
    注意这里的计算过程是并行的，同时计算两者，与 yaml2psr_with_extinction_time 不同，适用于较长 RES_TIME_LIST 的情况
    return PSR, PSR_extinction_time
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        PSR_T, PSR_P, PSR_phi = chem_args['PSR_T'], chem_args['PSR_P'], chem_args['PSR_phi']
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif PSR_condition is None:
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
    
    assert RES_TIME_LIST is not None, '请设置 RES_TIME_LIST'
    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'

    fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
    oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
    init_res_time = [init_res_time] * len(PSR_condition) if not isinstance(init_res_time, Iterable) else init_res_time
    PSR, PSR_extinction_time = [], []; gas = ct.Solution(chem_file)

    for tmp_res_time, condition, tmp_fuel, tmp_oxidizer, tmp_init_res_time in zip(RES_TIME_LIST, PSR_condition, fuel, oxidizer, init_res_time):
        psr_extinction_time, psr, _ = solve_psr_extinction_time(
            gas, condition, tmp_fuel, tmp_oxidizer, psr_error_tol, tmp_init_res_time, 
            exp_factor = exp_factor, need_PSR_T = True, RES_TIME_LIST = tmp_res_time
        )
        PSR.extend(psr); PSR_extinction_time.append(psr_extinction_time)
    
    return PSR, PSR_extinction_time


def yaml2PSRex_sensitivity(chem_file, PSR_condition = None, delta = 0.1, PSR_T = None, PSR_P = None, PSR_phi = None,
                             fuel:str|list = None, oxidizer:str|list = None, psr_tol = 10, init_res_time:float | list = 1.0, 
                             setup_file = None, exp_factor = 2, specific_reactions = None, save_path = None, 
                             need_base_PSRex = False, **kwargs):
    """
    计算 PSR_extinction_time 的敏感度; 由于一般而言该数值都不是非常敏感，因此在计算敏感度时选取了较大的步长范围，
    可以使用该函数搭配其他的函数进行多步搜索
    params:
        除了 delta = 0.1 之外都是 yaml2psr_extinction_time 的参数
        delta: 用于计算敏感度的步长
        specific_reactions: 用于计算敏感度的方程组，如果为 None 则使用默认的方程组
        save_path: 保存路径
        need_base_PSRex: 是否需要计算基准的 PSR extinction time
    return:
        PSRex_sensitivity: PSR extinction time 的敏感度
    """
    np.set_printoptions(precision=2, suppress=True)
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        PSR_T, PSR_P, PSR_phi = chem_args['PSR_T'], chem_args['PSR_P'], chem_args['PSR_phi']
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif PSR_condition is None:
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )

    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    gas = ct.Solution(chem_file)
    if specific_reactions is None: specific_reactions = gas.reaction_equations()    
    PSRex_sensitivity = {}
    base_PSRex = yaml2psr_extinction_time(
        chem_file, PSR_condition, PSR_T, PSR_P, PSR_phi, fuel, oxidizer, psr_tol, init_res_time, setup_file, exp_factor, **kwargs
    )    
    # 单值下快速运算
    if PSR_condition.ndim == 1:
        for m in range(gas.n_reactions):
            equation = gas.reaction(m).equation
            if equation not in specific_reactions: continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            psrex = solve_psr_extinction_time(
                gas, PSR_condition, fuel, oxidizer, psr_tol, init_res_time, exp_factor
            )
            PSRex_sensitivity.update(
                {equation: ((psrex - base_PSRex) / (delta * base_PSRex)).tolist()}
            )            
    else:
        fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
        for m in range(gas.n_reactions):
            equation = gas.reaction(m).equation
            if equation not in specific_reactions: continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            tmp_PSRex = []
            for condition, tmp_fuel, tmp_oxidizer in zip(PSR_condition, fuel, oxidizer):
                phi, T, P = condition
                gas.TP = T, P * ct.one_atm
                gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
                tmp_psrex = solve_psr_extinction_time(
                    gas, condition, tmp_fuel, tmp_oxidizer, psr_tol, init_res_time, exp_factor
                )
                tmp_PSRex.append(tmp_psrex)
            tmp_PSRex = np.array(tmp_PSRex)
            if equation in PSRex_sensitivity:
                # 求两者的最大值作为敏感度
                PSRex_sensitivity[equation] = np.maximum(PSRex_sensitivity[equation], (tmp_PSRex - base_PSRex) / (delta * base_PSRex)).tolist()
            else:
                PSRex_sensitivity.update(
                    {equation: ((tmp_PSRex - base_PSRex) / (delta * base_PSRex)).tolist()}
                )     
    if not save_path is None:
        # 使用 jSON 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(PSRex_sensitivity, ensure_ascii=False, indent=4, separators=(',', ':')))

    if need_base_PSRex: return PSRex_sensitivity, base_PSRex
    return PSRex_sensitivity


"""========================================================================================"""
"""=================================== PSR Concentration ========================================="""
"""========================================================================================"""

# def solve_psr_concentration(gas, residence_time_list, PSR_condition = None,
#                             T = None, P = None, phi = None, 
#                             fuel='nC7H16', 
#                             oxidizer='O2:1, N2:3.76', 
#                             species_list = None):
#     r'''
#     返回给定条件(T, P, phi)和residence_time情况下psr反应器反应至稳态时species_list的组分浓度
#     args:
#         gas: cantera的气体对象
#         residence_time_list: 反应时间列表
#         PSR_condition: PSR的工况，包括 phi, T, p
#         fuel: 燃料名称
#         oxidizer: 氧化剂名称
#         species_list: 希望返回组分浓度的组分列表
#     return:
#         concentration_list: n_species * n_residence_time的列表
#     '''
#     species_list = [fuel, "O2", "OH"] if species_list is None else species_list
#     Reduced_T_list = []
#     concentration_list = [[] for _ in range(len(species_list))]
#     if PSR_condition is not None:
#         phi, T, P = PSR_condition
#     gas.TP = T, P * ct.one_atm
#     gas.set_equivalence_ratio(phi, fuel, oxidizer)

#     # # 刚开始设定反应时间足够长，让火焰烧起来
#     # residence_time = 1

#     # inlet -> inlet_mfc -> combustor -> outlet_mfc -> exhaust
#     inlet = ct.Reservoir(gas)               # 进气口预混箱
#     gas.equilibrate('HP')
#     combustor = ct.IdealGasReactor(gas, volume=1.0)
#     exhaust = ct.Reservoir(gas)

#     def mdot(t):
#         return combustor.mass/residence_time
#     inlet_mfc = ct.MassFlowController(
#         upstream=inlet, downstream=combustor, mdot=mdot)
#     outlet_mfc = ct.PressureController(
#         upstream=combustor, downstream=exhaust, master=inlet_mfc, K=0.01)
#     sim = ct.ReactorNet([combustor])

#     # sim.set_initial_time(0)
#     # sim.advance_to_steady_state()
#     for residence_time in residence_time_list:
#         sim.set_initial_time(0)
#         sim.advance_to_steady_state()
#         Reduced_T_list.append(combustor.T)
#         for t, s in enumerate(species_list):
#             concentration_list[t].append(combustor.thermo[s].X[0])

#     return concentration_list


# def yaml2PSR_concentration(chem_file, RES_TIME_LIST, PSR_condition = None, species_list = None, PSR_T = None, PSR_P = None, PSR_phi = None,
#                             fuel:str|list = None, oxidizer:str|list = None, setup_file = None, **kwargs):
#     """
#     配套懒人版 solve_psr_concentration
#     """
#     if setup_file != None:
#         chem_args = get_yaml_data(setup_file)
#         PSR_T, PSR_P, PSR_phi = chem_args['PSR_T'], chem_args['PSR_P'], chem_args['PSR_phi']
#         PSR_condition = np.array(
#                     [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
#                 )
#         fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
#     elif PSR_condition is None:
#         PSR_condition = np.array(
#                     [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
#                 )    

#     gas = ct.Solution(chem_file)
#     if species_list is None: species_list = [fuel, "O2", "OH"]
#     fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
#     oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
#     PSR_concentration = []
#     for tmp_res_time, condition, tmp_fuel, tmp_oxidizer in zip(RES_TIME_LIST, PSR_condition, fuel, oxidizer):
#         phi, T, P = condition
#         tmp_concentration = solve_psr_concentration(
#             gas, tmp_res_time, T = T, P = P, phi = phi, fuel = tmp_fuel, oxidizer = tmp_oxidizer, species_list = species_list, **kwargs
#         )
#         PSR_concentration.append(tmp_concentration)
#     return np.array(PSR_concentration)


"""========================================================================================"""
"""=================================== PSR Curve similarity measure ======================="""
"""========================================================================================"""

def yaml2PSRCurveSimilarityMeasure(chem_file:str, measurebased_data: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, 
                                   setup_file:str = None, PSR_T = None, PSR_P = None, PSR_phi = None,
                                   fuel:str = None, oxidizer:str = None, psr_tol = 10, ):
    """
    用于衡量 chem_file 计算得到的 PSR 和 measurebased_data 的相似度; 相似度的计算方法是，计算 chem_file_PSR - measurebased_data_PSR 后
    再计算该差值关于时间的差商；将差商沿时间求绝对值和，和越小相似度越高
    Similarity = sum(abs(diff(chem_file_PSR - measurebased_data_PSR, t))) / len(t)
    params:
        chem_file: chem_file 的路径
        measurebased_data: 用于比较的数据，一般是详细机理的数据
            shape: (RES_TIME_LIST.shape)
        PSR_condition: 2D array, PSR 的工况，包括 phi, T, P
        RES_TIME_LIST: 可以指定 RES_TIME_LIST
        setup_file: yaml 文件的路径，如果指定了 setup_file 则会使用 setup_file 中的 PSR_condition
        PSR_T: PSR 的温度列表
        PSR_P: PSR 的压力列表
        PSR_phi: PSR 的 phi 列表
        fuel: 燃料名称
        oxidizer: 氧化剂名称
        psr_tol: 熄火的温度容差
    """ 
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        PSR_T, PSR_P, PSR_phi = chem_args['PSR_T'], chem_args['PSR_P'], chem_args['PSR_phi']
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif PSR_condition is None:
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    assert RES_TIME_LIST is not None, '请设置 RES_TIME_LIST'
    gas = ct.Solution(chem_file); DIFFs = []; PSR_T = []
    for condition, res_time_list, tmp_measurebased_data in zip(PSR_condition, RES_TIME_LIST, measurebased_data):
        phi, T, P = condition
        tmp_PSR = solve_psr(gas, res_time_list, T = T, P = P, phi = phi, fuel = fuel, oxidizer = oxidizer, error_tol = psr_tol)
        tmp_diff = tmp_PSR - tmp_measurebased_data
        # 计算 tmp_diff 的一阶差商
        tmp_diff = np.abs(np.diff(tmp_diff))
        tmp_diff = np.sum(tmp_diff) / len(tmp_diff)
        DIFFs.append(tmp_diff); PSR_T.extend(tmp_PSR)
    # DIFFs 求平均值
    return np.mean(DIFFs), PSR_T

