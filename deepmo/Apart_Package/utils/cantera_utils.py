# -*- coding:utf-8 -*-
import re
import cantera as ct
import numpy as np
import json, traceback, time
from .setting_utils import get_yaml_data
from .cantera_IDT_definations import idt_definition_CH3_max, idt_definition_OH_half_max, idt_definition_OH_max, idt_definition_OH_slope_max, idt_definition_OH_slope_max_intersect, idt_definition_pressure_slope_max, idt_definition_pressure_slope_max_intersect, didt_definition_Temperature_Threshold

from collections.abc import Iterable
from typing import Union, List


'''============================================================================================================='''
'''                                                    Utils                                                    '''
'''============================================================================================================='''

def get_optim_point(list1, time, need_index = False, didt_default = 0, mode = 'didt'):
    """
    通过温度列表找到单/双点火延迟时间；
    点火延迟时间的判定是输入的 list1 列表中差商变化率最大的点
    双点火延迟时间的判定是
    差商的极值点： (deltalist[j]>deltalist[j-1]) and (deltalist[j+1]<deltalist[j]) 且
    1. 温度小于 1000K
    2. 温度相较于初始温度大于 IDT / 1000 (为了防止将一些极小变化值的时间考虑进去)
    params:
        list1: 温度列表; time: 时间列表
        need_index: 返回的是双点火的index，不存在双点火则返回0
        didt_default: 默认为0，若不能双点火的默认返回值
    """
    deltalist = np.abs(np.array([list1[i+1]-list1[i] for i in range(len(list1)-1)]))
    deltatime = np.array([time[i+1]-time[i] for i in range(len(list1)-1)])
    deltalist = deltalist/deltatime
    index1 = np.argmax(deltalist)
    if mode == 'idt':
        if need_index:
            return index1
        return time[index1]
    if mode == 'didt':
        args = [list1[index1]/1000, 10] # 第二个参数是第一次点火相对于初值温度的抬升,小于arg2不认为是双点火
        index2 = 0 #初始化Index2
        for j in range(1, index1):
            if (deltalist[j]>deltalist[j-1]) and (deltalist[j+1]<deltalist[j]) and \
                (list1[j] <= 1000) and (list1[j]-list1[0]>args[1]):
                index2 = j
        if need_index:
            return index2
        if index2 == 0:
            return didt_default
        return time[index2]


def TimeTemperature2ThreeStagePoints(timelist, temperaturelist, threshold = 0.01, need_index = False):
    """
    根据时间温度曲线返回开始升温、热释放率最大、升温结束三个点对应的时间和索引值
    params:
        timelist: 时间列表
        temperaturelist: 温度列表

    return:
        start_point: 开始升温点的时间和索引值
        max_point: 热释放率最大点的时间和索引值
        end_point: 升温结束点的时间和索引值
    """
    deltalist = np.abs(np.array([temperaturelist[i+1]-temperaturelist[i] for i in range(len(temperaturelist)-1)]))
    # deltatime = np.array([timelist[i+1]-timelist[i] for i in range(len(temperaturelist)-1)])
    threshold = np.max(deltalist) * threshold + np.min(deltalist)
    # start point 为 deltalist > threshold 的第一个点对应的索引值
    start_point = np.where(deltalist > threshold)[0][0] if isinstance(np.where(deltalist > threshold)[0], Iterable) else np.where(deltalist > threshold)[0]
    # max point 为 deltalist[start_point:] 的最大值对应的索引值
    max_point = np.where(deltalist[start_point:] == np.max(deltalist[start_point:]))[0][0] + start_point
    # end point 为 deltalist[max_point:] < threshold 的第一个点对应的索引值
    end_point = np.where(deltalist[max_point:] > threshold)[0][0] + max_point if isinstance(np.where(deltalist[max_point:] < threshold)[0], Iterable) else np.where(deltalist[max_point:] < threshold)[0] + max_point
    if need_index:
        return timelist[start_point], timelist[max_point], timelist[end_point], start_point, max_point, end_point
    return timelist[start_point], timelist[max_point], timelist[end_point]

'''============================================================================================================='''
'''                                          ignition delay time                                                '''
'''============================================================================================================='''


def solve_idt(gas, mode = 0, cut_time = 1, idt_defined_species = 'OH',save_traj_dir=None, P_mean=None, **kwargs):
    """
    Solve IDT 的集大成者，其中包含了所有已知的 IDT 定义。在调用时请注明使用哪一种定义\\
    可以使用的定义如下：\\
    mode:
        0：使用最最原始的定义，即温度升高 400 K 所用的时间\\
        1：最常用的定义，OH 释放达到最大变化率处切线与零浓度水平的水平线相交处对应时间\\
        2：OH释放达到最大值的50%所用的时间\\
        3：dP/dt 最大值对应的时间\\
        4：OH释放率达到最大值所用的时间\\
        5：OH释放达到最大变化率处所用的时间\\
        6: dP/dt 最大值处切线与零压力水平的水平线相交处对应时间\\
        7: CH/dt 最大值处切线与零压力水平的水平线相交处对应时间\\
    params:
        gas: 输入的是 ct.Solution, 且已经设置过当量比\\
        cut_time: 最大演化时间,超时仍未点火则返回该时间\\
        idt_defined_species: 对于某些 IDT 的定义需要一个合适的物质来指示。一般为 OH\\
        need_max_HRR: 是否需要返回最大热释放率\\
        kwargs: 对于特定的函数不同的输入。\\
    20240201: 增加 'CH_slope_max_intersect' 单列
    """
    match mode:
        case 0:
            return solve_idt_fast(gas, cut_time, **kwargs)
        case 1:
            return idt_definition_OH_slope_max_intersect(gas, s = idt_defined_species, cut_time = cut_time,save_traj_dir=save_traj_dir, P_mean=P_mean, **kwargs)
        case 'CH_slope_max_intersect':
            return idt_definition_OH_slope_max_intersect(gas, s = 'CH', cut_time = cut_time, **kwargs)
        case 2:
            return idt_definition_OH_half_max(gas, s = idt_defined_species, cut_time = cut_time, **kwargs)
        case 3:
            return idt_definition_pressure_slope_max(gas, cut_time = cut_time, **kwargs)
        case 4:
            return idt_definition_OH_max(gas, s = idt_defined_species, cut_time = cut_time, **kwargs)
        case 5:
            return idt_definition_OH_slope_max(gas, s = idt_defined_species, cut_time = cut_time, **kwargs)
        case 6:
            return idt_definition_pressure_slope_max_intersect(gas, cut_time = cut_time, **kwargs)
        case 7:
            return idt_definition_OH_slope_max_intersect(gas, cut_time = cut_time, idt_defined_species = 'CH', **kwargs)
        case "ThreeStage":
            start_time, max_time, end_time, Tem = idt_definition_ThreeStage(gas, cut_time = cut_time, **kwargs)
            return [start_time, max_time, end_time], Tem
        case "DIDT":
            # didt, idt, T = solve_idt_fast(gas, cut_time, get_double_idt = True, **kwargs)
            didt, idt, T = didt_definition_Temperature_Threshold(gas, cut_time, need_maxHRR = True, **kwargs)
            return [didt, idt], T
        case _:
            raise KeyError(f"NOT found the mode {mode} in utils!")


def solve_idt_hrr(gas, IDT_mode = 0, cut_time = 1, idt_defined_species = 'OH', **kwargs):
    """
    Solve IDT 的集大成者，其中包含了所有已知的 IDT 定义。在调用时请注明使用哪一种定义\\
    与此同时计算最大热释放率\\
    可以使用的定义如下：\\
    mode:
        0：使用最最原始的定义，即温度升高 400 K 所用的时间\\
        1：最常用的定义，OH 释放达到最大变化率处切线与零浓度水平的水平线相交处对应时间\\
        2：OH释放达到最大值的50%所用的时间\\
        3：dP/dt 最大值对应的时间\\
        4：OH释放率达到最大值所用的时间\\
        5：OH释放达到最大变化率处所用的时间\\
        6: dP/dt 最大值处切线与零压力水平的水平线相交处对应时间\\
        7: CH/dt 最大值处切线与零压力水平的水平线相交处对应时间\\
    params:
        gas: 输入的是 ct.Solution, 且已经设置过当量比\\
        cut_time: 最大演化时间,超时仍未点火则返回该时间\\
        idt_defined_species: 对于某些 IDT 的定义需要一个合适的物质来指示。一般为 OH\\
        need_max_HRR: 是否需要返回最大热释放率\\
        kwargs: 对于特定的函数不同的输入。\\
    """
    match IDT_mode:
        case 0:
            return solve_hrr(gas, cut_time, **kwargs)
        case 1:
            return idt_definition_OH_slope_max_intersect(gas, s = idt_defined_species, cut_time = cut_time, need_maxHRR = True, **kwargs)
        case 2:
            return idt_definition_OH_half_max(gas, s = idt_defined_species, cut_time = cut_time, need_maxHRR = True, **kwargs)
        case 3:
            return idt_definition_pressure_slope_max(gas, cut_time = cut_time, need_maxHRR = True, **kwargs)
        case 4:
            return idt_definition_OH_max(gas, s = idt_defined_species, cut_time = cut_time, need_maxHRR = True, **kwargs)
        case 5:
            return idt_definition_OH_slope_max(gas, s = idt_defined_species, cut_time = cut_time, need_maxHRR = True, **kwargs)
        case 6:
            return idt_definition_pressure_slope_max_intersect(gas, cut_time = cut_time, need_maxHRR = True,**kwargs)
        case 7:
            return idt_definition_OH_slope_max_intersect(gas, cut_time = cut_time, idt_defined_species = 'CH', **kwargs)
        case "ThreeStage":
            start_time, max_time, end_time, Tem = idt_definition_ThreeStage(gas, cut_time = cut_time, need_maxHRR = True, **kwargs)
            return [start_time, max_time, end_time], Tem
        case "DIDT":
            # didt, idt, T = solve_idt_fast(gas, cut_time, get_double_idt = True, need_maxHRR = True, **kwargs)
            didt, idt, T = didt_definition_Temperature_Threshold(gas, cut_time, need_maxHRR = True, **kwargs)
            return [didt, idt], T
        case _:
            raise KeyError(f"NOT found the mode {IDT_mode} in utils!")


def solve_idt_fast(gas: ct.Solution, cut_time = 1, idt_defined_T_diff:float = 400, time_multiple = 1.2, time_step_min = 1e-7, 
                   need_judge = False, get_curve = False, get_double_idt = False, didt_default = 'IDT', 
                   need_index = False, cut_time_multipler = 10, **kwargs):
    """
    返回点火延迟和最终火焰温度数据,用sim.step()来进行加速
    params:
        gas:            输入的是 ct.Solution, 且已经设置过当量比
        cut_time:       最大演化时间,超时仍未点火则返回该时间
        time_multiple:  终止时刻为多少倍的点火延迟时间
        time_step_min:  自适应演化步长的最小值,小于该值则手动演化到下一时刻
        need_judge:     是否返回演化有没有超时
        get_curve:      是否返回整个点火曲线的时间温度列表,而不是idt和T
        get_double_idt: 是否计算双点火,返回两个点火延迟时间
        didt_default:   若不存在低温点火,则低温点火设置的最小值
        raise_error:    默认为 False, 当 raise error 为 True 时, 会在达到 cut_time 后返回错误信息而非 idt
        need_index:     默认为 False, 应用于 get_optim_point 函数
    """
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim_time, idt, diff_T = 0, 0, 0; ini_temperature = r.T
    achieve_max = False
    if get_curve or get_double_idt:
        sim_time_list = []
        T_list = []
    sim.max_time_step = 0.1
    
    # 点火延迟时间定义为温度与初始温度的差大于400K
    while diff_T < idt_defined_T_diff:  
        old_sim_time = sim_time
        sim_time = sim.step()
        # 若演化步长太小,则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        if get_curve or get_double_idt:
            sim_time_list.append(sim_time)
            T_list.append(r.T)
        diff_T = r.T - ini_temperature

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            if idt < 1e-8:
                idt = cut_time_multipler * cut_time
                achieve_max = True
            break
    idt = sim_time

    # 再往后演化一段时间,不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        sim_time = sim.step()
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        if get_curve or get_double_idt:
            sim_time_list.append(sim_time)
            T_list.append(r.T)
    
    # 模式1：获取点火曲线
    if get_curve:
        return sim_time_list, T_list, idt, r.T
    # 模式2：获取双点火
    elif get_double_idt:
        if didt_default == 'IDT': didt_default = idt
        didt = get_optim_point(T_list, sim_time_list, need_index = need_index, didt_default = didt_default)
        return didt, idt, r.T
    # 模式3：返回是否达到最大模拟时间
    elif need_judge:
        return idt, r.T, achieve_max
    else:
        return idt, r.T
 

def idt_definition_ThreeStage(gas, cut_time = 1, threshold = 0.02, **kwargs):
    """
    solve_idt_fast 高级版，求解出时间温度曲线的三个特征点位置的时间值。这三个特征点为：温度升高最大值，当前最大斜率下
    2 % 作为临界值 曲线斜率首次达到临界值和尾次达到临界值对应的时间点，求解方式为使用 solve_idt_fast 中的 get_curve 方法
    求出后调用 TimeTemperature2ThreeStagePoints 函数求解出三个特征点的位置
    params:
        gas: 燃料气体
        cut_time: 演化时长过后停止时间
    """
    timelist, tlist, idt, Tem = solve_idt_fast(gas, cut_time = cut_time, get_curve = True, **kwargs)
    start_time, max_time, end_time = TimeTemperature2ThreeStagePoints(timelist, tlist, threshold)
    return start_time, max_time, end_time, Tem


def solve_idt_species(gas, species_list, cut_time = 1, time_multiple = 1.2, time_step_min = 1e-7, cut_time_multipler = 10):
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    
    sim_time_list, T_list, concentration_species = [], [], []

    for _ in range(len(species_list)):
        concentration_species.append([])

    sim.max_time_step = 0.1
    
    # 点火延迟时间定义为温度与初始温度的差大于400K
    while diff_T < 400:  
        old_sim_time = sim_time
        sim_time = sim.step()
        # 若演化步长太小,则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        
        sim_time_list.append(sim_time)
        T_list.append(r.T)
        for t,s in enumerate(species_list):
            concentration_species[t].append(r.thermo[s].X[0])

        diff_T = r.T - ini_temperature

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            if idt < 1e-8:
                idt = cut_time * cut_time_multipler
            break
    idt = sim_time

    # 再往后演化一段时间,不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        sim_time = sim.step()
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        sim_time_list.append(sim_time)
        T_list.append(r.T)
        for t,s in enumerate(species_list):
            concentration_species[t].append(r.thermo[s].X[0])
    
    return sim_time_list, T_list, concentration_species


def yaml2idt(chem_file:str, mode:int | list | str = 0, cut_time:float | list = 1, setup_file:str = None, 
             IDT_condition = None, IDT_T = None, IDT_P = None, IDT_phi = None, 
             fuel: str| list = None, oxidizer: str| list = None, save_path:str = None, idt_defined_species = None,
             return_condition = False,  **kwargs):
    """
    params:
        chem_file
        mode: solve_idt 函数的 模式
        cut_time: 截止时间; 默认为 1, 可设置为真实点火的倍数
        setup_file: 保存基础设置的 setup.yaml; 请见模板
        IDT_condition: 如果 IDT_condition 是一维数组，在此情况下 IDT_condition 一定是单条数据。这样将返回一个单值的 IDT 和 T 而非列表
        IDT_T; IDT_P; IDT_phi
        fuel; oxidizer
        save_path: 默认为 None
        kwargs: 其他的 solve_idt_fast 的参数
    如果想要恢复 三维张量,请使用
    np.reshape(len(phi), len(T), len(P))
    """
    if isinstance(mode, str) and mode == 'DIDT':
        return yaml2didt(chem_file, setup_file = setup_file, DIDT_condition = IDT_condition, IDT_T = IDT_T, IDT_P = IDT_P, IDT_phi = IDT_phi,
                         fuel = fuel, oxidizer = oxidizer, save_path = save_path, **kwargs)
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        IDT_T, IDT_P, IDT_phi = chem_args['IDT_T'], chem_args['IDT_P'], chem_args['IDT_phi']
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif IDT_condition is None:
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )

    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    # 单值下快速运算
    if IDT_condition.ndim == 1:
        assert not isinstance(cut_time, Iterable), '单值模式下 cut_time 只能为 float 等单值'
        gas = ct.Solution(chem_file)
        phi, T, P = IDT_condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        idt, Tem = solve_idt(gas, mode = mode, cut_time = cut_time, idt_defined_species=idt_defined_species,**kwargs)
        if not save_path is None:
            np.savez(save_path, IDT = idt, T = Tem)
        if return_condition:
            return idt, Tem, IDT_condition
        else:
            return idt, Tem
    else:
        fuel = [fuel] * len(IDT_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(IDT_condition) if not isinstance(oxidizer, list) else oxidizer
        mode = [mode] * len(IDT_condition) if not isinstance(mode, Iterable) else mode
        IDT, Temperature = [], []; cut_time = cut_time * np.ones(len(IDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
        for tmp_cut_time, condition, tmp_fuel, tmp_oxidizer, tmp_mode in zip(cut_time, IDT_condition, fuel, oxidizer, mode):
            gas = ct.Solution(chem_file)
            phi, T, P = condition
            gas.TP = T, P * ct.one_atm
            gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
            idt, Tem = solve_idt(gas, mode = tmp_mode, cut_time = tmp_cut_time, idt_defined_species=idt_defined_species, **kwargs)
            IDT.append(idt); Temperature.append(Tem)
        IDT, Temperature = np.array(IDT), np.array(Temperature)
        if not save_path is None:
            np.savez(save_path, IDT = IDT, T = Temperature)
        return IDT, Temperature


def yaml2didt(chem_file:str, cut_time = 1, DIDT_condition:list[list] = None, IDT_T = None, IDT_P = None, IDT_phi = None,
              fuel = None, oxidizer = None, setup_file = None, save_path = None, didt_default = 'IDT', **kwargs):
    """
    params:
        chem_file
        didt_condition: 存在双点火的工况
        fuel; oxidizer
        kwargs:
            save_path: 默认为 None
            其他的 solve_idt_fast 的参数
        return:
            DIDT, IDT, T
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        IDT_T, IDT_P, IDT_phi = chem_args['IDT_T'], chem_args['IDT_P'], chem_args['IDT_phi']
        DIDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif DIDT_condition is None:
        DIDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )

    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    fuel = [fuel] * len(DIDT_condition) if not isinstance(fuel, list) else fuel
    oxidizer = [oxidizer] * len(DIDT_condition) if not isinstance(oxidizer, list) else oxidizer
    Temperature = []; cut_time = cut_time * np.ones(len(DIDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
    DIDT = []
    for tmp_cut_time, condition, tmp_fuel, tmp_oxidizer in zip(cut_time, DIDT_condition, fuel, oxidizer):
        gas = ct.Solution(chem_file)
        phi, T, P = condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
        didt, Tem = solve_idt(gas, mode = 'DIDT', cut_time = tmp_cut_time, didt_default = didt_default, **kwargs)
        DIDT.extend(didt); Temperature.append(Tem)
    Temperature, DIDT = np.array(Temperature), np.array(DIDT)
    if not save_path is None:
        np.savez(save_path, T = Temperature, DIDT = DIDT)
    return DIDT, Temperature


def yaml2idtcurve(chem_file:str, IDT_condition = None, IDT_T = None, IDT_P = None, IDT_phi = None,
                   setup_file:str = None, fuel = None, oxidizer = None, save_path = None, cut_time = 1, 
                   idt_defined_T_diff = 400, yaml2idtcurve_needIDT = False, **kwargs):
    """
    基本同 yaml2idt, 但是是为了计算时间温度变化曲线
    params:
        chem_file
        setup_file: 保存基础设置的 setup.yaml; 请见模板
        IDT_condition: 二维数组, 每一行为 [phi, T, P]
        IDT_T; IDT_P; IDT_phi
        fuel; oxidizer
        save_path: 默认为 None
    如果想要恢复 三维张量,请使用
    np.reshape(len(phi), len(T), len(P))
    return:
        timelist, tlist
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        IDT_T, IDT_P, IDT_phi = chem_args['IDT_T'], chem_args['IDT_P'], chem_args['IDT_phi']
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif IDT_condition is None:
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )

    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    gas = ct.Solution(chem_file)
    if IDT_condition.ndim == 1:
        phi, T, P = IDT_condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        timelist, tlist, idt, T = solve_idt_fast(gas, get_curve = True, cut_time = cut_time, idt_defined_T_diff = idt_defined_T_diff, **kwargs)
        if not save_path is None:
            np.savez(save_path, timelist = timelist, tlist = tlist)
        if yaml2idtcurve_needIDT:
            return timelist, tlist, idt, T
        else:
            return timelist, tlist
    else:
        timelist, tlist = [], []
        IDT, Temperature = [], []
        fuel = [fuel] * len(IDT_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(IDT_condition) if not isinstance(oxidizer, list) else oxidizer
        cut_time = cut_time * np.ones(len(IDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
        for condition, tmp_fuel, tmp_oxidizer, tmp_cut_time in zip(IDT_condition, fuel, oxidizer, cut_time):
            phi, T, P = condition
            gas.TP = T, P * ct.one_atm
            gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
            tmp_timelist, tmp_tlist, idt, tem = solve_idt_fast(gas, get_curve = True, cut_time = tmp_cut_time, idt_defined_T_diff = idt_defined_T_diff, **kwargs)
            timelist.append(tmp_timelist); tlist.append(tmp_tlist)
            IDT.append(idt); Temperature.append(tem)
        # timelist, tlist = np.array(timelist), np.array(tlist)
        # if not save_path is None:
        #     np.savez(save_path, timelist = timelist, tlist = tlist)
        if yaml2idtcurve_needIDT:
            return timelist, tlist, IDT, Temperature
        else:
            return timelist, tlist


def yaml2idt_sensitivity(chem_file:str, setup_file:str = None, IDT_condition = None, IDT_T = None, IDT_P = None, IDT_phi = None, 
                        fuel: str| list = None, oxidizer: str| list = None,  
                        delta:float = 1e-3, cut_time = 100, specific_reactions:list = None,
                        mode: int|str|list = 0, save_path = None, need_baseidt = False, idt_defined_species = None, **kwargs):
    r"""
    使用 Cantera 内置的方法求解 idt 关于所有反应的局部敏感度
    params:
        chem_file：化学机构文件
        setup_file: 保存基础设置的 setup.yaml; 请见模板
        IDT_condition: 二维数组, 每一行为 [phi, T, P]
        IDT_T; IDT_P; IDT_phi: 一维数组
        fuel; oxidizer: 一维数组
        delta: 用于计算局部敏感度的 delta
        cut_time: 用于计算 idt 的时间
        specific_reaction: 指定需要计算的反应, 默认为 None, 即所有反应
        mode: solve_idt 函数的模式
        save_path: 保存路径, 默认为 None
        need_baseidt: 是否需要计算基础 idt, 默认为 False
        kwargs:
            其他的 solve_idt_fast 的参数
    return:
        sensitivity from:
        $$
        k_i = \frac{IDT((1+\delta)A) - IDT(A)}{\delta}
        $$
    """
    print(f'Start calculating the sensitivity of IDT; Totally len of the probe points are {len(IDT_condition)}')
    time0 = time.time()
    np.set_printoptions(precision = 2, suppress = True)
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        IDT_T, IDT_P, IDT_phi = chem_args['IDT_T'], chem_args['IDT_P'], chem_args['IDT_phi']
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif IDT_condition is None:
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )

    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    gas = ct.Solution(chem_file)
    if specific_reactions is None: specific_reactions = gas.reaction_equations()
    # 计算 baseidt
    baseidt, _ = yaml2idt(
        chem_file, mode = mode, cut_time = cut_time, IDT_condition = IDT_condition,
        fuel = fuel, oxidizer = oxidizer, idt_defined_species = idt_defined_species, **kwargs
    )
    time1 = time.time()
    print(f'Base IDT calculation time: {time1 - time0:.2f}s')
    IDT_sensitivity = {}
    # 单值下快速运算
    if IDT_condition.ndim == 1:
        for m in range(gas.n_reactions):
            
            equation = gas.reaction(m).equation
            if equation not in specific_reactions: 
                continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            idt, _ = solve_idt(gas, mode = mode, idt_defined_species=idt_defined_species, **kwargs)
            if equation in IDT_sensitivity:
                # 求两者的最大值作为敏感度
                IDT_sensitivity[equation] = max(IDT_sensitivity[equation], (np.log(idt) - np.log(baseidt)) / (np.log(1+delta)))
            else:
                IDT_sensitivity.update(
                    {equation: (np.log(idt) - np.log(baseidt)) / (np.log(1+delta))}
                )
            time2 = time.time()
            print(f'{equation} has been calculated, time cost: {time2 - time1:.2f}s')
    else:
        fuel = [fuel] * len(IDT_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(IDT_condition) if not isinstance(oxidizer, list) else oxidizer
        cut_time = cut_time * np.ones(len(IDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
        mode = [mode] * len(IDT_condition) if not isinstance(mode, Iterable) else mode
        gas = ct.Solution(chem_file)
        for m in range(gas.n_reactions):
            equation = gas.reaction(m).equation
            if equation not in specific_reactions:
                continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            IDT = []
            for ind, tmp_cut_time, condition, tmp_fuel, tmp_oxidizer, tmp_mode in zip(range(len(IDT_condition)), cut_time, IDT_condition, fuel, oxidizer, mode):
                phi, T, P = condition
                gas.TP = T, P * ct.one_atm
                gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
                try:
                    idt, _ = solve_idt(gas, mode = tmp_mode, cut_time = tmp_cut_time, idt_defined_species=idt_defined_species, **kwargs)
                except:
                    print(traceback.format_exc()  )
                    print(f'{equation} failed at {condition}')
                    print(f'fuel: {tmp_fuel}, oxidizer: {tmp_oxidizer}')
                    print(f'phi: {phi}, T: {T}, P: {P}')
                    idt = baseidt[ind]
                IDT.append(idt)
            IDT = np.array(IDT)
            if equation in IDT_sensitivity:
                # 求两者的最大值作为敏感度
                IDT_sensitivity[equation] = np.maximum(IDT_sensitivity[equation], (np.log(IDT) - np.log(baseidt)) / np.log(1+delta)).tolist()
            else:
                IDT_sensitivity.update(
                    {equation: ((np.log(IDT) - np.log(baseidt)) / (np.log(1+delta))).tolist()}
                )
            time2 = time.time()
            print(f'{equation} has been calculated, time cost: {time2 - time1:.2f}s')
    if not save_path is None:
        # 使用 jSON 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(IDT_sensitivity, ensure_ascii=False, indent=4, separators=(',', ':')))

    if need_baseidt: return IDT_sensitivity, baseidt
    return IDT_sensitivity

'''============================================================================================================='''
'''                                                   psr                                                       '''
'''============================================================================================================='''

def yaml2psr(chem_file, PSR_condition: np.ndarray = None, RES_TIME_LIST = None, PSR_T = None, PSR_P = None, PSR_phi = None,
             setup_file = None, error_tol:float = 50, save_path = None, fuel = None, oxidizer = None, **kwargs):
    """
    params:
        chem_file: 化学机构文件
        PSR_condition: 等同于 PST_T PSR_P PSR_phi
        RES_TIME_LIST
        setup_file: 保存基础设置的 setup.yaml; 请见模板
        kwargs:
            fuel; oxidizer
            save: 默认为 false, 若为 true 会自动保存到 chem_file 的位置
    """
    assert RES_TIME_LIST is not None, '请设置 RES_TIME_LIST'
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
    # 单值下快速运算
    if PSR_condition.ndim == 1:
        gas = ct.Solution(chem_file)
        assert np.array(RES_TIME_LIST).ndim == 1, '请设置 RES_TIME_LIST 为一维数组'
        phi, T, P = PSR_condition
        # gas.TP = T, P * ct.one_atm
        # gas.set_equivalence_ratio(phi, fuel, oxidizer)
        psr_T = solve_psr(gas, RES_TIME_LIST, T, P, phi, fuel, oxidizer, error_tol = error_tol, **kwargs)
        psr_T = np.array(psr_T)
        if not save_path is None:
            np.savez(save_path, PSR = psr_T)
        return psr_T
    else:
        fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
        PSR = []; 
        for tmp_res_time, condition, tmp_fuel, tmp_oxidizer in zip(RES_TIME_LIST, PSR_condition, fuel, oxidizer):
            gas = ct.Solution(chem_file)
            phi, T, P = condition
            # gas.TP = T, P * ct.one_atm
            # gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
            psr_T = solve_psr(gas, tmp_res_time, T, P, phi,  tmp_fuel, tmp_oxidizer, error_tol = error_tol, **kwargs)
            PSR.extend(psr_T)
        PSR = np.array(PSR)
        if not save_path is None:
            np.savez(save_path, PSR = PSR)
        return PSR


def yaml2PSR_concentration(chem_file, concentration_species = None, PSR_condition: np.ndarray = None, concentration_res_time = None, PSR_T = None, PSR_P = None, PSR_phi = None, setup_file = None, error_tol:float = 50, fuel = None, oxidizer = None, diluent = {}, return_list = True, return_condition = False, **kwargs):
    """
    params:
        chem_file: 化学机构文件
        PSR_condition: 等同于 PST_T PSR_P PSR_phi
        RES_TIME_LIST
        setup_file: 保存基础设置的 setup.yaml; 请见模板
        kwargs:
            fuel; oxidizer
            save: 默认为 false, 若为 true 会自动保存到 chem_file 的位置
    """
    
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        PSR_T, PSR_P, PSR_phi = chem_args['PSR_concentration_T'], chem_args['PSR_concentration_P'], chem_args['PSR_concentration_phi']
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
        fuel, oxidizer = chem_args.get('PSR_concentration_fuel', fuel), chem_args.get('PSR_concentration_oxidizer', oxidizer)
        concentration_res_time = chem_args.get('PSR_concentration_res_time', concentration_res_time)
        if 'PSR_concentration_species' in chem_args:
            concentration_species = chem_args['PSR_concentration_species']
        if 'PSR_concentration_diluent' in chem_args:
            diluent = chem_args['PSR_concentration_diluent']


    if PSR_condition is None:
        PSR_condition = np.array(
                    [[phi, T, P] for phi in PSR_phi for T in PSR_T for P in PSR_P]
                )
    assert isinstance(diluent, dict), f"PSR_concentration_diluent should be a dict, e.g. 'N2': 0.21, 'Ar': 0.79, but got {diluent}"
    if diluent and 'diluent' not in diluent and 'fraction' not in diluent:
        diluent = {
            'diluent': list(diluent.keys())[0],
            'fraction': {
                'diluent': list(diluent.values())[0]
            }
        } 
    
    assert fuel is not None and oxidizer is not None, f'请设置 fuel 和 oxidizer, now get fuel: {fuel}, oxidizer: {oxidizer}'
    assert concentration_res_time is not None, '请设置 concentration_res_time'
    # 单值下快速运算
    if PSR_condition.ndim == 1:
        gas = ct.Solution(chem_file)
        assert np.array(concentration_res_time).ndim == 1, '请设置 concentration_res_time 为一维数组'
        phi, T, P = PSR_condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer, **diluent)
        concentration_dict = solve_psr_concentration(gas, concentration_species, residence_time = concentration_res_time, **kwargs)
        return concentration_dict
    else:
        fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
        concentration_res_time = concentration_res_time * np.ones(len(PSR_condition)) if not isinstance(concentration_res_time, Iterable) else concentration_res_time
        concentration_dicts = {
                species: [] for species in concentration_species
            }
        
        for condition, tmp_fuel, tmp_oxidizer, res_time in zip(PSR_condition, fuel, oxidizer, concentration_res_time):
            gas = ct.Solution(chem_file)
            phi, T, P = condition
            gas.TP = T, P * ct.one_atm
            gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer, **diluent)
            
            concentration_dict = solve_psr_concentration(gas, concentration_species, residence_time = res_time, **kwargs)
            for species in concentration_species:
                concentration_dicts[species].append(concentration_dict[species])
        if return_list:
            if return_condition:
                return concentration_dicts, PSR_condition
            return np.concatenate([concentration_dicts[species] for species in concentration_species], axis = 0)
        
        return concentration_dicts


def solve_psr(gas, residence_time_list, T = None, P = None, phi = None, fuel = None, oxidizer = None, error_tol = 0., **kwargs) -> list:
    """
    params:
        error_tol: default 0; IF not 0, error will raise if the |PSR_T - Initial temperature|< error_tol.
    """
    Reduced_T_list = []
    if T is not None and P is not None and phi is not None:
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
    else:
        T = gas.T
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

    # sim.set_initial_time(0)
    # sim.advance_to_steady_state()
    for residence_time in residence_time_list:
        sim.set_initial_time(0)
        try:
            sim.advance_to_steady_state()
            Reduced_T_list.append(combustor.T)
        except:
            print(f"solve_psr residence_time: {residence_time} can't reach steady state! Tem set to be initial temperature: {T}")
            Reduced_T_list.append(T)
        if error_tol and np.any(np.abs(np.array(Reduced_T_list) - T) < error_tol):
            raise ValueError(f"PSR cantera simulation error: Meet the PSR_T diff error tolerance! Reduced T list is {Reduced_T_list} but {T=}")

    return Reduced_T_list


def solve_psr_concentration(gas, concentration_on_species, error_tol = 0., residence_time = 1, **kwargs) -> list:

    assert isinstance(concentration_on_species, list), 'concentration_on_species must be a list of species names'
    for species in concentration_on_species:
        if species not in gas.species_names:
            raise ValueError(f"Species {species} not found in the gas mixture.")
        
    concentration_dict = {species: [] for species in concentration_on_species}
    print(f'The pressure of the gas is {gas.P}, ')
    # inlet -> inlet_mfc -> combustor -> outlet_mfc -> exhaust
    inlet = ct.Reservoir(gas)               # 进气口预混箱
    combustor = ct.IdealGasReactor(gas, volume = 1.0)
    exhaust = ct.Reservoir(gas)
    def mdot(t):
        return combustor.mass/residence_time
    inlet_mfc = ct.MassFlowController(upstream = inlet, downstream = combustor, mdot = mdot)
    outlet_mfc = ct.PressureController(upstream = combustor, downstream = exhaust, master=inlet_mfc, K=0.01)
    sim = ct.ReactorNet([combustor])
    sim.set_initial_time(0)
    sim.advance_to_steady_state()
    for species in concentration_on_species:
        concentration_dict[species] = combustor.thermo.X[combustor.thermo.species_index(species)]
    print(f'The pressure of the reactor is {combustor.thermo.P}, and the temperature is {combustor.T}')
    return concentration_dict


def solve_psr_true(gas, ini_res_time = 1, exp_factor = 2 ** (1/2), species = None, **kwargs):
    """
    输入设定好的gas，返回机理生成的residence_time列表和对应的温度
    params:
        ini_res_time: 初始反应时间; 刚开始设定反应时间足够长，让火焰烧起来; 默认为 1
    """
    if exp_factor == 1: 
        raise ValueError("exp_factor can't be 1")   
    residence_time = ini_res_time
    Res_Time_L, True_T_list = [], []
    ini_T = gas.T
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

    # while combustor.T > ini_T + 100 and residence_time > 1e-3:
    sim.set_initial_time(0)
    sim.advance_to_steady_state()
    True_T_list.append(combustor.T)
    Res_Time_L.append(residence_time) # 第一次加入的是1499K
    if species is not None: 
        species_profile = {
            sp: [combustor.thermo.X[combustor.thermo.species_index(sp)]] for sp in species
        }
    while combustor.T > ini_T + 10:
        residence_time /= exp_factor
        sim.set_initial_time(0)
        try:
            sim.advance_to_steady_state()
            True_T_list.append(combustor.T)
            Res_Time_L.append(residence_time) # 第一次加入的是1499K
            if species is not None:
                for sp in species:
                    species_profile[sp].append(combustor.thermo.X[combustor.thermo.species_index(sp)])
        except:
            residence_time *= exp_factor
            break
        print(f"here at solve_psr_true residence_time: {residence_time}, expfactor: {exp_factor}, T: {combustor.T}")
    if species is not None:
        return Res_Time_L, True_T_list, species_profile
    return Res_Time_L, True_T_list


def yaml2psr_sensitivity(chem_file:str, RES_TIME_LIST:list, PSR_condition: np.ndarray = None, 
                         PSR_T = None, PSR_P = None, PSR_phi = None,
                         setup_file = None, fuel:str = None, oxidizer:str = None,
                         delta:float = 1e-3, specific_reactions: list = None,
                         save_path = None, result_abs = False, need_basepsr = False, **kwargs):
    """
    使用 Cantera 内置的方法求解 psr 关于所有反应的局部敏感度
    最终返回值为 (psr - base_psr) / (base_psr * delta)
    params:
        chem_file: 机理文件路径
        RES_TIME_LIST: PSR RES_TIME_LIST
        PSR_condition: PSR 条件列表, shape = (n, 3), n 为 PSR_condition 的个数, 3 为 phi, T, P
        PSR_T: PSR 温度列表
        PSR_P: PSR 压力列表
        PSR_phi: PSR 当量比列表
        setup_file: PSR 条件设置文件路径
        fuel: 燃料名称
        oxidizer: 氧化剂名称
        delta: 敏感度分析中， A 值调整的范围
        specific_reactions: 指定需要计算敏感度的反应列表
        save_path: 保存路径
        result_abs: 是否保存敏感度的绝对值
        need_basepsr: 是否需要计算基准 PSR
    return: 
        psr_sensitivity: dict, key 为反应方程式, value 为敏感度列表; 计算方式与 solve_psr_sensitivity 相同
    """
    np.set_printoptions(precision=2, suppress=True)
    assert RES_TIME_LIST is not None, '请设置 RES_TIME_LIST'
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
    PSR_sensitivity = {}
    # 计算 base_psr
    basepsr = yaml2psr(
        chem_file, PSR_condition, RES_TIME_LIST, 
        fuel = fuel, oxidizer = oxidizer,
        error_tol = 0
    )
    # 单值下快速运算
    if PSR_condition.ndim == 1:
        phi, T, P = PSR_condition
        assert np.array(RES_TIME_LIST).ndim == 1, '请设置 RES_TIME_LIST 为一维数组'
        for m in range(gas.n_reactions):
            equation = gas.reaction(m).equation
            if equation not in specific_reactions: continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            psr = solve_psr(gas, RES_TIME_LIST, T, P, phi, fuel, oxidizer, error_tol = 0.); psr = np.array(psr)
            PSR_sensitivity.update(
                {equation: ((psr - basepsr) / (delta)).tolist()}
            )

    else:
        fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
        for m in range(gas.n_reactions):
            equation = gas.reaction(m).equation
            if equation not in specific_reactions: continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            tmp_PSR = []
            for tmp_res_time, condition, tmp_fuel, tmp_oxidizer in zip(RES_TIME_LIST, PSR_condition, fuel, oxidizer):
                phi, T, P = condition
                gas.TP = T, P * ct.one_atm
                gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
                tmp_psr = solve_psr(gas, tmp_res_time, error_tol = 0.)  
                tmp_PSR.extend(tmp_psr)
            tmp_PSR = np.array(tmp_PSR)
            if equation in PSR_sensitivity:
                # 求两者的最大值作为敏感度
                PSR_sensitivity[equation] = np.maximum(PSR_sensitivity[equation], (tmp_PSR - basepsr) / (delta)).tolist()
            else:
                PSR_sensitivity.update(
                    {equation: ((tmp_PSR - basepsr) / (delta)).tolist()}
                )     
    if result_abs:
        for k, v in PSR_sensitivity.items():
            PSR_sensitivity[k] = np.abs(v).tolist()
    if not save_path is None:
        # 使用 jSON 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(PSR_sensitivity, ensure_ascii=False, indent=4, separators=(',', ':')))

    if need_basepsr: return PSR_sensitivity, basepsr
    return PSR_sensitivity

'''============================================================================================================='''
'''                                               flame speed                                                   '''
'''============================================================================================================='''

def solve_flame_speed(gas, width = 0.03, loglevel = 0, ratio = 3,
                      slope = 0.06, curve = 0.12, transport_model = 'Mix', **kwargs) -> list:
    """
    输入设定好的gas,返回层流火焰速度（单位[m/s]）
    """
    # gas.transport_model = transport_model       
    f = ct.FreeFlame(gas, width = width)
    f.set_refine_criteria(ratio = ratio, slope = slope, curve = curve)
    f.transport_model = transport_model
    f.solve(loglevel = loglevel, auto = True)
    return f.velocity[0]


def yaml2FS(chem_file, FS_condition = None, FS_T = None, FS_P = None, FS_phi = None,
             setup_file = None, save_path = None, fuel = None, oxidizer = None, return_condition = False, **kwargs):
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        fuel, oxidizer = chem_args['fuel'], chem_args['oxidizer']
        if FS_condition is None:
            FS_T, FS_P, FS_phi = chem_args['LFS_T'], chem_args['LFS_P'], chem_args['LFS_phi']
            FS_condition = np.array(
                [[phi, T, P] for phi in FS_phi for T in FS_T for P in FS_P]
            )
    elif FS_condition is None:
        FS_condition = np.array(
            [[phi, T, P] for phi in FS_phi for T in FS_T for P in FS_P]
        )
    gas = ct.Solution(chem_file)
    # 单值快速计算
    if FS_condition.ndim == 1:
        phi, T, P = FS_condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        FS = solve_flame_speed(gas, **kwargs)
        if return_condition:
            return FS, FS_condition
        else:
            return FS
    else:
        fuel = [fuel] * len(FS_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(FS_condition) if not isinstance(oxidizer, list) else oxidizer
        FS  = []  
        for condition, tmp_fuel, tmp_oxidizer in zip(FS_condition, fuel, oxidizer):
            phi, T, P = condition
            gas.TP = T, P * ct.one_atm
            gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
            fs = solve_flame_speed(gas, **kwargs); FS.append(fs)
        FS = np.array(FS)
    if save_path is not None:
        np.savez(save_path, FS = FS)
    if return_condition:
        return FS, FS_condition
    else:
        return np.array(FS)


def yaml2LFS_sensitivity(chem_file:str, LFS_condition: np.ndarray = None, base_LFS = None, 
                                      LFS_T = None, LFS_P = None, LFS_phi = None, setup_file = None, fuel:str = None, oxidizer:str = None,
                         delta:float = 1e-3, specific_reactions: list = None, save_path = None, result_abs = False, need_baseLFS = False, **kwargs):
    """
    详细参见 yaml2psr_sensitivity 函数的说明
    """
    np.set_printoptions(precision=2, suppress=True)
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        LFS_T, LFS_P, LFS_phi = chem_args['LFS_T'], chem_args['LFS_P'], chem_args['LFS_phi']
        LFS_condition = np.array(
                    [[phi, T, P] for phi in LFS_phi for T in LFS_T for P in LFS_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif LFS_condition is None:
        LFS_condition = np.array(
                    [[phi, T, P] for phi in LFS_phi for T in LFS_T for P in LFS_P]
                )
    
    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    gas = ct.Solution(chem_file)
    if specific_reactions is None: specific_reactions = gas.reaction_equations()
    LFS_sensitivity = {};  reaction_list = []
    # 计算 base_LFS:(不推荐使用)
    if base_LFS is None:
        base_LFS = yaml2FS(
            chem_file, LFS_condition, 
            fuel = fuel, oxidizer = oxidizer,
            save_dirpath = None,
        )
    # 单值下快速运算
    if LFS_condition.ndim == 1:
        phi, T, P = LFS_condition
        for m in range(gas.n_reactions):
            gas = ct.Solution(chem_file)
            gas.TP = T, P; gas.set_equivalence_ratio(phi, fuel, oxidizer)
            equation = gas.reaction(m).equation
            if equation not in specific_reactions: continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            reaction_list.append(equation)
            LFS = solve_flame_speed(gas, **kwargs)
        LFS_sensitivity.update(
                    {equation: ((lfs - base_LFS) / (delta * base_LFS)).tolist()} for equation, lfs in zip(reaction_list, LFS)
                )

    else:
        LFS = []
        fuel = [fuel] * len(LFS_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(LFS_condition) if not isinstance(oxidizer, list) else oxidizer
        for m in range(gas.n_reactions):
            gas = ct.Solution(chem_file)
            equation = gas.reaction(m).equation
            if equation not in specific_reactions: continue
            gas.set_multiplier(1.0)  # reset all multipliers
            gas.set_multiplier(1 + delta, m)  # perturb reaction m
            reaction_list.append(equation)
            for condition, tmp_fuel, tmp_oxidizer in zip(LFS_condition, fuel, oxidizer):
                phi, T, P = condition
                gas.TP = T, P; gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
                lfs = solve_flame_speed(gas, **kwargs)
                LFS.append(lfs)
            LFS = np.array(LFS)
        for equation, tmp_LFS in zip(reaction_list, LFS):
            LFS_sensitivity.update(
                {equation: ((tmp_LFS - base_LFS) / (delta * base_LFS)).tolist()}
            )     
    if result_abs:
        for k, v in LFS_sensitivity.items():
            LFS_sensitivity[k] = np.abs(v).tolist()
    if not save_path is None:
        # 使用 jSON 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(LFS_sensitivity, ensure_ascii=False, indent=4, separators=(',', ':')))

    if need_baseLFS: return LFS_sensitivity, base_LFS
    return LFS_sensitivity


'''============================================================================================================='''
'''                                            species mole fractions                                           '''
'''============================================================================================================='''

def species2mole(gas:ct.Solution, *species:str) -> float:
    """
    计算一个反应器中 gas 相对某个 species 的摩尔浓度分数
    Args:
        gas: 对应的气体
        species: 用于查询该 species 的摩尔浓度分数, 支持多个 species
    """
    num = len(species)
    if num == 1: result = gas.X[gas.species_index(species[0])]
    else: result = [gas.X[gas.species_index(species[k])] for k in range(num)]
    return result


def solve_mole(gas: ct.Solution, species, cut_time:float = 0., min_steps:int = 0, time_step_min = 1e-7,
                max_time_step:float = 0.1, mode:str = 'default', raise_error = False, **kwargs):
    """
    返回指定 species 的 mole分数 变化数据,用sim.step()来进行加速
    另一个使用方式是，用来计算 定义方式为某种物种(函数中的species)浓度最大值下的 IDT: 一般物种选择 OH 或 CH*
    params:
        gas:            输入的是 ct.Solution, 且已经设置过当量比
        species:        需要获得mole浓度分数的 species,支持 list 输入
        cut_time:       最大演化时间, 将返回这个时间的 mole 浓度; 默认值为 0, 0表示不设置 cut_time
        min_steps:      最大演化步长, 不论如何 sim.step() 都会执行这么多次; 不应同时和 cut_time 一起设置但两者必须有一个设置; 默认为 0
        time_step_min:  自适应演化步长的最小值,小于该值则手动演化到下一时刻
        
        max_time_step:  sim.step() 最大的单步演化时间
        mode:           solve mole 的模式，有以下几种选择：
            default:    默认值，返回 mole
            get_curve:  返回整个系统演化的时间mole浓度列表
            get_curve_max:    返回曲线斜率的最大值和对应的浓度
            get_max:    返回曲线的最大值和对应的浓度/ 这个是对应 IDT 的含义
        raise_error:    默认为 False, 当 raise error 为 True 时, 会在达到 cut_time 后返回错误信息而非 idt
    return:
        sim_time_list:  返回的是一个 list, 里面的元素是 sim.step() 演化的时间
        mole:           返回的是一个 list, 里面的元素是 species 的 mole 浓度分数
        
    """
    assert not ((cut_time != 0. and min_steps != 0) or (cut_time == 0. and min_steps == 0))
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim_time, mole = 0, 0
    if mode != 'default':
        sim_time_list = []; mole_list = []
    sim.max_time_step = max_time_step
    
    # cut_time 模式
    if cut_time != 0.:
        while sim_time < cut_time:  
            old_sim_time = sim_time
            sim_time = sim.step()
            # 若演化步长太小,则手动演化到下一时刻
            if sim_time - old_sim_time < time_step_min:
                sim_time += time_step_min
                sim.advance(sim_time)
            if mode != 'default':
                sim_time_list.append(sim_time)
                mole_list.append(species2mole(gas, species))
        # 是否达到最大模拟时间
        if sim_time > 10 * cut_time:
            if raise_error: raise ValueError("solvemole:The simulation time is beyond the 10 * cut time!")
    # minstep 模式
    elif min_steps != 0:
        for _ in range(min_steps):
            sim_time = sim.step()
            if mode != 'default':
                sim_time_list.append(sim_time)
                mole_list.append(species2mole(gas, species))
    # 模式：获取点火曲线
    if mode == 'get_curve':
        return sim_time_list, mole_list
    if mode == 'get_max':
        return sim_time_list[np.argmax(mole_list)], np.amax(mole_list)
    if mode == 'get_curve_max':
        index = get_optim_point(mole_list, sim_time_list, need_index = True, mode = 'idt')
        return sim_time_list[index], mole_list[index]
    else:
        mole = species2mole(gas, species)
        return mole


def solve_mole_steady(gas: ct.Solution, species, **kwargs):
    """
    返回指定 species 稳定状态的 mole分数, 直接使用 sim.advance_to_steady 计算稳定时的 mole分数
    params:
        gas:            输入的是 ct.Solution, 且已经设置过当量比和 TP
        species:        需要获得mole浓度分数的 species,支持 list 输入
        kwargs:
            输入 ct.ReactorNet 类中 advance_to_steady_state 的关键字参数
            int max_steps=10000, 
            double residual_threshold=0., 
            double atol=0., 
            bool return_residuals=False
    return:
        若advance_to_steady_state 不返回值则返回稳态下的 mole fraction \n
        否则返回稳态下的 mole fraction + advance_to_steady_state的值
    """
    # advance_to_steady_state 的关键字参数
    max_steps = kwargs.get('max_steps', 10000); residual_threshold = kwargs.get('residual_threshold', 0.); 
    atol = kwargs.get('atol', 0.); return_residuals = kwargs.get('return_residuals', False)

    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim.set_initial_time(0)
    sim.advance_to_steady_state(max_steps, residual_threshold, atol, return_residuals)
    mole = species2mole(gas, species)
    return mole


def yaml2mole_time(chem_file:str, species: str, mode = 'get_max', cut_time:float | list = 1, 
                   setup_file:str = None, mole_condition = None, mole_T = None, mole_P = None, 
                   mole_phi = None,  fuel: str| list = None, oxidizer: str| list = None,
                   save_path:str = None, **kwargs):
    """
    求解 mole 延迟时间
    params:
        chem_file
        cut_time: 截止时间; 默认为 1, 可设置为真实点火的倍数
        setup_file: 保存基础设置的 setup.yaml; 请见模板
        mole_condition: 如果 mole_condition 是一维数组，在此情况下 mole_condition 一定是单条数据。这样将返回一个单值的 mole 和 T 而非列表
        mole_T; mole_P; mole_phi
        fuel; oxidizer
        save_path: 默认为 None
        mode:           solve mole 的模式，有以下几种选择：
            default:    默认值，返回 mole
            get_curve:  返回整个系统演化的时间mole浓度列表
            get_curve_max:    返回曲线斜率的最大值和对应的浓度
            get_max:    返回曲线的最大值和对应的浓度/ 这个是对应 IDT 的含义
    如果想要恢复 三维张量,请使用
    np.reshape(len(phi), len(T), len(P))
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        mole_T, mole_P, mole_phi = chem_args['mole_T'], chem_args['mole_P'], chem_args['mole_phi']
        mole_condition = np.array(
                    [[phi, T, P] for phi in mole_phi for T in mole_T for P in mole_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif mole_condition is None:
        mole_condition = np.array(
                    [[phi, T, P] for phi in mole_phi for T in mole_T for P in mole_P]
                )
    
    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    # 单值下快速运算
    if mole_condition.ndim == 1:
        gas = ct.Solution(chem_file)
        phi, T, P = mole_condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        mole, _ = solve_mole(gas, species = species, mode = mode, cut_time = tmp_cut_time, **kwargs)
        if not save_path is None:
            np.savez(save_path, mole = mole)
        return mole
    else:
        Mole = []
        fuel = [fuel] * len(mole_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(mole_condition) if not isinstance(oxidizer, list) else oxidizer
        mole, Temperature = [], []; cut_time = cut_time * np.ones(len(mole_condition)) if not isinstance(cut_time, Iterable) else cut_time
        for tmp_cut_time, condition, tmp_fuel, tmp_oxidizer in zip(cut_time, mole_condition, fuel, oxidizer):
            gas = ct.Solution(chem_file)
            phi, T, P = condition
            gas.TP = T, P * ct.one_atm
            gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
            mole, _ = solve_mole(gas, species = species, mode = mode, cut_time = tmp_cut_time, **kwargs)
            Mole.append(mole)
        Mole = np.array(Mole)
        if not save_path is None:
            np.savez(save_path, Mole = Mole)
        return Mole
    

def yaml2mole_curve(chem_file:str, species: str, cut_time:float | list = 0, min_steps:int = 0,
                   setup_file:str = None, mole_condition = None, mole_T = None, mole_P = None, 
                   mole_phi = None,  fuel: str| list = None, oxidizer: str| list = None,
                   save_path:str = None, **kwargs):
    """ 
    返回 mole 的曲线 
    return:
        sim_time_list, mole_list
    
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        mole_T, mole_P, mole_phi = chem_args['mole_T'], chem_args['mole_P'], chem_args['mole_phi']
        mole_condition = np.array(
                    [[phi, T, P] for phi in mole_phi for T in mole_T for P in mole_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif mole_condition is None:
        mole_condition = np.array(
                    [[phi, T, P] for phi in mole_phi for T in mole_T for P in mole_P]
                )
    
    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    # 单值下快速运算
    if mole_condition.ndim == 1:
        gas = ct.Solution(chem_file)
        phi, T, P = mole_condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        sim_time_list, mole_list = solve_mole(gas, species = species, mode = 'get_curve', cut_time = cut_time, min_steps = min_steps, **kwargs)
        if not save_path is None:
            np.savez(save_path, sim_time_list = sim_time_list, mole_list = mole_list)
        return np.array(sim_time_list), np.array(mole_list)
    else:
        Mole = []; Timelist = []
        fuel = [fuel] * len(mole_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(mole_condition) if not isinstance(oxidizer, list) else oxidizer
        cut_time = cut_time * np.ones(len(mole_condition)) if not isinstance(cut_time, Iterable) else cut_time
        for tmp_cut_time, condition, tmp_fuel, tmp_oxidizer in zip(cut_time, mole_condition, fuel, oxidizer):
            gas = ct.Solution(chem_file)
            phi, T, P = condition
            gas.TP = T, P * ct.one_atm
            gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
            sim_time_list, mole_list = solve_mole(gas, species = species, mode = 'get_curve', cut_time = tmp_cut_time, min_steps = min_steps, **kwargs)
            Mole.append(mole_list); Timelist.append(sim_time_list)
        Mole = np.array(Mole); Timelist = np.array(Timelist)
        if not save_path is None:
            np.savez(save_path, Mole = Mole, Timelist = Timelist)
        return Timelist, Mole


def solve_idt_mole_MaxSlopeIntersect(gas, s = 'OH', species = 'OH', cut_time = 0, min_steps = 0, max_time_step = 1e-6, time_multiple = 1.2, 
    time_step_min = 1e-7, **kwargs):
    """
    同时求解 IDT 与当前 species 最大 mole 浓度
    IDT 的定义为 s 的去激发的最大变化率(即化学发光)绘制的线与定义零浓度水平的水平线相交之间的时间
    params:
        gas: 需要已经设定好当量比和 TP
        s: IDT 定义的基准
        species: 想要获得的最大 mole 浓度的物种
        cut_time:       最大演化时间, 将返回这个时间的 mole 浓度; 默认值为 0, 0表示不设置 cut_time
        min_steps:      最大演化步长, 不论如何 sim.step() 都会执行这么多次; 不应同时和 cut_time 一起设置但两者必须有一个设置; 默认为 0
        time_step_min:  自适应演化步长的最小值,小于该值则手动演化到下一时刻
        max_time_step:  sim.step() 最大的单步演化时间
    return:
        IDT, mole
    """
    assert not ((cut_time != 0. and min_steps != 0) or (cut_time == 0. and min_steps == 0))
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim_time, mole, idt = 0, 0, 0
    sim.max_time_step = max_time_step
    sim_time_list, X_OH_list = [], []; mole_list = []

    # cut_time 模式
    if cut_time != 0.:
        while sim_time < cut_time:  
            old_sim_time = sim_time
            X_OH_list.append(species2mole(gas, s)); mole_list.append(species2mole(gas, species))
            sim_time_list.append(sim_time)
            sim_time = sim.step()

            # 若演化步长太小，则手动演化到下一时刻
            if sim_time - old_sim_time < time_step_min:
                sim_time += time_step_min
                sim.advance(sim_time)

        # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
        while sim_time < time_multiple * cut_time:
            old_sim_time = sim_time
            X_OH_list.append(species2mole(gas, s)); mole_list.append(species2mole(gas, species))
            sim_time_list.append(sim_time)
            sim_time = sim.step()
            if sim_time - old_sim_time < time_step_min:
                sim_time += time_step_min
                sim.advance(sim_time)
    if min_steps != 0:
        for _ in range(min_steps):
            sim_time = sim.step()
            sim_time_list.append(sim_time)
            X_OH_list.append(species2mole(gas, s)); mole_list.append(species2mole(gas, species))

    ##选取oh浓度斜率与零水平交点
    dOH = np.array([(X_OH_list[i+1] - X_OH_list[i]) / (sim_time_list[i+1] - sim_time_list[i])  for i in range(len(X_OH_list)-1)])
    argmax_dOH = np.argmax(dOH)
    ##求直线方程交点
    idt = sim_time_list[argmax_dOH] - (X_OH_list[0]-X_OH_list[argmax_dOH]) / dOH[argmax_dOH]

    # 求最大 Mole 对应的时间
    mole = np.amax(mole_list)

    return idt, mole


'''============================================================================================================='''
'''                                                反应通路流量                                                    '''
'''============================================================================================================='''

def solve_forward_progress(gas:ct.Solution, time_series = None, cut_time = 1e-2):
    """
    计算某一个反应在反应过程中正反应速率的变化情况，其中时间由 time_series 决定
    params:
        gas: 需要在函数外设置好温度压强和当量比
        reaction_name: 反应的名称，若为 None 需要提供反应序号
        reaction_num: 反应的序号，与名称互补
        time_series: float/list 反应过程中的时间序列，若为 None 将按照 sim.step() 由 cantera 自行决定
        cut_time: 反应终止时间
    return:
        forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates        
    """
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim_time = 0
    if isinstance(time_series, float):
        forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates = [], [], [], []
        while sim_time < cut_time:  # 点火延迟时间定义为温度与初始温度的差大于400K
            sim_time += time_series; sim.advance(sim_time)
            forward_rate_constants.append(gas.forward_rate_constants)
            forward_rates_of_progress.append(gas.forward_rates_of_progress)
            heat_production_rates.append(gas.heat_production_rates)
            net_production_rates.append(gas.net_production_rates)
        
    elif isinstance(time_series, list) or isinstance(time_series, np.ndarray):
        forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates = [], [], [], []
        for time in time_series:  # 点火延迟时间定义为温度与初始温度的差大于400K
            sim.advance(time)
            forward_rate_constants.append(gas.forward_rate_constants)
            forward_rates_of_progress.append(gas.forward_rates_of_progress)
            heat_production_rates.append(gas.heat_production_rates)
            net_production_rates.append(gas.net_production_rates)

    elif time_series is None:
        sim.max_time_step = 0.1; time_step_min = 1e-7; ini_temperature = r.T; time_series = []
        forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates = [], [], [], []
        # 点火延迟时间定义为温度与初始温度的差大于400K; 目的是演化到1.5倍的点火延迟时间
        diff_T = 0
        while diff_T <= 400:  
            old_sim_time = sim_time
            sim_time = sim.step()
            # 若演化步长太小,则手动演化到下一时刻
            if sim_time - old_sim_time < time_step_min:
                sim_time += time_step_min
                sim.advance(sim_time)
            time_series.append(sim_time); diff_T = r.T - ini_temperature
            forward_rate_constants.append(gas.forward_rate_constants)
            forward_rates_of_progress.append(gas.forward_rates_of_progress)
            heat_production_rates.append(gas.heat_production_rates)
            net_production_rates.append(gas.net_production_rates)
            assert len(time) == len(forward_rates_of_progress)
        idt = sim_time
        while sim_time < 1.5* idt:
            old_sim_time = sim_time
            sim_time = sim.step()
            # 若演化步长太小,则手动演化到下一时刻
            if sim_time - old_sim_time < time_step_min:
                sim_time += time_step_min
                sim.advance(sim_time)
            time_series.append(sim_time)
            forward_rate_constants.append(gas.forward_rate_constants)
            forward_rates_of_progress.append(gas.forward_rates_of_progress)
            heat_production_rates.append(gas.heat_production_rates)
            net_production_rates.append(gas.net_production_rates)
            assert len(time_series) == len(forward_rates_of_progress)
            
    forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates = \
                    np.array(forward_rate_constants), np.array(forward_rates_of_progress), np.array(heat_production_rates), np.array(net_production_rates)
    return forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates, np.array(time_series)


def yaml2FP(chem_file, time_series, T, P, phi, fuel, oxidizer, cut_time = 1e-2):
    """
    将 YAML 文件直接计算相应的 forward_rate_constants, forward_rates_of_progress, 
    heat_production_rates, net_production_rates 四项，完全使用 solve_forward_progress 中的设置
    params:
        chem_file; T, P, phi (只接受单个工况输入); fuel; oxidizer
    """
    gas = ct.Solution(chem_file); gas.TP = T, P * ct.one_atm; gas.set_equivalence_ratio(phi, fuel, oxidizer)
    # forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates, time = \
    #     solve_forward_progress(gas, time_series = time_series, cut_time = cut_time)
    # assert len(time) == len(forward_rates_of_progress)
    return solve_forward_progress(gas, time_series = time_series, cut_time = cut_time)


def yaml2total_forward_flux(chem_file, time_series, T, P, phi, fuel, oxidizer, cut_time = 10):
    """
    计算总的正反应通量
    params:
        chem_file; T, P, phi (只接受单个工况输入); fuel; oxidizer
    """
    gas = ct.Solution(chem_file); gas.TP = T, P * ct.one_atm; gas.set_equivalence_ratio(phi, fuel, oxidizer)
    forward_rate_constants, forward_rates_of_progress, heat_production_rates, net_production_rates, time = \
        solve_forward_progress(gas, time_series = time_series, cut_time = cut_time)
    # 使用朴实的积分计算
    ## 计算 time 的间隔值
    time_interval = np.diff(time)
    ## 计算正反应通量的积分值; 通量使用前后两点的均值
    total_forward_flux = np.sum((forward_rates_of_progress[1:] + forward_rates_of_progress[:-1]) / 2 * time_interval[:, np.newaxis], axis = 0)
    return total_forward_flux
       
        
"""========================================================================================================="""
"""                                              HRR                                                       """
"""========================================================================================================="""

def solve_hrr(gas: ct.Solution, cut_time = 1, idt_defined_T_diff = 400, time_multiple = 1.2, time_step_min = 1e-7, 
              ign_error_return = 1e-10, **kwargs):
    """
    返回 0D 容器点火 IDT + 最大热释放率的数据
    params:
        gas:            输入的是 ct.Solution, 且已经设置过当量比
        cut_time:       最大演化时间,超时仍未点火则返回该时间
        time_multiple:  终止时刻为多少倍的点火延迟时间
        time_step_min:  自适应演化步长的最小值,小于该值则手动演化到下一时刻
        idt_defined_T_diff: 点火延迟时间定义为温度与初始温度的差大于400K
        ign_error_return:   当点火延迟时间小于该值时,返回该值
    return:
        idt:            点火延迟时间
        max_hrr:        最大热释放率
        T:              末态温度
    """
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim_time, idt, diff_T = 0, 0, 0; ini_temperature = r.T
    sim.max_time_step = 0.1
    hrr_list = []
    
    # 点火延迟时间定义为温度与初始温度的差大于400K
    while diff_T < idt_defined_T_diff:  
        old_sim_time = sim_time
        sim_time = sim.step()
        # 若演化步长太小,则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        diff_T = r.T - ini_temperature
        hrr_list.append(r.kinetics.heat_release_rate)
        # 是否达到最大模拟时间
        if sim_time > cut_time:
            if idt < 1e-8:
                idt = cut_time
            break
    idt = sim_time

    # 再往后演化一段时间,不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        sim_time = sim.step()
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
    hrr = max(hrr_list); hrr = max(hrr, ign_error_return)
    return idt, r.T, hrr


def yaml2idt_hrr(chem_file:str, IDT_condition = None, IDT_T = None, IDT_P = None, IDT_phi = None, 
                cut_time:float | list = 1, setup_file:str = None, IDT_mode = 0, idt_defined_T_diff = 400,
                fuel: str| list = None, oxidizer: str| list = None, save_path:str = None,
                idt_defined_species = 'OH', return_condition = False,  **kwargs):
    """
    将 YAML 文件直接计算相应的 idt, hrr, T
    params:
        chem_file:      YAML 文件路径
        cut_time:       最大演化时间,超时仍未点火则返回该时间
        setup_file:     YAML 文件中的 setup 文件路径, 默认为 None, 即不设置
        IDT_condition:  IDT 的工况, 默认为 None, 即不设置
        IDT_T:          IDT 的温度, 默认为 None, 即不设置
        IDT_P:          IDT 的压力, 默认为 None, 即不设置
        IDT_phi:        IDT 的当量比, 默认为 None, 即不设置
        fuel:           燃料, 默认为 None, 即不设置
        oxidizer:       氧化剂, 默认为 None, 即不设置
        save_path:      保存路径, 默认为 None, 即不保存
    return:
        idt:            点火延迟时间
        max_hrr:        最大热释放率
        T:              末态温度
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        IDT_T, IDT_P, IDT_phi = chem_args['IDT_T'], chem_args['IDT_P'], chem_args['IDT_phi']
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif IDT_condition is None:
        IDT_condition = np.array(
                    [[phi, T, P] for phi in IDT_phi for T in IDT_T for P in IDT_P]
                )

    assert fuel is not None and oxidizer is not None, '请设置 fuel 和 oxidizer'
    # 单值下快速运算
    if IDT_condition.ndim == 1:
        assert not isinstance(cut_time, Iterable), '单值模式下 cut_time 只能为 float 等单值'
        gas = ct.Solution(chem_file)
        phi, T, P = IDT_condition
        gas.TP = T, P * ct.one_atm
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        idt, Tem, maxhrr = solve_idt_hrr(gas, IDT_mode = IDT_mode, idt_defined_species= idt_defined_species, cut_time = cut_time, idt_defined_T_diff = idt_defined_T_diff, **kwargs)
        if not save_path is None:
            np.savez(save_path, IDT = idt, T = Tem, max_hrr = maxhrr)
        if return_condition:
            return idt, maxhrr, Tem, IDT_condition
        else:
            return idt, maxhrr, Tem
    else:
        fuel = [fuel] * len(IDT_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(IDT_condition) if not isinstance(oxidizer, list) else oxidizer
        IDT, Temperature, Maxhrr = [], [], []; cut_time = cut_time * np.ones(len(IDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
        for tmp_cut_time, condition, tmp_fuel, tmp_oxidizer in zip(cut_time, IDT_condition, fuel, oxidizer):
            gas = ct.Solution(chem_file)
            phi, T, P = condition
            gas.TP = T, P * ct.one_atm
            gas.set_equivalence_ratio(phi, tmp_fuel, tmp_oxidizer)
            idt, Tem, maxhrr = solve_idt_hrr(gas, IDT_mode = IDT_mode, idt_defined_species= idt_defined_species, cut_time = tmp_cut_time, idt_defined_T_diff = idt_defined_T_diff, **kwargs)
            IDT.append(idt); Temperature.append(Tem); Maxhrr.append(maxhrr)
        IDT, Temperature = np.array(IDT), np.array(Temperature)
        
        if not save_path is None:
            np.savez(save_path, IDT = IDT, T = Temperature, max_hrr = Maxhrr)
        if return_condition:
            return IDT, Maxhrr, Temperature, IDT_condition
        else:
            return IDT, Maxhrr, Temperature
    
    
'''============================================================================================================='''
'''                                                  数据转换                                                    '''
'''============================================================================================================='''

# 给定一个组分的01向量,返回对应的反应,这些反应的反应物生成物不包含删除的组分
def species2reaction(vector, all_species, all_reactions, ref_phase):
    # 删除的组分
    rm_species = []
    for i, S in enumerate(all_species):
        if vector[i] == 0:
            rm_species.append(S.name)
    sub_reactions = []
    for R in all_reactions:
        species_names = {}  # 获得简化机理涉及的组分
        species_names.update(R.reactants)
        species_names.update(R.products)
        R_species = [ref_phase.species(name).name for name in species_names]
        common_species = [x for x in rm_species if x in R_species] #两个列表表都存在的元素
        if len(common_species) == 0:
                sub_reactions.append(R)
    return sub_reactions

# need_N2: 给定01向量,返回对应的组分并加上组分"N2",若为False,则不会刻意加N2
def get_sub_spicies(vector, all_species, need_N2 = True):
    sub_species = []
    for i, S in enumerate(all_species):
        if need_N2:
            if vector[i] == 1 or S.name == "N2":
                sub_species.append(S)
        else:
            if vector[i] == 1:
                sub_species.append(S)
    return sub_species

# 给定01向量和详细机理的路径,返回简化机理的gas和简化组分和反应
def get_sub_mechanism(vector, detail_mechanism_path, need_N2 = True):
    all_species = ct.Species.listFromFile(detail_mechanism_path)
    ref_phase = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species)
    all_reactions = ct.Reaction.listFromFile(detail_mechanism_path, ref_phase)
    species = get_sub_spicies(vector, all_species, need_N2)
    reactions = species2reaction(vector, all_species, all_reactions, ref_phase)
    gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=species, reactions=reactions)
    return gas, species, reactions


