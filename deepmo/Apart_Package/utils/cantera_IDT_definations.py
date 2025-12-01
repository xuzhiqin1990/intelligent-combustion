# -*- coding:utf-8 -*-
import cantera as ct
import numpy as np
import warnings
from collections.abc import Iterable

def is_temperature_stable(timelist, temperaturelist, threshold=0.01, stable_duration=100):
    """
    判断温度是否已经保持不变
    params:
        timelist: 时间列表
        temperaturelist: 温度列表
        threshold: 判断温度变化是否在阈值范围内
        stable_duration: 需要温度变化在阈值范围内的持续时间

    return:
        bool: 温度是否已经保持不变
    """
    def max_consecutive_trues(bool_array, trues = True):
        count = max_count = 0
        for val in bool_array:
            count = count + 1 if val == trues else 0
            max_count = max(max_count, count)
        return max_count
    
    if len(temperaturelist) < 2:
        return False  # 数据不足
    
    deltalist = np.abs(np.array([temperaturelist[i+1]-temperaturelist[i] for i in range(len(temperaturelist)-1)]))
    # deltatime = np.array([timelist[i+1]-timelist[i] for i in range(len(temperaturelist)-1)])
    threshold = np.max(deltalist) * threshold + np.min(deltalist)
    # start point 为 deltalist > threshold 的第一个点对应的索引值
    start_point = np.where(deltalist > threshold)[0][0] if isinstance(np.where(deltalist > threshold)[0], Iterable) else np.where(deltalist > threshold)[0]
    # 计算一阶差分
    temp_diff = deltalist[start_point:]
    
    # 判断差分值是否在阈值范围内
    stable_points = np.abs(temp_diff) < threshold
    if not np.any(stable_points):
        return False
    over_stable_nums = max_consecutive_trues(stable_points, trues=False)
    if over_stable_nums > stable_duration // 2:
        return False
    # 判断是否有连续 stable_duration 个点都在阈值范围内
    # 用 numpy 实现查找是否存在连续的 True
    stable_nums = max_consecutive_trues(stable_points)
    # print(stable_nums)
    return stable_nums >= stable_duration


def is_sequence_suddenly_changing(sequence, time_sequence, delta_threshold=1e6):
    """
    判断一个序列是否开始突然变化，突然变化的定义是序列和时间序列的一阶差商大于 delta_threshold

    参数:
    sequence (list or array): 要检查的序列
    time_sequence (list or array): 对应的时间序列

    返回:
    bool: 如果序列开始突然变化，返回第一个大于阈值的位置，否则返回 False
    """
    if len(sequence) != len(time_sequence):
        raise ValueError("序列和时间序列的长度必须相同")

    delta_sequence = np.diff(sequence)
    delta_time = np.diff(time_sequence)
    rate_of_change = delta_sequence / delta_time
    if any(rate_of_change > delta_threshold):
        # 返回第一个大于阈值的位置
        sudden_change_pos = np.where(rate_of_change > delta_threshold)[0][0]
        return sudden_change_pos
    
    return False


"""========================================================================================================="""
"""                             Other Definition of IDTs: OH definition                                     """
"""========================================================================================================="""

def idt_definition_OH_half_max(gas, 
                               idt_defined_species= 'OH', 
                               cut_time = 1, 
                               time_multiple = 3, 
                               need_maxHRR = False,
                               time_step_min = 1e-7, 
                               idt_defined_T_diff:float = 400, **kwargs):  
    """
    OH *发射达到最大值的50%所用的时间
    """
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    sim_time_list, X_OH_list = [], []
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    hrr_list = []
    while diff_T < idt_defined_T_diff: 
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()

        # 若演化步长太小，则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
            hrr_list.append(r.kinetics.heat_release_rate)
        diff_T = r.T - ini_temperature 

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            idt = cut_time
            return idt, r.T
    idt = sim_time

    # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        
    # 选取OH浓度达到最大值一半的时间作为idt
    X_OH_list = np.array(X_OH_list); half_OH = max(X_OH_list) / 2
    # 找到 X_OH_LIST 第一次跨越 half_OH 的位置
    k = 0
    while X_OH_list[k] < half_OH:
        k += 1
    idt = sim_time_list[k]; maxHRR = max(hrr_list)
    if need_maxHRR:
        return idt, r.T, maxHRR

    return idt, r.T

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

def idt_definition_OH_slope_max_intersect(
    gas, 
    idt_defined_species='OH', 
    cut_time=1, 
    time_multiple=50, 
    time_step_min=1e-8, 
    need_maxHRR=False,
    slope_tolerance=1e-2,
    idt_defined_T_diff: float = 50,
    need_OH=False,
    save_traj_dir = None,
    P_mean=None,
    stable_duration=100,
    **kwargs):
    """
    OH去激发的最大变化率(即化学发光)绘制的线与定义零浓度水平的水平线相交之间的时间
    """
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    sim_time_list, X_OH_list = [], []
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    hrr_list = []
    temperature_list = []
    P_list = []

    while diff_T < idt_defined_T_diff: 
        old_sim_time = sim_time
        X_OH_list.append(gas.X[gas.species_index(idt_defined_species)])
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        P_list.append(gas.P)
        sim_time = sim.step()
        temperature_list.append(r.T)
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        diff_T = r.T - ini_temperature 
        if sim_time > cut_time:
            idt = cut_time
            warnings.warn(f'Reach the max simulation time; idt = cut_time; The initial T is {ini_temperature}; The final T is {r.T}')
            break
    
    stable_flag = is_temperature_stable(sim_time_list, temperature_list, 
                                        threshold=0.01, stable_duration=stable_duration)
    if sim_time > cut_time:
        pass
    else:
        if not stable_flag:
            post_simulation_count = 0
            stable_flag = False
            while (not stable_flag):
                old_sim_time = sim_time
                X_OH_list.append(gas.X[gas.species_index(idt_defined_species)])
                sim_time_list.append(sim_time)
                hrr_list.append(r.kinetics.heat_release_rate)
                temperature_list.append(r.T)
                P_list.append(gas.P)
                sim_time = sim.step()
                if sim_time - old_sim_time < time_step_min:
                    sim_time += time_step_min
                    sim.advance(sim_time)
                post_simulation_count += 1
                if post_simulation_count % 10 == 0:
                    stable_flag = is_temperature_stable(sim_time_list, temperature_list, 
                                                        threshold=0.01, stable_duration=stable_duration)
                    if stable_flag:
                        break
                if sim_time > cut_time:
                    stable_flag = True
                    break
    
    # 计算Plist 出现陡增的位置
    P_list = np.array(P_list)
    sim_time_list = np.array(sim_time_list)
    sudden_change_pos = is_sequence_suddenly_changing(P_list, sim_time_list)
    if not sudden_change_pos:
        return 1, r.T

    # 计算OH斜率
    dOH = np.array([
        (X_OH_list[i+1] - X_OH_list[i]) / (sim_time_list[i+1] - sim_time_list[i])
        for i in range(len(X_OH_list)-1)
    ])
    dOH[:sudden_change_pos] = 0
    argmax_dOH = np.argmax(dOH)

    # 求直线方程交点
    if dOH[argmax_dOH] < slope_tolerance:
        warnings.warn(f'dOH[argmax_dOH] {dOH[argmax_dOH]} 小于 slope_tolerance {slope_tolerance}; 此时 idt = sim_time_list[argmax_dOH]')
        idt = sim_time_list[argmax_dOH]
    else:
        idt = sim_time_list[argmax_dOH] - (X_OH_list[argmax_dOH] - X_OH_list[0]) / dOH[argmax_dOH]
    assert idt <= sim_time_list[argmax_dOH], 'idt > sim_time_list[argmax_dOH]; which is impossible'
    # # 打印调试
    # print(f'slope of OH: {dOH[argmax_dOH]}; idt = {idt}; sim_time_list[argmax_dOH] = {sim_time_list[argmax_dOH]}')

    if idt == 0:
        idt = sim_time_list[-1]
        warnings.warn('idt 模拟为 0; 请检查是否发生化学发光; 此时取最大模拟时间')
    maxHRR = max(hrr_list)
    if idt < 0:
        idt = 1.0
        warnings.warn(f"Negative IDT detected ({idt:.2f}); forcing IDT to 1.0")
    maxHRR = max(hrr_list)

    # ===================== 绘图并保存 =====================
    # save_dir = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35_alpha0.1_from_circ3/validation_results/He/IDT"
    # os.makedirs(save_dir, exist_ok=True)
    # # 获取phi等参数
    # try:
    #     phi_used = gas.equivalence_ratio()
    # except Exception:
    #     phi_used = 0
    # filename = f"OH_vs_time_T{int(ini_temperature)}K_P{gas.P/ct.one_atm:.2f}atm_phi{phi_used:.2f}.png"
    # save_path = os.path.join(save_dir, filename)

    # plt.figure(figsize=(7, 4), dpi=120)
    # plt.plot(sim_time_list, X_OH_list, label='[OH] vs time')
    # plt.scatter([sim_time_list[argmax_dOH]], [X_OH_list[argmax_dOH]], color='r', zorder=5, label='max d[OH]/dt')
    # plt.xlabel('Time (s)')
    # plt.ylabel('[OH] (mole fraction)')
    # plt.xscale('log')
    # plt.title('OH Evolution (debug)')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(save_path)
    # plt.close()
    # print(f"已保存OH轨线图: {save_path}")
    # ===================== 绘图并保存 =====================
        # 保存轨线到文件（新增部分）
    if save_traj_dir is not None:
        os.makedirs(save_traj_dir, exist_ok=True)
        try:
            phi_used = gas.equivalence_ratio()
        except Exception:
            phi_used = 0
        P_to_save = P_mean if P_mean is not None else gas.P/ct.one_atm
        fname = f"oh_traj_T{int(ini_temperature)}K_P{P_to_save:.2f}atm_phi{phi_used:.2f}.npz"
        save_path = os.path.join(save_traj_dir, fname)
        np.savez(save_path, 
                 time=np.array(sim_time_list), 
                 xoh=np.array(X_OH_list),
                 temp=np.array(temperature_list),
                 doh=np.array([
                      (X_OH_list[i+1] - X_OH_list[i]) / (sim_time_list[i+1] - sim_time_list[i])
                      for i in range(len(X_OH_list)-1)
                 ]),
                 P=np.array(P_list),
                 idt=idt)
        print(f"[DEBUG] OH轨线已保存到: {save_path}")
    if need_maxHRR:
        return idt, r.T, maxHRR
    if need_OH:
        return idt, r.T, np.array(sim_time_list), np.array(X_OH_list), np.array(temperature_list), dOH, np.array(P_list)
    return idt, r.T

# def idt_definition_OH_slope_max_intersect(
#     gas, 
#     idt_defined_species= 'OH', 
#     cut_time = 1, 
#     time_multiple = 50, 
#     time_step_min = 1e-7, 
#     need_maxHRR = False,
#     slope_tolerance = 1e-4,
#     idt_defined_T_diff:float = 50,
#     need_OH = False,
#     **kwargs):
#     """
#     OH去激发的最大变化率(即化学发光)绘制的线与定义零浓度水平的水平线相交之间的时间
    
#     20240225: add slope_tolerance to avoid the case that the slope is too small, so that the idt is negative
#     """
#     r = ct.IdealGasReactor(gas)
#     sim = ct.ReactorNet([r])

#     sim_time_list, X_OH_list = [], []
#     sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
#     hrr_list = []
#     temperature_list = []
#     P_list = []

#     while diff_T < idt_defined_T_diff: 
#         old_sim_time = sim_time
#         X_OH_list.append(gas.X[gas.species_index(idt_defined_species)])
#         sim_time_list.append(sim_time)
#         hrr_list.append(r.kinetics.heat_release_rate)
#         P_list.append(gas.P)
#         sim_time = sim.step()
#         temperature_list.append(r.T)
#         # 若演化步长太小，则手动演化到下一时刻
#         if sim_time - old_sim_time < time_step_min:
#             sim_time += time_step_min
#             sim.advance(sim_time)
        
#         diff_T = r.T - ini_temperature 

#         # 是否达到最大模拟时间
#         if sim_time > cut_time:
#             idt = cut_time
#             warnings.warn(f'Reach the max simulation time; idt = cut_time; The initial T is {ini_temperature}; The final T is {r.T}')
#             break
#             # return idt, r.T
#     stable_flag = is_temperature_stable(sim_time_list, temperature_list)
#     if sim_time > cut_time:
#         pass
#     else:
#         # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
#         if not stable_flag:
#             post_simulation_count = 0
#             stable_flag = False
#             while (not stable_flag):
#                 old_sim_time = sim_time
#                 X_OH_list.append(gas.X[gas.species_index(idt_defined_species)])
#                 sim_time_list.append(sim_time)
#                 hrr_list.append(r.kinetics.heat_release_rate)
#                 temperature_list.append(r.T)
#                 P_list.append(gas.P)
#                 sim_time = sim.step()
#                 if sim_time - old_sim_time < time_step_min:
#                     sim_time += time_step_min
#                     sim.advance(sim_time)
#                 post_simulation_count += 1
#                 if post_simulation_count % 10 == 0:
#                     stable_flag = is_temperature_stable(sim_time_list, temperature_list)
#                     if stable_flag:
#                         break
#                 if sim_time > cut_time:
#                     stable_flag = True
#                     break
        
#     # 计算Plist 出现陡增的位置
#     P_list = np.array(P_list); sim_time_list = np.array(sim_time_list)
#     sudden_change_pos = is_sequence_suddenly_changing(P_list, sim_time_list)
#     if not sudden_change_pos:
#         return 1,r.T
    
#     ##选取oh浓度斜率最大的位置
#     dOH = np.array([(X_OH_list[i+1] - X_OH_list[i]) / (sim_time_list[i+1] - sim_time_list[i])  for i in range(len(X_OH_list)-1)])

#     dOH[:sudden_change_pos] = 0
#     argmax_dOH = np.argmax(dOH)

#     ##求直线方程交点
#     if dOH[argmax_dOH] < slope_tolerance:
#         warnings.warn(f'dOH[argmax_dOH] {dOH[argmax_dOH]} 小于 slope_tolerance {slope_tolerance}; 此时 idt = sim_time_list[argmax_dOH]')
#         idt = sim_time_list[argmax_dOH]
#     else:
#         idt = sim_time_list[argmax_dOH] - (X_OH_list[argmax_dOH] - X_OH_list[0]) / dOH[argmax_dOH]
#     assert idt <= sim_time_list[argmax_dOH], 'idt > sim_time_list[argmax_dOH]; which is impossible'
#     print(f'slope of OH: {dOH[argmax_dOH]}; idt = {idt}; sim_time_list[argmax_dOH] = {sim_time_list[argmax_dOH]}')
#     # 如果最后 idt = 0 则说明没有交点，即没有发生化学发光，此时取最大模拟时间
#     if idt == 0:
#         idt = sim_time_list[-1]
#         # 发出 warn 提示此时 idt 模拟为 0
#         warnings.warn('idt 模拟为 0; 请检查是否发生化学发光; 此时取最大模拟时间')
#     maxHRR = max(hrr_list)
#     # 新增：检查IDT是否为负值
#     if idt < 0:
#         idt = 1.0
#         warnings.warn(f"Negative IDT detected ({idt:.2f}); forcing IDT to 1.0")
#     # print('idt = ', idt)
#     maxHRR = max(hrr_list)
    
#     if need_maxHRR:
#         return idt, r.T, maxHRR
#     if need_OH:
#         return idt, r.T, np.array(sim_time_list), np.array(X_OH_list), np.array(temperature_list), dOH, np.array(P_list)
#     return idt, r.T


def idt_definition_OH_max(
    gas, 
    idt_defined_species= 'OH', 
    cut_time = 1, 
    time_multiple = 4, 
    time_step_min = 1e-7, 
    need_maxHRR = False, 
    idt_defined_T_diff:float = 400, **kwargs):
    """
    点火延迟时间定义为即OH浓度最大的时间
    """
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    sim_time_list, X_OH_list, P_list = [], [], []
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    hrr_list = []
    while diff_T < idt_defined_T_diff: 
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()

        # 若演化步长太小，则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
            hrr_list.append(r.kinetics.heat_release_rate)
        diff_T = r.T - ini_temperature 

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            idt = cut_time
            return idt, r.T

    # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)

    P_list = np.array(P_list)

    ##选取oh浓度最大的时刻
    X_OH_list = np.array(X_OH_list)
    k = np.argmax(X_OH_list)
    idt = sim_time_list[k]; maxHRR = max(hrr_list)
    if need_maxHRR:
        return idt, r.T, maxHRR
    return idt, r.T


def idt_definition_OH_slope_max(
    gas, 
    idt_defined_species= 'OH', 
    cut_time = 1, 
    time_multiple = 4, 
    time_step_min = 1e-7, 
    need_maxHRR = False, 
    idt_defined_T_diff:float = 400,
    **kwargs):
    """
    点火延迟时间定义为到达OH(或吸收)最快速增加的时刻所用时间。dOH 最大的时间
    """
    
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    sim_time_list, X_OH_list, P_list = [], [], []
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    hrr_list = []
    while diff_T < idt_defined_T_diff: 
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()

        # 若演化步长太小，则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
            hrr_list.append(r.kinetics.heat_release_rate)
        diff_T = r.T - ini_temperature 

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            idt = cut_time
            return idt, r.T

    # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        sim_time = sim.step()
        hrr_list.append(r.kinetics.heat_release_rate)
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)

    P_list = np.array(P_list)

    ##选取oh上升速度最快的时刻
    X_OH_list = np.array(X_OH_list)
    dOH = (X_OH_list[1:] - X_OH_list[:-1]) / (sim_time_list[1:] - sim_time_list[:-1])
    k = np.argmax(dOH)
    idt = sim_time_list[k]
    if need_maxHRR:
        return idt, r.T, max(hrr_list)
    return idt, r.T


"""========================================================================================================="""
"""                             Other Definition of IDTs: Pressure definition                               """
"""========================================================================================================="""


def idt_definition_pressure_slope_max(gas, cut_time = 1, time_multiple = 4, time_step = 5e-7, need_maxHRR = False, need_P = False,
                                      idt_defined_T_diff = 25, time_step_min = 1e-7, **kwargs):
    """
    点火延迟时间定义为最大压力上升速率时刻
    """
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    sim_time_list, P_list = [], []
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    hrr_list = []
    temperature_list = []

    while diff_T < idt_defined_T_diff: 
        old_sim_time = sim_time
        P_list.append(gas.P)
        # X_OH_list.append(gas.X[gas.species_index(idt_defined_species)])
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()
        temperature_list.append(r.T)
        # 若演化步长太小，则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        
        diff_T = r.T - ini_temperature 

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            idt = cut_time
            warnings.warn(f'Reach the max simulation time; idt = cut_time; The initial T is {ini_temperature}; The final T is {r.T}')
            break
            # return idt, r.T
    stable_flag = is_temperature_stable(sim_time_list, temperature_list)
    if sim_time > cut_time:
        pass
    # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
    else:
        if not stable_flag:
            post_simulation_count = 0
            stable_flag = False
            while (not stable_flag):
                old_sim_time = sim_time
                P_list.append(gas.P)
                # X_OH_list.append(gas.X[gas.species_index(idt_defined_species)])
                sim_time_list.append(sim_time)
                hrr_list.append(r.kinetics.heat_release_rate)
                temperature_list.append(r.T)
                sim_time = sim.step()
                if sim_time - old_sim_time < time_step_min:
                    sim_time += time_step_min
                    sim.advance(sim_time)
                post_simulation_count += 1
                if post_simulation_count % 10 == 0:
                    stable_flag = is_temperature_stable(sim_time_list, temperature_list)
                    if stable_flag:
                        break
                if sim_time > cut_time:
                    stable_flag = True
                    break
        
        P_list = np.array(P_list); sim_time_list = np.array(sim_time_list)
        #plot(P_list)
        diff_P = (P_list[1:] - P_list[:-1]) / (sim_time_list[1:] - sim_time_list[:-1])
        # 选取压强上升速率最大的时刻作为idt
        i = np.argmax(diff_P)
        idt = sim_time_list[i+1]
    maxHRR = max(hrr_list)
    if need_maxHRR:
        return idt, r.T, maxHRR
    elif need_P:
        return idt, r.T, np.array(sim_time_list), np.array(P_list)
    return idt, r.T


def idt_definition_pressure_slope_max_intersect(gas, cut_time = 1, time_multiple = 4, time_step_min = 1e-6, need_maxHRR = False, need_P = False,
                                      idt_defined_T_diff = 100, slope_tolerance = 1e-3, **kwargs):
    """
    点火延迟时间定义为最大压力上升速率对应的切线与初始压力水平线的交点
    20240225: add slope_tolerance to avoid the case that the slope is too small to be considered as a slope
    """
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    sim_time_list, P_list = [], []
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    hrr_list = []
    while diff_T < idt_defined_T_diff: 
        old_sim_time = sim_time
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        try:
            sim_time = sim.step()
            # 若演化步长太小，则手动演化到下一时刻
            if sim_time - old_sim_time < time_step_min:
                sim_time += time_step_min
                sim.advance(sim_time)
        except:
            print('time_step_min = ', time_step_min)
            print('sim_time = ', sim_time)
            print('sim_time + time_step_min = ', sim_time + time_step_min)
            print('sim_time_list = ', sim_time_list)
            print('P_list = ', P_list)
            print('hrr_list = ', hrr_list)
            print('-------------------')
            raise ValueError()
        diff_T = r.T - ini_temperature 
        # 是否达到最大模拟时间
        if sim_time > cut_time:
            idt = cut_time
            break
    idt = sim_time

    # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)        
    
    P_list = np.array(P_list); sim_time_list = np.array(sim_time_list)
    #plot(P_list)
    diff_P = (P_list[1:] - P_list[:-1]) / (sim_time_list[1:] - sim_time_list[:-1])
    # 选取压强上升速率最大的时刻作为idt
    argmax_dP = np.argmax(diff_P)
    idt = sim_time_list[argmax_dP] - (P_list[argmax_dP] - P_list[0]) / diff_P[argmax_dP] if diff_P[argmax_dP] > slope_tolerance else sim_time_list[argmax_dP]
    assert idt <= sim_time_list[argmax_dP], 'idt > sim_time_list[argmax_dP]; which is impossible'
    maxHRR = max(hrr_list)
    if need_maxHRR:
        return idt, r.T, maxHRR
    elif need_P:
        return idt, r.T, np.array(sim_time_list), np.array(P_list)
    return idt, r.T


"""========================================================================================================="""
"""                             Other Definition of IDTs: CH* definition                                    """
"""========================================================================================================="""


# def idt_definition_CH_slope_max_intersect(
#         gas, 
#         idt_defined_species= 'CH', cut_time = 1, time_multiple = 4, time_step_min = 1e-7, 
#         need_maxHRR = False, **kwargs):
#     """
#     点火延迟时间定义为反射冲击波到达测量位置(距离端壁2cm)到点火开始的时间间隔
#     根据CH *排放历史，通过找到最急剧上升的时间并线性外推到点火前基线来确定开始点火的时间
#     """
#     r = ct.IdealGasReactor(gas)
#     sim = ct.ReactorNet([r])

#     sim_time_list, X_OH_list, P_list = [], [], []
#     sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
#     hrr_list = []
#     while diff_T < idt_defined_T_diff: 
#         old_sim_time = sim_time
#         X_OH_list.append(r.thermo[idt_defined_species].X[0])
#         P_list.append(gas.P)
#         sim_time_list.append(sim_time)
#         hrr_list.append(r.kinetics.heat_release_rate)
#         sim_time = sim.step()

#         # 若演化步长太小，则手动演化到下一时刻
#         if sim_time - old_sim_time < time_step_min:
#             sim_time += time_step_min
#             sim.advance(sim_time)
#             hrr_list.append(r.kinetics.heat_release_rate)
#         diff_T = r.T - ini_temperature 

#         # 是否达到最大模拟时间
#         if sim_time > cut_time:
#             idt = cut_time
#             return idt, r.T

#     # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
#     while sim_time < time_multiple * idt:
#         old_sim_time = sim_time
#         X_OH_list.append(r.thermo[idt_defined_species].X[0])
#         P_list.append(gas.P)
#         sim_time_list.append(sim_time)
#         hrr_list.append(r.kinetics.heat_release_rate)
#         sim_time = sim.step()
#         if sim_time - old_sim_time < time_step_min:
#             sim_time += time_step_min
#             sim.advance(sim_time)

#     P_list = np.array(P_list)

#     X_OH_list = np.array(X_OH_list)
#     dOH = []
#     for i in range(len(X_OH_list)-1):
#         d = (X_OH_list[i+1] - X_OH_list[i])
#         dOH.append(d)
#     k = 0
#     while dOH[k] < max(dOH):
#         k += 1
#     ##求直线方程交点
#     ##CH斜率最大交点或者压力斜率最大的交点
#     idt = sim_time_list[k] - (X_OH_list[0]-X_OH_list[k])*time_step_min/dOH[k]
#     maxHRR = max(hrr_list)
#     if need_maxHRR:
#         return idt, r.T, maxHRR
#     return idt, r.T


def idt_definition_CH3_max(
    gas, 
    idt_defined_species= 'CH3', 
    cut_time = 1, 
    time_multiple = 4, 
    time_step_min = 1e-7, 
    need_maxHRR = False, 
    idt_defined_T_diff:float = 400,
    **kwargs):
    """
    点火延迟时间定义为即CH3浓度最大的时间
    """
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])

    sim_time_list, X_OH_list, P_list = [], [], []
    sim_time, idt, diff_T, ini_temperature = 0, 0, 0, r.T
    hrr_list = []
    while diff_T < idt_defined_T_diff: 
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()

        # 若演化步长太小，则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
            hrr_list.append(r.kinetics.heat_release_rate)
        diff_T = r.T - ini_temperature 

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            idt = cut_time
            return idt, r.T

    # 再往后演化一段时间，不然r.T可能无法准确表征最终火焰温度
    while sim_time < time_multiple * idt:
        old_sim_time = sim_time
        X_OH_list.append(r.thermo[idt_defined_species].X[0])
        P_list.append(gas.P)
        sim_time_list.append(sim_time)
        hrr_list.append(r.kinetics.heat_release_rate)
        sim_time = sim.step()
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)

    P_list = np.array(P_list)

    ##选取oh浓度最大的时刻
    X_OH_list = np.array(X_OH_list)
    k = 0
    while X_OH_list[k] < max(X_OH_list):
        k += 1
    idt = sim_time_list[k]
    if need_maxHRR:
        return idt, r.T, max(hrr_list)
    return idt, r.T


"""========================================================================================================="""
"""                                             Definition of DIDTs                                         """
"""========================================================================================================="""


def didt_definition_Temperature_Threshold(gas: ct.Solution, cut_time = 1, idt_defined_T_diff1:float = 55, idt_defined_T_diff2:float = 200, 
                                          time_multiple = 4, time_step_min = 1e-7, get_curve = False, need_HRR = False, **kwargs):
    """
    返回点火延迟和最终火焰温度数据,用sim.step()来进行加速
    params:
        gas:            输入的是 ct.Solution, 且已经设置过当量比
        cut_time:       最大演化时间,超时仍未点火则返回该时间
        time_multiple:  终止时刻为多少倍的点火延迟时间
        time_step_min:  自适应演化步长的最小值,小于该值则手动演化到下一时刻
        get_curve:      是否返回整个点火曲线的时间温度列表,而不是idt和T
    """
    r = ct.IdealGasReactor(gas); sim = ct.ReactorNet([r])
    sim_time, idt, diff_T = 0, 0, 0; ini_temperature = r.T; didt = 0
    if get_curve:
        sim_time_list = []
        T_list = []
    sim.max_time_step = 0.1; maxHRR_list = []
    
    # 点火延迟时间定义为温度与初始温度的差大于400K
    while diff_T < idt_defined_T_diff1:  
        old_sim_time = sim_time
        sim_time = sim.step()
        # 若演化步长太小,则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        if get_curve:
            sim_time_list.append(sim_time)
            T_list.append(r.T)
        diff_T = r.T - ini_temperature
        maxHRR_list.append(r.kinetics.heat_release_rate)

        # 是否达到最大模拟时间
        if sim_time > cut_time:
            if didt < 1e-8:
                didt = cut_time
            break
    didt = sim_time
    
    # 点火延迟时间定义为温度与初始温度的差大于400K
    while diff_T < idt_defined_T_diff2:  
        old_sim_time = sim_time
        sim_time = sim.step()
        # 若演化步长太小,则手动演化到下一时刻
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        if get_curve:
            sim_time_list.append(sim_time)
            T_list.append(r.T)
        diff_T = r.T - ini_temperature
        maxHRR_list.append(r.kinetics.heat_release_rate)

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
        maxHRR_list.append(r.kinetics.heat_release_rate)
        if sim_time - old_sim_time < time_step_min:
            sim_time += time_step_min
            sim.advance(sim_time)
        if get_curve:
            sim_time_list.append(sim_time)
            T_list.append(r.T)
    
    # 模式1：获取点火曲线
    if need_HRR:
        return didt, idt, r.T, max(maxHRR_list)
    if get_curve:
        return sim_time_list, T_list, didt, idt, r.T
    else:
        return didt, idt, r.T
        