# -*- coding:utf-8 -*-
import os
from .cantera_utils import *
from .cantera_PSR_definations import *
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from collections.abc import Iterable

from concurrent.futures import ProcessPoolExecutor, wait
from collections.abc import Iterable

# 定义任务包装函数（必须在模块顶层定义，确保可pickle）
def task_wrapper(func, index, *args, **kwargs):
    """return: (index, result)"""
    result = func(*args, **kwargs)  # 调用实际的计算函数
    return index, result  # 返回原始索引和计算结果


def yaml2idt_Mcondition(chem_file:str, mode:int = 0, setup_file:str = None, 
             IDT_condition = None, IDT_T = None, IDT_P = None, IDT_phi = None, 
             fuel: str| list = None,save_traj_dir = None,P_mean=None, oxidizer: str| list = None, cut_time:float | list = 1,  
             save_dirpath:str = None, cpu_process = 1, **kwargs):
    """
    yaml2idt 函数的多进程版本，M表示Multi-process; 在本版本中，只接受 IDT_condition 作为多进程计算的对象，
    同时运算是阻塞运算以保证数据的完整性。
    多进程使用 Python 自身的 concurrent.futures.ProcessPoolExecutor 模块，不支持 MPI 并行。
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
    
    fuel = [fuel] * len(IDT_condition) if isinstance(fuel, str) else fuel
    oxidizer = [oxidizer] * len(IDT_condition) if isinstance(oxidizer, str) else oxidizer
    cut_time = cut_time * np.ones(len(IDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
    mode = mode * np.ones(len(IDT_condition)) if not isinstance(mode, Iterable) else mode
    # 使用 ProcessPoolExecutor 模块进行多进程计算
    futures = []; RES = []; indices = []
    with ProcessPoolExecutor(max_workers = cpu_process) as executor:
        for i, condition in enumerate(IDT_condition):
            save_path = save_dirpath + f'/yaml2idt_Mcondition_{i}th.npz' if save_dirpath != None else None
            future = executor.submit(
                task_wrapper,
                yaml2idt,
                i,
                chem_file = chem_file,
                mode = mode[i],
                cut_time = cut_time[i],
                IDT_condition = condition,
                fuel = fuel[i],
                oxidizer = oxidizer[i],
                save_path = save_path,
                P_mean= P_mean,
                return_condition = True,
                save_traj_dir= save_traj_dir,
                **kwargs
            )
            futures.append(future)
    wait(futures)
    for future in as_completed(futures):
        RES.append(future.result()[1][0])  # 取出第一个元素作为结果（IDT值）
        indices.append(future.result()[0])  # 索引
    
    RES = np.array(RES); indices = np.array(indices)
    sort_indices = np.argsort(indices)
    RES = RES[sort_indices]
    return RES


def yaml2idt_hrr_Mcondition(chem_file:str, mode:int = 0, setup_file:str = None, 
             IDT_condition = None, IDT_T = None, IDT_P = None, IDT_phi = None, 
             fuel: str| list = None, oxidizer: str| list = None, cut_time:float | list = 1,  
             save_dirpath:str = None, cpu_process = 1, **kwargs):
    """
    yaml2idt_hrr 函数的多进程版本，M表示Multi-process; 使用task_wrapper包裹函数调用，
    返回结果按索引排序，保证结果顺序与输入一致。
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
    
    fuel = [fuel] * len(IDT_condition) if isinstance(fuel, str) else fuel
    oxidizer = [oxidizer] * len(IDT_condition) if isinstance(oxidizer, str) else oxidizer
    cut_time = cut_time * np.ones(len(IDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
    mode = mode * np.ones(len(IDT_condition)) if not isinstance(mode, Iterable) else mode
    # 使用 ProcessPoolExecutor 模块进行多进程计算
    futures = []; RES = []; max_hrr = []; indices = []
    with ProcessPoolExecutor(max_workers = cpu_process) as executor:
        for i, condition in enumerate(IDT_condition):
            save_path = save_dirpath + f'/yaml2idt_hrr_Mcondition_{i}th.npz' if save_dirpath != None else None
            future = executor.submit(
                task_wrapper,
                yaml2idt_hrr,
                i,
                chem_file = chem_file,
                mode = mode[i],
                cut_time = cut_time[i],
                IDT_condition = condition,
                fuel = fuel[i],
                oxidizer = oxidizer[i],
                save_path = save_path,
                return_condition = True,
                **kwargs
            )
            futures.append(future)
    wait(futures)
    for future in as_completed(futures):
        result = future.result()
        RES.append(result[1][0])  # IDT值
        max_hrr.append(result[1][1])  # max_hrr值
        indices.append(result[0])  # 索引
    
    RES = np.array(RES); max_hrr = np.array(max_hrr); indices = np.array(indices)
    sort_indices = np.argsort(indices)
    RES = RES[sort_indices]; max_hrr = max_hrr[sort_indices]
    return RES, max_hrr


def yaml2psr_Mcondition(chem_file:str, RES_TIME_LIST, PSR_condition = None, setup_file:str = None, 
              PSR_T = None, PSR_P = None, PSR_phi = None, fuel: str| list = None, oxidizer: str| list = None, 
             save_dirpath:str = None, cpu_process = 1,  **kwargs):
    """
    yaml2psr 函数的多进程版本，M表示Multi-process; 在本版本中，只接受 PSR_condition 作为多进程计算的对象，
    同时运算是阻塞运算以保证数据的完整性。
    多进程使用 Python 自身的 concurrent.futures.ProcessPoolExecutor 模块，不支持 MPI 并行。
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
    
    fuel = [fuel] * len(PSR_condition) if isinstance(fuel, str) else fuel
    oxidizer = [oxidizer] * len(PSR_condition) if isinstance(oxidizer, str) else oxidizer
    
    # 使用 ProcessPoolExecutor 模块进行多进程计算
    futures = []; RES = []; indices = []
    with ProcessPoolExecutor(max_workers = cpu_process) as executor:
        for i, condition in enumerate(PSR_condition):
            save_path = save_dirpath + f'/yaml2psr_Mcondition_{i}th.npz' if save_dirpath != None else None
            future = executor.submit(
                task_wrapper,
                yaml2psr,
                i,
                chem_file = chem_file,
                PSR_condition = condition,
                RES_TIME_LIST = RES_TIME_LIST[i],
                fuel = fuel[i],
                oxidizer = oxidizer[i],
                error_tol = 0.,
                save_path = save_path,
                **kwargs
            )
            futures.append(future)
    wait(futures)
    for future in as_completed(futures):
        RES.append(future.result()[1])  # 结果
        indices.append(future.result()[0])  # 索引
    
    RES = np.array(RES); indices = np.array(indices)
    sort_indices = np.argsort(indices)
    RES = RES[sort_indices]
    return RES


def yaml2FS_Mcondition(chem_file:str, FS_condition = None, setup_file:str = None, 
                FS_T = None, FS_P = None, FS_phi = None, fuel: str| list = None, oxidizer: str| list = None, 
                cpu_process = 1,  **kwargs):
    """
    yaml2FS 函数的多进程版本，M表示Multi-process; 在本版本中，只接受 FS_condition 作为多进程计算的对象，
    同时运算是阻塞运算以保证数据的完整性。
    多进程使用 Python 自身的 concurrent.futures.ProcessPoolExecutor 模块，不支持 MPI 并行。
    """
    if setup_file != None:
        chem_args = get_yaml_data(setup_file)
        FS_T, FS_P, FS_phi = chem_args['FS_T'], chem_args['FS_P'], chem_args['FS_phi']
        FS_condition = np.array(
                    [[phi, T, P] for phi in FS_phi for T in FS_T for P in FS_P]
                )
        fuel, oxidizer = chem_args.get('fuel', fuel), chem_args.get('oxidizer', oxidizer)
    elif FS_condition is None:
        FS_condition = np.array(
                    [[phi, T, P] for phi in FS_phi for T in FS_T for P in FS_P]
                )
    
    fuel = [fuel] * len(FS_condition) if isinstance(fuel, str) else fuel
    oxidizer = [oxidizer] * len(FS_condition) if isinstance(oxidizer, str) else oxidizer
    
    # 使用 ProcessPoolExecutor 模块进行多进程计算
    futures = []; RES = []; indices = []
    with ProcessPoolExecutor(max_workers = cpu_process) as executor:
        for i, condition in enumerate(FS_condition):
            # save_path = save_dirpath + f'/yaml2FS_Mcondition_{i}th.npz' if save_dirpath != None else None
            future = executor.submit(
                task_wrapper,
                yaml2FS,
                i,
                chem_file = chem_file,
                FS_condition = condition,
                fuel = fuel[i],
                oxidizer = oxidizer[i],
                **kwargs
            )
            futures.append(future)
            indices.append(i)
    wait(futures)
    indices = []
    for future in as_completed(futures):
        RES.append(future.result()[1])
        indices.append(future.result()[0])
    
    RES = np.array(RES); indices = np.array(indices)
    sort_indices = np.argsort(indices)
    RES = RES[sort_indices]
    return RES


def yaml2PSRex_Mcondition(chem_file:str, PSR_condition = None, setup_file:str = None, 
              PSR_T = None, PSR_P = None, PSR_phi = None, fuel: str| list = None, oxidizer: str| list = None, 
              ini_res_time = 1, exp_factor = 2 ** 0.5, save_dirpath:str = None, cpu_process = 1,  **kwargs):
    """
    yaml2PSR 函数的多进程版本，M表示Multi-process; 在本版本中，只接受 PSR_condition 作为多进程计算的对象，
    同时运算是阻塞运算以保证数据的完整性。
    多进程使用 Python 自身的 concurrent.futures.ProcessPoolExecutor 模块，不支持 MPI 并行。
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
    
    fuel = [fuel] * len(PSR_condition) if isinstance(fuel, str) else fuel
    oxidizer = [oxidizer] * len(PSR_condition) if isinstance(oxidizer, str) else oxidizer
    
    # 使用 ProcessPoolExecutor 模块进行多进程计算
    futures = []; RES = []; indices = []
    with ProcessPoolExecutor(max_workers = cpu_process) as executor:
        for i, condition in enumerate(PSR_condition):
            save_path = save_dirpath + f'/yaml2PSRex_Mcondition_{i}th.npz' if save_dirpath != None else None
            future = executor.submit(
                task_wrapper,
                yaml2psr_extinction_time,
                i,
                chem_file = chem_file,
                PSR_condition = condition,
                fuel = fuel[i],
                oxidizer = oxidizer[i],
                ini_res_time = ini_res_time,
                exp_factor = exp_factor,
                save_path = save_path,
                **kwargs
            )
            futures.append(future)
    wait(futures)
    for future in as_completed(futures):
        RES.append(future.result()[1])  # 结果
        indices.append(future.result()[0])  # 索引
    
    RES = np.array(RES); indices = np.array(indices)
    sort_indices = np.argsort(indices)
    RES = RES[sort_indices]
    return RES


def yaml2IDT_sensitivity_Multiprocess(chem_file:str, IDT_condition: np.ndarray = None, mode = None, base_IDT = None, 
                                      IDT_T = None, IDT_P = None, IDT_phi = None, setup_file = None, fuel:str = None, oxidizer:str = None,
                         delta:float = 1e-3, specific_reactions: list = None, multiprocess = None,
                         save_path = None, result_abs = False, need_baseIDT = False, idt_defined_species = None, cut_time = 1, **kwargs):
    """
    详细参见 yaml2psr_sensitivity 函数的说明
    multiprocess: int, default None
    """

    np.set_printoptions(precision=2, suppress=True)
    multiprocess = os.cpu_count() - 1 if multiprocess is None else multiprocess
    print(f'use {multiprocess} process to calculate')
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
    IDT_sensitivity = {}; futures = []; reaction_list = []; 
    # 计算 base_IDT:(不推荐使用)
    if base_IDT is None:
        base_IDT = yaml2idt_Mcondition(
            chem_file, IDT_condition = IDT_condition, 
            fuel = fuel, oxidizer = oxidizer, mode = mode,
            save_dirpath = None, cpu_process = multiprocess
        )
    # 单值下快速运算
    if IDT_condition.ndim == 1:
        with ProcessPoolExecutor(max_workers = multiprocess) as executor:
            for m in range(gas.n_reactions):
                futures.append(
                    executor.submit(
                    solve_IDT_for_yaml2IDT_sensitivity_Multiprocess,
                    chem_file, IDT_condition, delta, m, fuel, oxidizer, 
                    mode = mode, idt_defined_species = idt_defined_species, cut_time = cut_time, **kwargs
                    )
                )
        wait(futures); IDT = []; m_s = []
        for future in as_completed(futures):
            IDT.append(future.result()[0])
            m_s.append(future.result()[1])
        m_s_sort = np.argsort(m_s)
        IDT = np.array(IDT)[m_s_sort]
        Total_IDT.append(IDT)
        Total_IDT = np.array(Total_IDT)
        print(f'Total_IDT is {Total_IDT}')
        for equation, tmp_IDT in zip(reaction_list, Total_IDT):
            if equation in IDT_sensitivity:
                # 求两者的最大值作为敏感度
                IDT_sensitivity[equation] = np.maximum(IDT_sensitivity[equation], (tmp_IDT - base_IDT) / (delta)).tolist()
            else:
                IDT_sensitivity.update(
                    {equation: ((tmp_IDT - base_IDT) / (delta)).tolist()}
                )     

    else:
        fuel = [fuel] * len(IDT_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(IDT_condition) if not isinstance(oxidizer, list) else oxidizer
        mode = [mode] * len(IDT_condition) if not isinstance(mode, list) else mode
        cut_time = cut_time * np.ones(len(IDT_condition)) if not isinstance(cut_time, Iterable) else cut_time
        Total_IDT = []
        for condition, tmp_fuel, tmp_oxidizer, tmp_mode, tmp_cut_time in zip(IDT_condition, fuel, oxidizer, mode, cut_time):
            futures = []
            with ProcessPoolExecutor(max_workers = multiprocess) as executor:
                for m in range(gas.n_reactions):
                    equation = gas.reaction(m).equation
                    if equation not in specific_reactions: continue
                    reaction_list.append(equation)
                
                    futures.append(
                        executor.submit(
                        solve_IDT_for_yaml2IDT_sensitivity_Multiprocess,
                        chem_file, condition, delta, m, tmp_fuel, tmp_oxidizer,
                        mode = tmp_mode, idt_defined_species = idt_defined_species, cut_time = tmp_cut_time,**kwargs
                        )
                    )
            wait(futures); IDT = []; m_s = []
            for future in as_completed(futures):
                IDT.append(future.result()[0])
                m_s.append(future.result()[1])
            m_s_sort = np.argsort(m_s)
            IDT = np.array(IDT)[m_s_sort]
            Total_IDT.append(IDT)
        Total_IDT = np.array(Total_IDT) # shape: (len(IDT_condition), gas.n_reactions)
        Total_IDT = np.transpose(Total_IDT, (1, 0)) # shape: (gas.n_reactions, len(IDT_condition))
        print(f'Total_IDT is {Total_IDT}')
        for equation, tmp_IDT in zip(reaction_list, Total_IDT):
            if equation in IDT_sensitivity:
                # 求两者的最大值作为敏感度
                IDT_sensitivity[equation] = np.maximum(IDT_sensitivity[equation], (tmp_IDT - base_IDT) / (delta)).tolist()
            else:
                IDT_sensitivity.update(
                    {equation: ((tmp_IDT - base_IDT) / (delta)).tolist()}
                )     
        print(f'The current reaction is {equation}, the sensitivity is {IDT_sensitivity[equation]}')
    if result_abs:
        for k, v in IDT_sensitivity.items():
            IDT_sensitivity[k] = np.abs(v).tolist()
    if not save_path is None:
        # 使用 jSON 保存
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(IDT_sensitivity, ensure_ascii=False, indent=4, separators=(',', ':')))

    if need_baseIDT: return IDT_sensitivity, base_IDT
    return IDT_sensitivity


def yaml2psr_sensitivity_Multiprocess(chem_file:str, RES_TIME_LIST:list, PSR_condition: np.ndarray = None, 
                         PSR_T = None, PSR_P = None, PSR_phi = None,
                         setup_file = None, fuel:str = None, oxidizer:str = None,
                         delta:float = 1e-3, specific_reactions: list = None, multiprocess = None,
                         save_path = None, result_abs = False, need_basepsr = False, **kwargs):
    """
    详细参见 yaml2psr_sensitivity 函数的说明
    multiprocess: int, default None
    """
    warnings.warn(f'Not ready to use this function for some alignment problems, please use yaml2FS_sensitivity instead')
    np.set_printoptions(precision=2, suppress=True)
    multiprocess = os.cpu_count() - 1 if multiprocess is None else multiprocess
    print(f'use {multiprocess} process to calculate')
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
    PSR_sensitivity = {}; futures = []; reaction_list = []; 
    # 计算 base_psr
    basepsr = yaml2psr(
        chem_file, PSR_condition, RES_TIME_LIST, 
        fuel = fuel, oxidizer = oxidizer,
        error_tol = 0
    )
    # 单值下快速运算
    if PSR_condition.ndim == 1:
        assert np.array(RES_TIME_LIST).ndim == 1, '请设置 RES_TIME_LIST 为一维数组'
        phi, T, P = PSR_condition
        with ProcessPoolExecutor(max_workers = multiprocess) as executor:
            for m in range(gas.n_reactions):
                gas = ct.Solution(chem_file)
                equation = gas.reaction(m).equation
                if equation not in specific_reactions: continue
                gas.set_multiplier(1.0)  # reset all multipliers
                gas.set_multiplier(1 + delta, m)  # perturb reaction m
                reaction_list.append(equation)
                futures.append(
                    executor.submit(
                    solve_psr, gas, RES_TIME_LIST, T, P, phi, fuel, oxidizer, error_tol = 0.
                    )
                )
        wait(futures)
        PSR = np.array([future.result() for future in futures])
        PSR_sensitivity.update(
                    {equation: ((psr - basepsr) / (delta)).tolist()} for equation, psr in zip(reaction_list, PSR)
                )

    else:
        fuel = [fuel] * len(PSR_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(PSR_condition) if not isinstance(oxidizer, list) else oxidizer
        with ProcessPoolExecutor(max_workers = multiprocess) as executor:
            for m in range(gas.n_reactions):
                gas = ct.Solution(chem_file)
                equation = gas.reaction(m).equation
                if equation not in specific_reactions: continue
                gas.set_multiplier(1.0)  # reset all multipliers
                gas.set_multiplier(1 + delta, m)  # perturb reaction m
                reaction_list.append(equation)
                for tmp_res_time, condition, tmp_fuel, tmp_oxidizer in zip(RES_TIME_LIST, PSR_condition, fuel, oxidizer):
                    phi, T, P = condition
                    futures.append(
                        executor.submit(
                        solve_psr, gas, tmp_res_time, T, P, phi, tmp_fuel, tmp_oxidizer, error_tol = 0.
                        )
                    )
        wait(futures)
        PSR = np.array([future.result() for future in as_completed(futures)])
        PSR = PSR.reshape(len(PSR_condition), len(RES_TIME_LIST))
        for equation, tmp_PSR in zip(reaction_list, PSR):
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


def yaml2LFS_sensitivity_Multiprocess(chem_file:str, LFS_condition: np.ndarray = None, base_LFS = None, 
                                      LFS_T = None, LFS_P = None, LFS_phi = None, setup_file = None, fuel:str = None, oxidizer:str = None,
                         delta:float = 1e-3, specific_reactions: list = None, multiprocess = None,
                         save_path = None, result_abs = False, need_baseLFS = False, **kwargs):
    """
    详细参见 yaml2psr_sensitivity 函数的说明
    multiprocess: int, default None
    """
    warnings.warn(f'Not ready to use this function for some alignment problems, please use yaml2FS_sensitivity instead')
    np.set_printoptions(precision=2, suppress=True)
    multiprocess = os.cpu_count() - 1 if multiprocess is None else multiprocess
    print(f'use {multiprocess} process to calculate')
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
    LFS_sensitivity = {}; futures = []; reaction_list = []; 
    # 计算 base_LFS:(不推荐使用)
    if base_LFS is None:
        base_LFS = yaml2FS_Mcondition(
            chem_file, LFS_condition, 
            fuel = fuel, oxidizer = oxidizer,
            save_dirpath = None, cpu_process = multiprocess
        )
    # 单值下快速运算
    if LFS_condition.ndim == 1:
        with ProcessPoolExecutor(max_workers = multiprocess) as executor:
            for m in range(gas.n_reactions):
                futures.append(
                    executor.submit(
                    solve_FS_for_yaml2LFS_sensitivity_Multiprocess,
                    chem_file, LFS_condition, delta, m, fuel, oxidizer, **kwargs
                    )
                )
        wait(futures)
        LFS = []
        for future in as_completed(futures):
            try:
                LFS.append(future.result())
            except:
                print(traceback.format_exc())
        LFS_sensitivity.update(
                    {equation: ((lfs - base_LFS) / (delta)).tolist()} for equation, lfs in zip(reaction_list, LFS)
                )

    else:
        fuel = [fuel] * len(LFS_condition) if not isinstance(fuel, list) else fuel
        oxidizer = [oxidizer] * len(LFS_condition) if not isinstance(oxidizer, list) else oxidizer
        with ProcessPoolExecutor(max_workers = multiprocess) as executor:
            for m in range(gas.n_reactions):
                equation = gas.reaction(m).equation
                if equation not in specific_reactions: continue
                reaction_list.append(equation)
                for condition, tmp_fuel, tmp_oxidizer in zip(LFS_condition, fuel, oxidizer):
                    futures.append(
                        executor.submit(
                        solve_FS_for_yaml2LFS_sensitivity_Multiprocess,
                        chem_file, condition, delta, m, tmp_fuel, tmp_oxidizer, **kwargs
                        )
                    )
        wait(futures); LFS = []
        for future in as_completed(futures):
            try:
                LFS.append(future.result())
            except:
                print(traceback.format_exc())
        LFS = np.array(LFS)
        for equation, tmp_LFS in zip(reaction_list, LFS):
            LFS_sensitivity.update(
                {equation: ((tmp_LFS - base_LFS) / (delta)).tolist()}
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
       
        
def solve_IDT_for_yaml2IDT_sensitivity_Multiprocess(chem_file, condition, delta, m, fuel, oxidizer, mode, idt_defined_species, cut_time = 1, **kwargs):
    phi, T, P = condition
    gas = ct.Solution(chem_file)
    gas.TP = T, P; gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.set_multiplier(1.0)  # reset all multipliers
    gas.set_multiplier(1.0 + delta, m)  # perturb reaction m
    idt = solve_idt(gas, mode = mode, idt_defined_species = idt_defined_species, cut_time = cut_time, **kwargs)[0]
    assert not isinstance(idt, Iterable), f'IDT must be a single value, but got {idt}'
    return idt, m
 
def solve_FS_for_yaml2LFS_sensitivity_Multiprocess(chem_file, condition, delta, equation, fuel, oxidizer, **kwargs):
    phi, T, P = condition
    gas = ct.Solution(chem_file)
    gas.TP = T, P; gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.set_multiplier(1.0)  # reset all multipliers
    gas.set_multiplier(1 + delta, equation)  # perturb reaction m
    return solve_flame_speed(gas, **kwargs), equation