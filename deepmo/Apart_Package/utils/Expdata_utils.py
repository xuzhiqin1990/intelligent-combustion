# -*- coding:utf-8 -*-
import os, yaml, json, re, logging, time
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


def LoadIDTExpDataWithoutMode(csv_path, fuel_name='NH3', phi_name='phi', T_name='T/K', P_name='P/atm', 
experiment_data_name='igntime(s)', oxidizer_name=['O2',], save_path=None, raw_data=True, condition_prefix='IDT', uncertainty = None, return_df = False  ):
    """
    加载实验数据并将整理好的实验数据重新储存为 csv 文件

    参数:
        csv_path (str): 原始数据的路径
        fuel_name (str or list): 燃料对应的列名
        phi_name (str): 当量比对应的列名
        T_name (str): 温度对应的列名
        P_name (str): 压力对应的列名
        experiment_data_name (str): 实验数据对应的列名
        oxidizer_name (list): 氧化剂对应的列名
        save_path (str, optional): 保存文件的路径
        raw_data (bool): 是否为原始数据
        condition_prefix (str): 条件前缀
        uncertainty (str): 不确定度对应的属性; 
            若为 None 则表示不存在不确定度这一列
            若为 absolute 则表示不确定度是绝对数值，若为 percent 表示为相对数值
    返回:
        tuple: IDT_conditions, fuel, oxidizers, experiment_data
    """
    df = pd.read_csv(csv_path, sep=",")
    if raw_data and all('condition' not in col for col in df.columns):
        # 读取每一条数据的 oxidizer
        oxidizers = []; IDT_conditions = []; experiment_data = df[experiment_data_name].to_numpy()
        fuel = []
        for k in range(df.shape[0]):
            if isinstance(fuel_name, list):
                fuel.append(",".join([f"{tmp_fuel_name}: {df.loc[k, tmp_fuel_name]}" for tmp_fuel_name in fuel_name]))
            else:
                fuel.append(f'{fuel_name}: {df.loc[k, fuel_name]}')
            oxidizers.append(",".join([f"{tmp_oxidizer_name}: {df.loc[k, tmp_oxidizer_name]}" for tmp_oxidizer_name in oxidizer_name]))
            IDT_conditions.append([df.loc[k, phi_name], df.loc[k, T_name], df.loc[k, P_name]])
        if uncertainty is not None:
            # assert 'uncertainty' in df.columns or 'uncertainty_absolute' in df.columns or 'uncertainty_percent' in df.columns, \
            #     f"uncertainty: {uncertainty} is not in df.columns"
            if uncertainty == 'absolute' or 'uncertainty_absolute' in df.columns:
                uncertainty_col = 'uncertainty_absolute' if 'uncertainty_absolute' in df.columns else 'uncertainty'
                uncertainty_values = df[uncertainty_col].to_numpy()
                uncertainty_values = uncertainty_values / experiment_data  # 将不确定度转换为相对值
            elif uncertainty == 'percent' or 'uncertainty_percent' in df.columns:
                uncertainty_col = 'uncertainty_percent' if 'uncertainty_percent' in df.columns else 'uncertainty'
                uncertainty_values = df[uncertainty_col].to_numpy()
            else:
                uncertainty_values = np.ones_like(experiment_data)
                UserWarning(f"uncertainty: {uncertainty} is not supported, set uncertainty_values to zeros")
        else:
            uncertainty_values = np.ones_like(experiment_data)
        print(f"uncertainty_values={uncertainty_values}; absolute={'uncertainty_absolute' in df.columns}; percent={'uncertainty_percent' in df.columns}, the column name is {df.columns}")
        dataframe = {f'{condition_prefix}_condition': IDT_conditions}
        dataframe.update(
            fuel = fuel, oxidizers = oxidizers, experiment_data = experiment_data, uncertainty_values = uncertainty_values
        )
        new_df = pd.DataFrame(dataframe, index = None)
        if save_path is not None: new_df.to_csv(save_path) 
        if return_df:
            return_tuple = new_df
        else:
            return_tuple = (np.array(IDT_conditions), fuel, oxidizers, experiment_data)
            if uncertainty is not None:
                return_tuple += (uncertainty_values,)
        return return_tuple
    else:
        if return_df: 
            return_tuple = new_df
        else:
            return_tuple = (np.concatenate(df[f'{condition_prefix}_condition'].map(eval), axis = 0).reshape(-1, 3), 
                            df['fuel'].to_list(), df['oxidizers'].to_list(), df['experiment_data'].to_numpy())
            if uncertainty is not None and 'uncertainty_values' in df.columns:
                return_tuple += (df['uncertainty_values'].to_numpy(),)
            else:
                return_tuple += (np.ones_like(df['experiment_data'].to_numpy()),)
                
        return return_tuple


def LoadIDTExpData(csv_path, fuel_name=['NH3',], phi_name='phi', T_name='T/K', P_name='P/atm', experiment_data_name='igntime(s)', oxidizer_name=['O2',], save_path=None, raw_data=True, IDT_mode_flag='IDT_mode', uncertainty = None, df=None, return_df = False):
    """
    加载实验数据并将整理好的实验数据重新储存为 csv 文件; 增设读取 IDT_mode 的部分代码；要求 csv 文件具有以下的 columns:
    fuel_name, phi_name 默认: 'phi', T_name 默认: 'T/K', P_name 默认: 'P/atm', experiment_data_name 默认: 'igntime(s)', oxidizer_name 默认: ['O2',]
    IDT_mode_flag 默认: 'IDT mode'

    参数:
        csv_path (str): 原始数据的路径
        fuel_name (str or list): 燃料对应的列名
        phi_name (str): 当量比对应的列名
        T_name (str): 温度对应的列名
        P_name (str): 压力对应的列名
        experiment_data_name (str): 实验数据对应的列名
        oxidizer_name (list): 氧化剂对应的列名
        save_path (str, optional): 保存文件的路径
        raw_data (bool): 是否为原始数据
        IDT_mode_flag (str): IDT mode 对应的列名
        df (DataFrame, optional): 数据框
        uncertainty (str, optional): 不确定度对应的属性; 
            若为 None 则表示不存在不确定度这一列
            若为 absolute 则表示不确定度是绝对数值，若为 percent 表示为相对数值
        return_df (bool): 是否返回 DataFrame

    返回:
        tuple: IDT_conditions, fuel, oxidizers, mode, experiment_data, uncertainty_values
    """
    if csv_path is None:
        if df is None:
            raise ValueError("csv_path 和 df 不能同时为 None")
    else:
        df = pd.read_csv(csv_path, sep=",")
    # 检查其中是否有 IDT_mode_flag 这一列
    if not IDT_mode_flag in df.columns:
        IDT_conditions, fuel, oxidizers, experiment_data, uncertainty_values = LoadIDTExpDataWithoutMode(csv_path, fuel_name, phi_name, T_name, P_name, experiment_data_name, oxidizer_name, save_path, raw_data, uncertainty=uncertainty)
        # IDT_mode 的位置由 1 代替
        IDT_mode = np.ones_like(experiment_data, dtype=int)
        return IDT_conditions, fuel, oxidizers, IDT_mode, experiment_data, uncertainty_values
    else:
        mode_dict = {
            'max dOH/dtEX': 1,
            'max OH': 4,
            'max dP/dt': 3,
            'max dP/dtEX': 6,
            'max (dCH/dt)EX': 'CH_slope_max_intersect',
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
        }
        if raw_data or all('IDT_condition' not in col for col in df.columns):
            # 读取每一条数据的 oxidizer
            oxidizers = []; IDT_conditions = []; experiment_data = df[experiment_data_name].to_numpy()
            fuel = []
            # fuel = df[fuel_name].to_list(); fuel = [fuel_name + f": {tmp_fuel}" for tmp_fuel in fuel]
            IDT_mode = df[IDT_mode_flag].to_list()
            # 按照 mode_dict 的键值对替换 IDT_mode
            # 如果 mode_dict 里面直接是数字，那么直接替换；如果是字符串，那么保留不变
            IDT_mode = [mode_dict[tmp_mode] if tmp_mode in mode_dict else 'NaN' for tmp_mode in IDT_mode]
            
            if 'NaN' in IDT_mode:
                raise ValueError(f"IDT_mode_flag: {IDT_mode_flag} is not in mode_dict")
            for k in range(df.shape[0]):
                if isinstance(fuel_name, list):
                    fuel.append(",".join([f"{tmp_fuel_name}: {df.loc[k, tmp_fuel_name]}" for tmp_fuel_name in fuel_name]))
                else:
                    fuel.append(f'{fuel_name}: {df.loc[k, fuel_name]}')
                oxidizers.append(",".join([f"{tmp_oxidizer_name}: {df.loc[k, tmp_oxidizer_name]}" for tmp_oxidizer_name in oxidizer_name]))
                IDT_conditions.append([df.loc[k, phi_name], df.loc[k, T_name], df.loc[k, P_name]])
            
            if uncertainty is not None:
                # assert 'uncertainty' in df.columns or 'uncertainty_absolute' in df.columns or 'uncertainty_percent' in df.columns, \
                #     f"uncertainty: {uncertainty} is not in df.columns"
                if uncertainty == 'absolute' or 'uncertainty_absolute' in df.columns:
                    uncertainty_col = 'uncertainty_absolute' if 'uncertainty_absolute' in df.columns else 'uncertainty'
                    uncertainty_values = df[uncertainty_col].to_numpy()
                    uncertainty_values = uncertainty_values / experiment_data  # 将不确定度转换为相对值
                elif uncertainty == 'percent' or 'uncertainty_percent' in df.columns:
                    uncertainty_col = 'uncertainty_percent' if 'uncertainty_percent' in df.columns else 'uncertainty'
                    uncertainty_values = df[uncertainty_col].to_numpy()
                else:
                    uncertainty_values = np.ones_like(experiment_data)
                    UserWarning(f"uncertainty: {uncertainty} is not supported, set uncertainty_values to zeros")
            else:
                uncertainty_values = np.ones_like(experiment_data)
            
            dataframe = {f'IDT_condition': IDT_conditions}
            dataframe.update(
                fuel = fuel, oxidizers = oxidizers, IDT_mode = IDT_mode, experiment_data = experiment_data,
                uncertainty_values = uncertainty_values
            )
            new_df = pd.DataFrame(dataframe, index = None)
            if save_path is not None: new_df.to_csv(save_path)
            if return_df:
                return_tuple = new_df
            else:
                return_tuple = (np.array(IDT_conditions), fuel, oxidizers, IDT_mode, experiment_data)
                if uncertainty is not None:
                    return_tuple += (uncertainty_values,)
            return return_tuple
        else:
            if return_df: 
                return_tuple = df
            else:
                # 将 IDT_mode 可以转换为数字的部分转换为数字；字符串的部分保留不变
                IDT_mode = df[IDT_mode_flag].to_list()
                return_tuple = (np.concatenate(df[f'IDT_condition'].map(eval), axis = 0).reshape(-1, 3),
                                df['fuel'].to_list(), df['oxidizers'].to_list(), 
                                [mode_dict[tmp_mode] if tmp_mode in mode_dict else tmp_mode for tmp_mode in IDT_mode], 
                                df['experiment_data'].to_numpy())
                if uncertainty is not None and 'uncertainty_values' in df.columns:
                    return_tuple += (df['uncertainty_values'].to_numpy(),)
                else:
                    return_tuple += (np.ones_like(df['experiment_data'].to_numpy()),)
            return return_tuple


def LoadPSR_concentrationExpData(csv_path, concentration_species, fuel_name='NH3', phi_name='phi', T_name='T/K', P_name='P/atm', oxidizer_name=['O2',], save_path=None, raw_data=True, return_df = False  ):
    """
    加载实验数据并将整理好的实验数据重新储存为 csv 文件

    参数:
        csv_path (str): 原始数据的路径
        fuel_name (str or list): 燃料对应的列名
        phi_name (str): 当量比对应的列名
        T_name (str): 温度对应的列名
        P_name (str): 压力对应的列名
        experiment_data_name (str): 实验数据对应的列名
        oxidizer_name (list): 氧化剂对应的列名
        save_path (str, optional): 保存文件的路径
        raw_data (bool): 是否为原始数据

    返回:
        tuple: PSR_condition, fuel, oxidizers, experiment_data
    """
    df = pd.read_csv(csv_path, sep=",")
    if raw_data:
        # 读取每一条数据的 oxidizer
        oxidizers = []; PSR_condition = []
        experiment_data = {
            species: df[f'{species}_PSR'].to_numpy() for species in concentration_species
        }
        res_time = df['res_time'].to_numpy()
        fuel = []
        for k in range(df.shape[0]):
            if isinstance(fuel_name, list):
                fuel.append(",".join([f"{tmp_fuel_name}: {df.loc[k, tmp_fuel_name]}" for tmp_fuel_name in fuel_name]))
            else:
                fuel.append(f'{fuel_name}: {df.loc[k, fuel_name]}')
            oxidizers.append(",".join([f"{tmp_oxidizer_name}: {df.loc[k, tmp_oxidizer_name]}" for tmp_oxidizer_name in oxidizer_name]))
            PSR_condition.append([df.loc[k, phi_name], df.loc[k, T_name], df.loc[k, P_name]])
       
        dataframe = {f'PSR_condition': PSR_condition}
        dataframe.update(
            fuel = fuel, oxidizers = oxidizers, res_time = res_time, **experiment_data
        )
        new_df = pd.DataFrame(dataframe, index = None)
        if save_path is not None: new_df.to_csv(save_path) 
        if return_df:
            return_tuple = new_df
        else:
            # experiment_data = np.array([experiment_data[species] for species in concentration_species])
            # experiment_data = np.transpose(experiment_data, (1, 0))
            # experiment_data = experiment_data.flatten()
            experiment_data = np.concatenate([experiment_data[species] for species in concentration_species], axis=0)
            return_tuple = (np.array(PSR_condition), fuel, oxidizers, res_time, experiment_data)
        return return_tuple
    else:
        if return_df: 
            return_tuple = new_df
        else:
            return_tuple = (np.concatenate(df[f'PSR_condition'].map(eval), axis = 0).reshape(-1, 3), df['fuel'].to_list(), df['oxidizers'].to_list(), df['res_time'].to_numpy())
            experiment_data = {
                species: df[f'{species}'].to_numpy() for species in concentration_species
            }
            # experiment_data = np.array([experiment_data[species] for species in concentration_species])
            # experiment_data = np.transpose(experiment_data, (1, 0))
            # experiment_data = experiment_data.flatten()
            experiment_data = np.concatenate([experiment_data[species] for species in concentration_species], axis=0)
            return_tuple += (experiment_data,)
        return return_tuple


def return_probe_point_index(csv_path, data_mode_flag='data_mode'):
    """
    返回 probe point 的 index

    参数:
        csv_path (str): 原始数据的路径
        data_mode_flag (str): 数据模式对应的列名

    返回:
        list: probe point 的 index 列表
    """
    df = pd.read_csv(csv_path, sep=",")
    if data_mode_flag not in df.columns:
        UserWarning(f"data_mode_flag: {data_mode_flag} is not in df.columns, return None")
        return None
    return df.loc[df[data_mode_flag] == 2].index.to_list()

        
def LoadIDTExpDataWithProbePoint(csv_path, fuel_name=['NH3',], phi_name='phi', T_name='T/K', P_name='P/atm', 
                            experiment_data_name='igntime(s)', oxidizer_name=['O2',], save_path=None, raw_data=True,
                            IDT_mode_flag='IDT_mode', data_mode_flag='data_mode'):
    """
    加载实验数据并将整理好的实验数据重新储存为 csv 文件; 增设读取 IDT_mode 的部分代码；要求 csv 文件具有以下的 columns:
    fuel_name, phi_name 默认: 'phi', T_name 默认: 'T/K', P_name 默认: 'P/atm', experiment_data_name 默认: 'igntime(s)', oxidizer_name 默认: ['O2',]
    IDT_mode_flag 默认: 'IDT mode'

    参数:
        csv_path (str): 原始数据的路径
        fuel_name (str or list): 燃料对应的列名
        phi_name (str): 当量比对应的列名
        T_name (str): 温度对应的列名
        P_name (str): 压力对应的列名
        experiment_data_name (str): 实验数据对应的列名
        oxidizer_name (list): 氧化剂对应的列名
        save_path (str, optional): 保存文件的路径
        raw_data (bool): 是否为原始数据
        IDT_mode_flag (str): IDT mode 对应的列名
        data_mode_flag (str): 数据模式对应的列名

    返回:
        tuple: (测试数据, 训练数据)
    """
    df = pd.read_csv(csv_path, sep=",")
    test_df = df.loc[df[data_mode_flag] == 1]
    train_df = df.loc[df[data_mode_flag] == 2]
    test_df = test_df.reset_index()
    train_df = train_df.reset_index()
    return (LoadIDTExpData(csv_path = None, fuel_name = fuel_name, phi_name = phi_name, T_name = T_name, P_name = P_name, 
                            experiment_data_name = experiment_data_name, oxidizer_name = oxidizer_name, save_path = None, raw_data = True, 
                            IDT_mode_flag = IDT_mode_flag, df = test_df), 
            LoadIDTExpData(csv_path = None, fuel_name = fuel_name, phi_name = phi_name, T_name = T_name, P_name = P_name, 
                            experiment_data_name = experiment_data_name, oxidizer_name = oxidizer_name, save_path = None, raw_data = True, 
                            IDT_mode_flag = IDT_mode_flag, df = train_df))
    
def concat_simulation_experiment_data(simulation_data, simulation_kwargs, experiment_kwargs, save_path):
    """
    将模拟数据和实验数据合并为一个 DataFrame，并保存为 CSV 文件

    参数:
        simulation_data (np.ndarray): 模拟数据
        simulation_kwargs (dict): 模拟数据的参数
            - condition: (list): 模拟数据的条件列表
            - fuel: (str): 燃料名称
            - oxidizers: (str): 氧化剂名称
        experiment_kwargs (dict): 实验数据的参数，用于 LoadIDTExpData 函数
            - csv_path (str): 实验数据的路径
            - fuel_name (str or list): 燃料对应的列名
            - phi_name (str): 当量比对应的列名
            - T_name (str): 温度对应的列名
            - P_name (str): 压力对应的列名
            - experiment_data_name (str): 实验数据对应的列名
            - oxidizer_name (list): 氧化剂对应的列名
            - condition_prefix (str): 条件前缀
        save_path (str, optional): 保存文件的路径

    返回:
        DataFrame: 合并后的数据
    """
    df = LoadIDTExpDataWithoutMode(return_df = True, **experiment_kwargs)
    IDT_condition = simulation_kwargs['condition']
    fuel = simulation_kwargs['fuel']
    oxidizers = simulation_kwargs['oxidizers']
    
    # df 增加一列 source，表示数据来源
    df['source'] = 'experiment'
    min_uncertainty = df['uncertainty_values'].min() if 'uncertainty_values' in df.columns else 1.0
    simulation_rows = []
    for tmp_idt_data, tmp_idt_condition in zip(simulation_data, IDT_condition):
        formatted_condition = f"[{tmp_idt_condition[0]}, {tmp_idt_condition[1]}, {tmp_idt_condition[2]}]"
        new_row = {
            'IDT_condition': formatted_condition,
            'fuel': fuel,
            'oxidizers': oxidizers,
            'experiment_data': tmp_idt_data,
            'source': 'simulation',
            'uncertainty_values': min_uncertainty  # 假设模拟数据没有不确定度
        }
        simulation_rows.append(new_row)
    simulation_df = pd.DataFrame(simulation_rows)
    df = pd.concat([df, simulation_df], ignore_index=True)
    
    # 删除 df 第一列
    if df.columns[0] == 'Unnamed: 0':
        df = df.iloc[:, 1:]
    df.to_csv(save_path, index=False)
    return df