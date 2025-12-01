# -*- coding:utf-8 -*-
import shutil
import cantera as ct
import numpy as np
import sys
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker


sys.path.append('..')
from utils.cantera_utils import *
from utils.setting_utils import *
from utils.yamlfiles_utils import *
from utils.DeePMR_base_network import *

from collections.abc import Generator, Iterable
from typing import Union

def checkout_iter(*elements)  -> Union[Generator, Iterable]:
    """
    Turn the element (or a tuple of elements) to the generator;
    You are recommended to use the generator by
    a0, = checkout_iter(a0)
    when encounter single element.
    """
    if not isinstance(elements[0], Iterable):
        return ((element,) for element in elements)
    else:
        return elements

# 用于筛选 condition 的函数
def classify_array_T_P(arr):
    """
    获取温度和压强的唯一组合
    """
    unique_conditions = np.unique(arr[:, 1:], axis=0)
    
    # 创建一个字典来存储分类结果
    classified_indices = {tuple(cond): [] for cond in unique_conditions}
    
    # 遍历数组，将索引添加到相应的分类中
    for idx, row in enumerate(arr):
        condition = tuple(row[1:])
        classified_indices[condition].append(idx)
    
    # 将分类结果转换为列表
    classified_index = [classified_indices[tuple(cond)] for cond in unique_conditions]
    
    return classified_index

def classify_array_phi_P(arr):
    """
    获取当量比和压强的唯一组合
    """
    unique_conditions = np.unique(arr[:, [0, 2]], axis=0)
    
    # 创建一个字典来存储分类结果
    classified_indices = {tuple(cond): [] for cond in unique_conditions}
    
    # 遍历数组，将索引添加到相应的分类中
    for idx, row in enumerate(arr):
        condition = tuple(row[[0, 2]])
        classified_indices[condition].append(idx)
    
    # 将分类结果转换为列表
    classified_index = [classified_indices[tuple(cond)] for cond in unique_conditions]
    
    return classified_index



def compare_nn_train(true_data:np.ndarray,
                    *data_arrays: np.ndarray, 
                    **kwargs):
    """
    已经不再维护！
    绘制反问题结果的单个对比图, 输入可以是单个 ndarray 或者 ndarray 的 list. 多图请调用多次此函数, 合理的输入:

    compare_nn_train(true_idt, reduced_idt, labels = ['reduced']) \n
    compare_nn_train(true_idt, reduced_idt, final_idt, labels = ['reduced', 'final']) \n

    Notice:
        function expects log scale of IDT input \n
        the order of the points layer is ordered by the input order
    params:
        true_data: the true value of asamples; which is used as the x axis
        data_arrays: a sequence of data arrays which contains cantera data, reduced data and initial data
        kwargs:
            flatten: default False; if True, the figure will rotate 45 degree so that the baseline will be horizental
            title: the title of the figure; should choose in [IDT, PSR, mole] etc
            labels: the label of data_arrays, len must be the same; defualt None
            colors: the same as labels; control the color of points
            markers: the same as labels; control the marker of points
            save_path: the entire path to save the figure; default None means not save
            need_abs: 默认是 False; 如果是 True, 则最后会展示经过绝对值的结果
        
    return:
        None, but save 1 figures
    """
    compare_pic_name = kwargs.get('save_path', './analysis/compare_nn_train/compare.png')
    mkdirplus(os.path.dirname(compare_pic_name))
    
    labels = kwargs.get('labels', [None] * len(data_arrays))
    colors = kwargs.get('colors', [None] * len(data_arrays))
    markers = kwargs.get('markers', [None] * len(data_arrays))
    title = kwargs.get('title', 'IDT')
    zorders = np.arange(len(data_arrays)) * 0.1 + 1; zorders = zorders[::-1]
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(6 ,6), dpi = 600)
    if not kwargs.get('flatten', False):
        # 获得上下界
        x_lim_left = np.amin(data_arrays); x_lim_r = np.amax(data_arrays); 
        axes.set_xlim(left = x_lim_left - 0.5, right = x_lim_r + 0.5)
        axes.set_ylim(ymin = x_lim_left - 0.5, ymax = x_lim_r + 0.5)
        axes.plot([x_lim_left, x_lim_r], [x_lim_left, x_lim_r], 'k--', lw = 0.5)
        for data_array, label, color, marker, zorder in zip(data_arrays, labels, colors, markers, zorders):
            if marker in ['o','circle']:
                axes.scatter(true_data, data_array, edgecolors = color, facecolors = 'w', s = 15, marker = marker, label = label, zorder = zorder)
            else:    
                axes.scatter(true_data, data_array, color = color, s = 15, marker = marker, label = label, zorder = zorder)
        axes.legend(loc='upper left', fontsize = 12, frameon=False)
        axes.set_xlabel(r'True $\log s$'); axes.set_ylabel(r'Cantera/Reduced $\log s$'); axes.set_title(title)
        plt.xticks(fontsize=12); plt.yticks(fontsize=12) # 子图坐标字号
    else:
        # 获得上下界
        x_lim_left = np.amin(np.array(data_arrays) - true_data); x_lim_r = np.amax(np.array(data_arrays) - true_data); 
        axes.axhline(y = 0, ls = '--', lw = 0.5, color = 'black')
        for data_array, label, color, marker, zorder in zip(data_arrays, labels, colors, markers, zorders):
            if kwargs.get('need_abs', False):
                tmp_data = np.abs(data_array - true_data)
            else:
                tmp_data = data_array - true_data
            if marker in ['o','circle']:
                axes.scatter(np.arange(len(data_array)), tmp_data, edgecolors = color, facecolors = 'w', s = 15, marker = marker, label = label, zorder = zorder)
            else:
                axes.scatter(np.arange(len(data_array)), tmp_data, c = color, s = 15, marker = marker, label = label, zorder = zorder)
        axes.legend(loc='upper left', fontsize = 12, frameon=False)
        axes.set_xlim(left = -1, right = len(data_array) + 1)
        axes.set_ylim(ymin = x_lim_left - 1, ymax = x_lim_r + 1)
        axes.set_xlabel(r'Working Conditions'); axes.set_ylabel(r'$Cantera - True$'); axes.set_title(title)
    plt.savefig(compare_pic_name)
    plt.close(fig)


def compare_nn_train2(true_data:np.ndarray,
                    *data_arrays: np.ndarray, 
                    **kwargs):
    """
    已经不再维护！
    改版的 compare_nn_train; 将原本的绘图旋转45度来达到分得清工况的目的; \n
    各种参数和一版完全相同
    kwargs:
        (add) abs: 默认是 False; 如果是 True, 则最后会展示经过绝对值的结果
        Tlist    : 输入Tlist用于给图像增加工况温度标识
    """
    # 调整数据维度，最后一个维度放温度 T
    data_arrays = flatten_data_permutation(*data_arrays, **kwargs)
    Tlist = kwargs.get('Tlist', None)

    compare_pic_name = kwargs.get('save_path', './analysis/compare_nn_train/compare.png')
    mkdirplus(os.path.dirname(compare_pic_name))
    need_abs = kwargs.get('abs', False)
    
    labels = kwargs.get('labels', [None] * len(data_arrays))
    colors = kwargs.get('colors', [None] * len(data_arrays))
    markers = kwargs.get('markers', [None] * len(data_arrays))
    title = kwargs.get('title', 'IDT')
    zorders = np.arange(len(data_arrays)) * 0.1 + 1; zorders = zorders[::-1]
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(1.25 * len(Tlist) ,6), dpi = 600)
    # 获得上下界
    x_lim_left = np.amin(np.array(data_arrays) - true_data); x_lim_r = np.amax(np.array(data_arrays) - true_data); 
    count_num = len(data_arrays[0]) // len(Tlist); major_tick_location = np.arange(len(data_arrays[0]) + 1)[::count_num]
    tickerlocation = np.array(major_tick_location + count_num / 2, dtype = np.float16)[:-1]
    tickercontent = [rf"$T_0 = {T}K$" for T in Tlist]

    axes.axhline(y = 0, ls = '--', lw = 0.5, color = 'black')
    for data_array, label, color, marker, zorder in zip(data_arrays, labels, colors, markers, zorders):
        if need_abs:
            tmp_data = np.abs(data_array - true_data)
        else:
            tmp_data = data_array - true_data
        if marker in ['o','circle']:
            axes.scatter(np.arange(len(data_array)), tmp_data, edgecolors = color, facecolors = 'w', s = 15, marker = marker, label = label, zorder = zorder)
        else:
            axes.scatter(np.arange(len(data_array)), tmp_data, c = color, s = 15, marker = marker, label = label, zorder = zorder)
    axes.legend(loc='upper left', fontsize = 12, frameon=False)
    axes.set_xlim(left = -1, right = len(data_array) + 1)
    axes.set_ylim(ymin = x_lim_left - 1, ymax = x_lim_r + 1)
    axes.set_xlabel(r'Working Conditions'); axes.set_ylabel(r'$\log Cantera - \log True$'); axes.set_title(title)
    axes.xaxis.set_major_locator(ticker.FixedLocator(major_tick_location))
    axes.xaxis.set_major_formatter(ticker.NullFormatter())
    axes.xaxis.set_minor_locator(ticker.FixedLocator(tickerlocation))
    axes.xaxis.set_minor_formatter(ticker.FixedFormatter(tickercontent))
    for tick in axes.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    plt.xticks(fontsize=8); plt.yticks(fontsize=8) # 子图坐标字号
    plt.savefig(compare_pic_name)
    plt.close(fig)

    
def compare_nn_train3(true_data:np.ndarray,
                *data_arrays: np.ndarray, 
                xlims: tuple = None,
                **kwargs):
    """
    改版的 compare_nn_train2; 将 compare_nn_train2 的 X - Y 轴对调来达到分得清工况的目的; \n
    各种参数和一版完全相同
    kwargs:
        save_path: 保存的位置
        abs: 默认是 False; 如果是 True, 则最后会展示经过绝对值的结果
        wc: 输入工况，要求为如下格式且与数据点一一对应：
            [
                [phi, T, P]
                    ...
            ]
    """
    # 调整数据维度，最后一个维度放温度 T
    np.set_printoptions(suppress=True)
    wc = kwargs.get('wc', range(len(true_data)))
    save_path = kwargs.get('save_path', './analysis/compare_nn_train/compare.png')
    need_abs = kwargs.get('abs', False)
    
    labels = kwargs.get('labels', [None] * len(data_arrays))
    colors = kwargs.get('colors', [None] * len(data_arrays))
    markers = kwargs.get('markers', [None] * len(data_arrays))
    title = kwargs.get('title', 'IDT')
    zorders = np.arange(len(data_arrays)) * 0.1 + 1; zorders = zorders[::-1]
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize=(6, 0.25 * len(true_data)), dpi = 300)
    # 获得上下界
    x_lim_left = np.amin(np.array(data_arrays) - true_data); x_lim_r = np.amax(np.array(data_arrays) - true_data); 
    
    axes.axvline(x = 0, ls = '--', lw = 0.5, color = 'black')
    for data_array, label, color, marker, zorder in zip(data_arrays, labels, colors, markers, zorders):
        if need_abs:
            tmp_data = np.abs(data_array - true_data)
        else:
            tmp_data = data_array - true_data
        if marker in ['o','circle']:
            axes.scatter(tmp_data, np.arange(len(data_array)), edgecolors = color, facecolors = 'w', s = 15, marker = marker, label = label, zorder = zorder)
        else:
            axes.scatter(tmp_data, np.arange(len(data_array)), c = color, s = 15, marker = marker, label = label, zorder = zorder)
    axes.legend(loc='upper left', fontsize = 12, frameon=False)
    axes.set_ylim(ymin = -1, ymax = len(data_array) + 1)
    if xlims is None:
        axes.set_xlim(left = x_lim_left - 1, right = x_lim_r + 1)
    else:
        axes.set_xlim(left = xlims[0], right = xlims[1])
    axes.set_ylabel(r'Working Conditions'); axes.set_title(title)
    if title == 'IDT':
        axes.set_xlabel(r'$\log Cantera - \log True$')
    else:
        axes.set_xlabel(r'$Cantera - True$')
    
    # 调整 Y 轴坐标
    tickercontent = wc
    axes.yaxis.set_major_locator(ticker.FixedLocator(np.arange(len(data_arrays[0]))))
    axes.yaxis.set_major_formatter(ticker.FixedFormatter(tickercontent))

    plt.xticks(fontsize=8); plt.yticks(fontsize=8) # 子图坐标字号
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def compare_PSR_concentration(
    true_data,
    cantera_data,
    reduced_data,
    PSR_concentration_condition,
    PSR_concentration_species,
    final_data = None,
    save_path = './analysis/compare_nn_PSR_concentration',
    **kwargs
):
    classified_index = classify_array_phi_P(PSR_concentration_condition)
    print(classified_index)
    n_col = np.ceil(len(classified_index) ** 1/2).astype(int).item()
    n_row = np.ceil(len(classified_index) / n_col).astype(int).item()
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row), dpi = 200, sharey = False, squeeze = False)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    colors = ['#011627', "#BD7304", '#8DC0C8']
    interval = len(true_data) // len(PSR_concentration_species)
    for tmp_ax, index in zip(ax.flatten()[:len(classified_index)], classified_index):
        tmp_condition = PSR_concentration_condition[index]
        for k, sp in enumerate(PSR_concentration_species):
            tmp_T = tmp_condition[:, 1]
            tmp_true_data =       true_data[k * interval:(k+1)*interval][index]
            tmp_cantera_data = cantera_data[k * interval:(k+1)*interval][index]
            tmp_reduced_data = reduced_data[k * interval:(k+1)*interval][index]
            if final_data is not None: tmp_final_data = final_data[k * interval:(k+1)*interval][index]
            if final_data is not None: 
                tmp_ax.scatter(tmp_T, tmp_final_data, edgecolors = colors[k], marker = 'o', lw = 2, label = sp + 'DNN', zorder = 4, s = 100, facecolors = 'none')
                tmp_ax.scatter(tmp_T, tmp_true_data, edgecolors = colors[k], marker = '.', lw = 2, label = sp + 'detail', zorder = 5, s = 50)
            else:
                tmp_ax.scatter(tmp_T, tmp_true_data, edgecolors = colors[k], marker = 'o', lw = 2, label = sp + 'detail', zorder = 5, s = 100, facecolors = 'none')
            tmp_ax.plot(tmp_T, tmp_reduced_data, c = colors[k], ls = '--', lw = 2.5,  label = sp + 'reduced', zorder = 2)
            tmp_ax.plot(tmp_T, tmp_cantera_data, c = colors[k],ls = '-', lw = 2.5,  label = sp + 'optimized', zorder = 2.5)
            
        tmp_ax.set_title(f'φ = {tmp_condition[0, 0]}, P = {tmp_condition[0, 2]} atm')
        tmp_ax.set_xlabel('Temperature (K)')
        tmp_ax.set_ylabel('Concentration (mol/m³)')
        tmp_ax.legend()
        tmp_ax.set_xlim(np.min(tmp_T) - 50, np.max(tmp_T) + 50)
        tmp_ax.tick_params(axis='x', labelsize=16)
        tmp_ax.tick_params(axis='y', labelsize=16)
        # tmp_ax.grid(True, linestyle='--', alpha=0.5)
        tmp_ax.set_xticks(tmp_T)
        tmp_ax.set_xticklabels([str(T) for T in tmp_T], fontsize=16)
            
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    


""" ================================================================================================================= """

""" ================================================================================================================== """

def sns_data_prepare(npdata: np.ndarray, axis_name:list,) -> pd.DataFrame:
    """
    将 npdata 转化为适合 seaborn 绘图的数据结构, 即将每个维度展平并打上正确的tag 
    每个维度的 name 值将由 axis_name 决定

    params:
        npdata: 支持高维数组，因为在构建的时候会将所有维度重新按照 axis name 排列
        axis_name: 要求传入一个与 npdata 相同维度的数组，作为每个维度的值; 
                   其中第一个值应该是数组第一维度；维度的对应关系必须明确
    return:
        df: 满足条件的 dataframe
    """
    nde = np.ndenumerate(npdata); df = pd.DataFrame(columns = axis_name)
    for  index, value in nde:
        list_index = list(index); list_index.append(value); axis_name.append("value")
        tmp_datum = pd.DataFrame(list_index).T
        tmp_datum = tmp_datum.rename(columns = lambda i: axis_name[i])
        df = pd.concat([df, tmp_datum], ignore_index = True)

    return df


def sample_distribution(asamples:np.ndarray, 
                        idt_data:np.ndarray, 
                        model_forward:Callable, 
                        true_idt:np.ndarray, 
                        reduced_idt:np.ndarray, 
                        **kwargs):
    """
    检查采样的数据点，确认覆盖真实值, 此处有网络输出分布的比较
    params:
        asamples: Asamples_.npy 中生成的数据
        idt_data: asamples 对应的 apart_idt_data
        model_forward: 最终结果的模型中 forward 方法; 如果需要调用 detach().numpy() 则需要在输入前使用
            lambda x: model.forward(x).detach().numpy() 的方式
        true_idt, reduced_idt: 对 IDT 而言需要先 log 标准化; 对 PSR 而言需要先 zscore
        
    kwargs:
        wc_list: 输入对应工况的 label; 默认为 None, 若为 None 则不会在y轴上标注具体工况
    
    """
    np.set_printoptions(suppress = True, precision = 3)
    
    trainNN_data = model_forward(asamples).reshape(-1, len(true_idt))
    gongkuang = np.arange(0, len(true_idt))
    wc_list = kwargs.get("wc_list", None)

    pd_trainDNN = pd.DataFrame(trainNN_data, columns = gongkuang)
    pd_trainDNN['resource'] = 'testdata_DNN'

    pd_sample = pd.DataFrame(idt_data, columns = gongkuang)
    pd_sample['resource'] = 'sample'

    data = pd.concat([pd_sample, pd_trainDNN])
    data = pd.melt(data, var_name =  'WorkingCondition', value_name = 'IDT', id_vars = 'resource')

    fig = plt.figure(figsize = (10, 10 * len(gongkuang) / 27), dpi = 400)
    ax = fig.add_axes([0.15,0.10,0.75,0.75])
    trueplot = ax.scatter(y = gongkuang, x = true_idt, marker = '|', s = 700, c = 'maroon')
    idtnplot = ax.scatter(y = gongkuang, x = reduced_idt,  marker = '|', s = 500, c = 'teal')
    legend = ax.legend(['True', 'Reduced', ],bbox_to_anchor=(1, 1),
                         loc='upper left', borderaxespad=0.)
    sns.violinplot(data = data, y = 'WorkingCondition', x = 'IDT', hue = 'resource', 
                    split = True , ax = ax, bw = 0.1, orient = 'h')

    ax.set_ylabel('Working Condition ($\phi$, $T$, $p$)'); ax.set_xlabel('IDT in Log scale')
    if not wc_list is None:
        ax.set_yticks(gongkuang, wc_list)
    ax.add_artist(legend)
    fig.tight_layout()
    plt.title('Samples and Test Data Distribution')
    plt.savefig(kwargs.get('save_path', './analysis/Sample_Distribution.png'))
    plt.close(fig)


def sample_distribution_IDT(idt_data:np.ndarray, 
                        true_idt:np.ndarray, 
                        reduced_idt:np.ndarray, 
                        marker_idt: np.ndarray = None,
                        IDT_func:Callable = None,
                        asamples = None,
                        wc_list: None | list = None,
                        xlim: tuple = (-2, 2),
                        xlabel: str = 'sample - true',
                        title: str = 'Samples Data Distribution',
                        save_path: str = './sample_distribution.png',
                        **kwargs):
    """
    检查采样的数据点，确认覆盖真实值; 
    params:
        asamples: Asamples_.npy 中生成的数据
        idt_data: asamples 对应的 apart_idt_data
        true_idt, reduced_idt: 对 IDT 而言需要先 log 标准化; 对 PSR 而言需要先 zscore
    
    """
    np.set_printoptions(precision = 3)
    gongkuang = np.arange(0, len(true_idt))
    idt_data = idt_data - true_idt
    pd_sample = pd.DataFrame(idt_data, columns = gongkuang)

    fig, ax = plt.subplots(1, 1, figsize = (10, 10 * len(gongkuang) / 27) ,dpi = 400)
    idtnplot = ax.scatter(y = gongkuang, x = reduced_idt - true_idt,  marker = '|', s = 500, c = 'teal')
    if not marker_idt is None:
        markerplot = ax.scatter(y = gongkuang, x = marker_idt - true_idt, marker = '*', s = 50, c = 'red', zorder = 10) 
    # 绘制一个经过 x = 0 的垂直线
    ax.axvline(x = 0, linestyle = '--', color = 'black', linewidth = 1)
    # legend = ax.legend(['True', 'Reduced', ],bbox_to_anchor=(1, 1),
    #                      loc='upper left', borderaxespad=0.)
    # legend 写在 标题位置
    legend = ax.legend(['Reduced', 'PreviousBest'],bbox_to_anchor=(1, 1),
                            loc='upper right', borderaxespad=0., frameon = False)
    if not IDT_func is None:
        trainNN_data = IDT_func(asamples).reshape(-1, len(true_idt)) - true_idt
        pd_sample['resource'] = 'sample'
        pd_trainDNN = pd.DataFrame(trainNN_data, columns = gongkuang)
        pd_trainDNN['resource'] = 'testdata_DNN'
        data = pd.concat([pd_sample, pd_trainDNN])
        data = pd.melt(data, var_name =  'WorkingCondition', value_name = 'IDT', id_vars = 'resource')
        sns.violinplot(data = data, ax = ax, y = 'WorkingCondition', x = 'IDT', hue = 'resource',
                        split = True , bw = 0.1, orient = 'h')
    else:
        sns.violinplot(data = pd_sample, ax = ax, bw = 1, orient = 'h')
    ax.set_xlim(*xlim)

    ax.set_ylabel('Working Condition ($\phi$, $T$, $p$, residence time)'); 
    ax.set_xlabel(xlabel)
    if not wc_list is None:
        ax.set_yticks(gongkuang, wc_list)
    ax.add_artist(legend)
    fig.tight_layout()
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)


def sample_distribution_PSR(psr_data:np.ndarray, 
                        true_psr:np.ndarray, 
                        reduced_psr:np.ndarray, 
                        PSR_func:Callable = None,
                        asamples = None,
                        trainNN_data: np.ndarray = None,
                        marker_psr = None,
                        PSR_condition = None,
                        RES_TIME_LIST: list = None,
                        xlim: tuple = (-100, 100),
                        xlabel: str = 'sample - true',
                        title: str = 'Samples Data Distribution',
                        save_path: str = './sample_distribution.png',
                        **kwargs):
    """
    检查采样的数据点，确认覆盖真实值; 
    params:
        asamples: Asamples_.npy 中生成的数据
        psr_data: asamples 对应的 apart_psr_data
        true_psr, reduced_psr
        PSR_condition: PSR 的工况，用于标注 y 轴
        RES_TIME_LIST: PSR 的 residence time 列表，用于标注 y 轴
    
    """
    np.set_printoptions(precision = 3)
    gongkuang = np.arange(0, len(true_psr))
    psr_data = psr_data - true_psr
    pd_sample = pd.DataFrame(psr_data, columns = gongkuang)
    wc_list = None if PSR_condition is None else np.hstack([np.repeat(PSR_condition, 3, axis = 0), RES_TIME_LIST.reshape(-1,1)])

    fig, ax = plt.subplots(1, 1, figsize = (10, 10 * len(gongkuang) / 27),dpi = 400)
    # trueplot = ax.scatter(y = gongkuang, x = true_psr, marker = '|', s = 700, c = 'maroon')
    psrnplot = ax.scatter(y = gongkuang, x = reduced_psr - true_psr,  marker = '|', s = 500, c = 'teal')
    if not marker_psr is None:
        markerplot = ax.scatter(y = gongkuang, x = marker_psr - true_psr, marker = '*', s = 50, c = 'red', zorder = 10)
    # 绘制一个经过 x = 0 的垂直线
    ax.axvline(x = 0, linestyle = '--', color = 'black', linewidth = 1)
    # legend = ax.legend(['True', 'Reduced', ],bbox_to_anchor=(1, 1),
    #                      loc='upper left', borderaxespad=0.)
    # legend 写在 标题位置
    legend = ax.legend(['Reduced', 'PreviousBest'],bbox_to_anchor=(1, 1),
                            loc='upper right', borderaxespad=0., frameon = False)
    if not PSR_func is None:
        if trainNN_data is None:
            trainNN_data = PSR_func(asamples).reshape(-1, len(true_psr)) - true_psr 
        pd_sample['resource'] = 'sample'
        pd_trainDNN = pd.DataFrame(trainNN_data, columns = gongkuang)
        pd_trainDNN['resource'] = 'testdata_DNN'
        data = pd.concat([pd_sample, pd_trainDNN])
        data = pd.melt(data, var_name =  'WorkingCondition', value_name = 'PSR', id_vars = 'resource')
        sns.violinplot(data = data, ax = ax, y = 'WorkingCondition', x = 'PSR', hue = 'resource',
                        split = True , bw = 0.1, orient = 'h')
    else:
        sns.violinplot(data = pd_sample, ax = ax, bw = 1, orient = 'h')
    ax.set_xlim(*xlim)

    ax.set_ylabel('Working Condition ($\phi$, $T$, $p$, residence time)'); 
    ax.set_xlabel(xlabel)
    if not wc_list is None:
        ax.set_yticks(gongkuang, wc_list)
    ax.add_artist(legend)
    fig.tight_layout()
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)


def sample_distribution_PSRex(psrex_data:np.ndarray, 
                        true_psrex:np.ndarray, 
                        reduced_psrex:np.ndarray, 
                        marker_psrex = None,
                        PSRex_func:Callable = None,
                        asamples = None,
                        PSRex_condition = None,
                        xlim: tuple = (-1.5, 1.5),
                        xlabel: str = 'sample - true',
                        title: str = 'Samples Data Distribution',
                        save_path: str = './sample_distribution.png',
                        **kwargs):
    """
    检查采样的数据点，确认覆盖真实值; 
    params:
        asamples: Asamples_.npy 中生成的数据
        psrex_data: asamples 对应的 apart_psrex_data
        true_psrex, reduced_psrex
        PSRex_condition: PSRex 的工况，用于标注 y 轴
        RES_TIME_LIST: PSRex 的 residence time 列表，用于标注 y 轴
    
    """
    np.set_printoptions(precision = 3)
    gongkuang = np.arange(0, len(true_psrex))
    psrex_data = psrex_data - true_psrex
    pd_sample = pd.DataFrame(psrex_data, columns = gongkuang)
    wc_list = PSRex_condition

    fig, ax = plt.subplots(1, 1, figsize = (10, 10 * len(gongkuang) / 27),dpi = 400)
    # trueplot = ax.scatter(y = gongkuang, x = true_psrex, marker = '|', s = 700, c = 'maroon')
    ax.scatter(y = gongkuang, x = reduced_psrex - true_psrex,  marker = '|', s = 500, c = 'teal')
    if not marker_psrex is None:
        markerplot = ax.scatter(y = gongkuang, x = marker_psrex - true_psrex, marker = '*', s = 50, c = 'red', zorder = 10)
    # 绘制一个经过 x = 0 的垂直线
    ax.axvline(x = 0, linestyle = '--', color = 'black', linewidth = 1)
    # legend = ax.legend(['True', 'Reduced', ],bbox_to_anchor=(1, 1),
    #                      loc='upper left', borderaxespad=0.)
    # legend 写在 标题位置
    legend = ax.legend(['Reduced', 'PreviousBest'],bbox_to_anchor=(1, 1),
                            loc='upper right', borderaxespad=0., frameon = False)
    if not PSRex_func is None:
        trainNN_data = PSRex_func(asamples).reshape(-1, len(true_psrex)) - true_psrex
        pd_sample['resource'] = 'sample'
        pd_trainDNN = pd.DataFrame(trainNN_data, columns = gongkuang)
        pd_trainDNN['resource'] = 'testdata_DNN'
        data = pd.concat([pd_sample, pd_trainDNN])
        data = pd.melt(data, var_name =  'WorkingCondition', value_name = 'PSRex', id_vars = 'resource')
        sns.violinplot(data = data, ax = ax, y = 'WorkingCondition', x = 'PSRex', hue = 'resource',
                        split = True , bw = 0.1, orient = 'h')
    else:
        sns.violinplot(data = pd_sample, ax = ax, bw = 1, orient = 'h')
    ax.set_xlim(*xlim)

    ax.set_ylabel('Working Condition ($\phi$, $T$, $p$, residence time)'); 
    ax.set_xlabel(xlabel)
    if not wc_list is None:
        ax.set_yticks(gongkuang, wc_list)
    ax.add_artist(legend)
    fig.tight_layout()
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)


def SA4Net(net1, A0, reduced_chem:str, alpha = 0.25, save_path = None):
    """
    Sensitivity Analysis of Net
    net1: 最后获得的网络；请将 device 设置为 cpu
    A0: 灵敏度分析开始的 A 值，一般设为简化机理的 A 值
    reduced_chem: 简化机理的位置
    alpha: 灵敏度分析 A 值的扰动大小
    """
    def diff_order_1(data):
        # 在灵敏度分析需要的一阶差分函数
        return np.mean(np.array([data[:,k+1] - data[:,k] for k in range(data.shape[1]-1)]), axis = 1)
    cantera_sa = []; net1_sa = []; mkdirplus('./tmp')
    for ind in range(len(A0)):
        idt_mat = []; net_idt_mat = []
        sample_list = A0[ind] + np.linspace(-alpha, alpha, 50)
        for i, sam in enumerate(sample_list):
            tmp_A = A0.copy(); tmp_A[ind] = sam; tmp_yaml = f"tmp/tmp_yaml_{ind}_dim{i}.yaml"
            A2yaml(reduced_chem, tmp_yaml, 10**tmp_A)
            tmp_idt, _ = yaml2idt(tmp_yaml, "./settings/setup.yaml")
            idt_mat.append(np.log10(tmp_idt)) # 第一维度是工况，第二维度是 sample

            tmp_A = torch.tensor(tmp_A, dtype = torch.float)
            tmp_idt = net1(tmp_A).detach().numpy()
            net_idt_mat.append(tmp_idt)
        diff_idt_mat = diff_order_1(np.array(idt_mat)); diff_net_mat = diff_order_1(np.array(net_idt_mat))
        cantera_sa.append(diff_idt_mat); net1_sa.append(diff_net_mat)
    cantera_sa = np.array(cantera_sa); net1_sa = np.array(net1_sa) # 第2个维度是 A 的分量
    # cantera_sa = np.transpose((0,2,1)); net1_sa = np.transpose((0,2,1)) # 将第三个维度换为 sample，第二维度是 A 的分量
    cantera_sa = sns_data_prepare(cantera_sa, ["working condition", "dim of A"]); net1_sa = sns_data_prepare(net1_sa,["working condition", "dim of A"])
    cantera_sa['source'] = 'cantera'; net1_sa['source'] = 'net'
    df = pd.concat([cantera_sa, net1_sa], ignore_index = True)
    # df.melt(value_vars = "value")
    print(cantera_sa)

    g = sns.FacetGrid(df,  col = 'working condition')
    g.map_dataframe(sns.barplot, x = 'dim of A', y =  'value', hue =  'source',palette = sns.color_palette("Set2"))
    g.add_legend(bbox_to_anchor = (0.5, 1))
    g.tight_layout(pad = 1.12)
    g.tick_params(axis = 'x', color='w')
    if save_path is None: g.savefig("analysis/sa1.png")
    else: g.savefig(save_path)
    shutil.rmtree('/tmp', ignore_errors = True)

