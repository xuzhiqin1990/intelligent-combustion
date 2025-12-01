# -*- coding:utf-8 -*-
import numpy as np
import sys, warnings
import seaborn as sns
import pandas as pd
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec

# 创建自定义图例
from matplotlib.lines import Line2D

sys.path.append('..')
from utils.cantera_utils import *
from utils.setting_utils import *
from utils.yamlfiles_utils import *
from APART_plot.APART_plot import classify_array_phi_P

def format_settings(
        wspace=0.25, 
        hspace=0.4, 
        left=0.12, 
        right=0.9, 
        bottom=0.15, 
        top=0.95,
        fs=12,
        dpi=300,
        lw=1.5,
        ms=5,
        axlw=1.5,
        major_tick_len=5,
        ):
    '''
        使用方法：
            fig = plt.figure(figsize=(12, 4), dpi=300)
            format_settings()
            grid = plt.GridSpec(2, 2)
            ax1 = fig.add_subplot(grid[0, 0]) # 左上角图
            ax2 = fig.add_subplot(grid[0, 1]) # 右上角图
            ax3 = fig.add_subplot(grid[:, 0]) # 底部空间合并一张图
        注意：
            以上文字和坐标轴粗细适用于figsize长度为12的情形,宽度可调。
            若要调整figsize长度,需要相应调整以上文字和坐标轴粗细。
    '''
    # 设置子图线宽
    plt.rcParams['lines.linewidth'] = lw
    
    # 子图点大小
    plt.rcParams['lines.markersize'] = ms
    
    # 子图间距与位置  w:左右 h:上下
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    # 字体大小
    plt.rcParams['font.size'] = fs
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['axes.titlesize'] = fs
    plt.rcParams['xtick.labelsize'] =fs
    plt.rcParams['ytick.labelsize'] = fs
    plt.rcParams['legend.fontsize'] = fs
    # 子图坐标轴宽度
    plt.rcParams['axes.linewidth'] = axlw
    # 子图坐标轴可见性
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True

    # 子图坐标轴刻度宽度
    plt.rcParams['xtick.major.width'] = axlw
    plt.rcParams['ytick.major.width'] = axlw
    # 子图坐标轴刻度长度
    plt.rcParams['xtick.major.size'] = major_tick_len
    plt.rcParams['ytick.major.size'] = major_tick_len
    # 子图坐标轴刻度长度
    plt.rcParams['xtick.minor.size'] = 0
    plt.rcParams['ytick.minor.size'] = 0
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.major.pad'] = major_tick_len
    plt.rcParams['ytick.major.pad'] = major_tick_len
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.minor.pad'] = 5
    plt.rcParams['ytick.minor.pad'] = 5
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.top'] = False 
    plt.rcParams['ytick.right'] = False
    # 子图坐标轴刻度标签位置
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False
    # 子图坐标轴刻度标签位置
    plt.rcParams['legend.frameon'] = False
    # 子图坐标轴刻度标签位置
    plt.rcParams['figure.dpi'] = dpi
    # 子图坐标轴刻度标签位置
    plt.rcParams['savefig.dpi'] = dpi
    # 默认字体 times new roman
    plt.rcParams['font.family'] = 'Times New Roman'
    # 默认字号 16
    plt.rcParams['font.size'] = 16
    # plt.rcParams["text.usetex"] = True


def get_color_list(n_colors, cmap='viridis', color_min=0.5, color_max=1, invert=False):
    r'''
        从cmap中取出n_colors个颜色
        cmap: 颜色映射
            纯色可选：蓝'Blues', 绿'Greens', 红'Reds', 橙'Oranges', 灰'Greys', 紫'Purples'
            渐变色可选：经典'viridis', 'plasma', 'inferno', 'magma', 红白蓝'seismic'
        color_min: 颜色映射最小值,纯色建议从0.5开始,渐变色建议从0.0开始
        color_max: 颜色映射最大值
        invert: 是否反转颜色,默认从浅到深,invert=True时从深到浅
    '''
    colormap = plt.cm.get_cmap(cmap)
    if invert:
        color_list = [colormap(i) for i in np.linspace(color_max, color_min, n_colors)]
    else:
        color_list = [colormap(i) for i in np.linspace(color_min, color_max, n_colors)]
    return color_list


def get_color_groups(n_group, n_colors, cmap_list=None, color_min=0.5, color_max=1, invert=False):
    r'''
        返回一组颜色,每组颜色有n_colors个
        cmap_list: 颜色映射列表,如果为None,则使用默认的颜色映射
    '''
    if cmap_list is None:
        cmap_list = ['Blues', 'Reds', 'Greens', 'Oranges', 'Greys', 'Purples', 'YlOrBr', 'PuBuGn', 'BuPu']
        
    color_groups = [get_color_list(n_colors, cmap=cmap_list[i], color_min=color_min, color_max=color_max, invert=invert) for i in range(n_group)]
    
    return color_groups



def CompareDRO_IDT_heatmap(detail_data:np.ndarray,
                reduced_data:np.ndarray,
                optimal_data:np.ndarray, 
                range_T:np.ndarray,
                range_P:np.ndarray,
                probe_point:np.ndarray = None,
                save_path:str = None,
                **kwargs):
    """
    Compare the result between detail mechanism, reduced mechanism and optimal reduced mechanism.
    Control the phi to be the same.
    We need two heatmaps to show the difference between optimal_data and detail_data, reduced_data and detail_data. 
    params:
        detail_data: the detail mechanism data; not input the log scale data
        reduced_data: the reduced mechanism data; not input the log scale data
        optimal_data: the optimal reduced mechanism data; not input the log scale data
        range_T & range_P: the IDT condition of the detail mechanism. We need to reshape the result according to them.
        probe_point: the probe point to show on the plot. Format:
            [[T1, P1], [T2, P2], ...]
        Please input the IDT_condition in APART Module.
        save_path: the path to save the figure
        **kwargs: the parameters of heatmap
    return:
        None
    """
    warnings.warn("This function is deprecated, please use CompareDRO_IDT_heatmap_NN instead", DeprecationWarning)
    # from kwargs get title of the plot
    title = kwargs.pop('title', None); cmap = kwargs.pop('cmap', 'RdBu_r')

    optimal_data = np.abs(optimal_data - detail_data) / detail_data; reduced_data = np.abs(reduced_data - detail_data) / detail_data
    optimal_data = optimal_data.reshape(len(range_T), len(range_P)); reduced_data = reduced_data.reshape(len(range_T), len(range_P)); detail_data = detail_data.reshape(len(range_T), len(range_P))
    vmin = min(np.min(optimal_data), np.min(reduced_data)); vmin = np.sign(vmin) * np.abs(vmin) * 1.5
    vmax = max(np.max(optimal_data), np.max(reduced_data)); vmax = np.sign(vmax) * np.abs(vmax) * 1.5

    fig, ax = plt.subplots(1, 2, figsize=(len(range_T), len(range_P)))
    format_settings()
    sns.heatmap(reduced_data, ax = ax[0], vmin =vmin, vmax = vmax, cmap = cmap, cbar = False, **kwargs)
    sns.heatmap(optimal_data, ax = ax[1], vmin =vmin, vmax = vmax, cmap = cmap, cbar = False, **kwargs)

    # set the xticks and yticks of seaborn
    ax[0].set_xticks(np.arange(len(range_P))+0.5)
    ax[0].set_yticks(np.arange(len(range_T))+0.5)
    ax[0].set_xticklabels(range_P)
    ax[0].set_yticklabels(range_T)
    ax[1].set_xticks(np.arange(len(range_P))+0.5)
    ax[1].set_yticks(np.arange(len(range_T))+0.5)
    ax[1].set_xticklabels(range_P)
    ax[1].set_yticklabels(range_T)
    ax[0].set_ylabel('Temperature (K)' + '\n' + 'Original mechanism', fontsize = 16)
    ax[1].set_ylabel('Temperature (K)' + '\n' + 'DeePMO reduced mechanism', fontsize = 16)
    ax[0].set_xlabel('Pressure (atm)', fontsize = 16)
    ax[1].set_xlabel('Pressure (atm)', fontsize = 16)
    # set ticklabels font size
    ax[0].tick_params(axis='x', which='major', labelsize=16)
    ax[1].tick_params(axis='x', which='major', labelsize=16)

    # ticklabels of axis y set to be horizontal
    ax[0].tick_params(axis='y', labelrotation=0, labelsize=12)    
    ax[1].tick_params(axis='y', labelrotation=0, labelsize=12)

    # only show one color bar of the whole plot
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.025, 0.76])
    cb = fig.colorbar(ax[0].collections[0], cax=cbar_ax)
    cb.outline.set_visible(False)

    if probe_point is not None:
        # plot a star marker on the probe point
        for working_condition in probe_point:
            try:
                # find the index of the probe point
                print(np.where(range_T == working_condition[0]))
                index_T = np.where(range_T == working_condition[0])[0][0]
                index_P = np.where(range_P == working_condition[1])[0][0]
                # plot the marker
                ax[0].plot(index_P + 1/2, index_T + 1/2, marker='^', markersize = 25 /2 , color="red", label = 'probe point')
                ax[1].plot(index_P + 1/2, index_T + 1/2, marker='^', markersize = 25 /2 , color="red", label = 'probe point')
                # show the label of marker on the bottom of the axis x; omit redundant legend
                lines, labels = ax[0].get_legend_handles_labels(); tmp_legend = dict(zip(labels, lines))
                lines, labels = tmp_legend.values(), tmp_legend.keys()
                ax[0].legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False )
                ax[1].legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

            except:
                print('The probe point is not in the IDT condition.')

    # save the figure
    fig.suptitle(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def CompareDRO_IDT_lineplot(detail_data:np.ndarray,
                reduced_data:np.ndarray,
                optimal_data:np.ndarray,
                range_T:np.ndarray,
                range_P:np.ndarray,
                range_phi:np.ndarray,
                concat_Pressure:bool = False,
                save_path:str = None,
                probe_point_phi = None,
                probe_point_P = None,
                probe_point_T = None,
                uncertainty: np.ndarray = None,
                **kwargs):
    """
    Compare the result between detail mechanism, reduced mechanism and optimal reduced mechanism.
    Only difference with CompareDRO_heatmap is that we use lineplot to show the result. So the phi could be changed.
    params:
        detail_data: the detail mechanism data
        reduced_data: the reduced mechanism data
        optimal_data: the optimal reduced mechanism data
        range_T & range_P & range_phi: the IDT condition of the detail mechanism. We need to reshape the result according to them.
        probe_point: the probe point to show on the plot. Format:
            [[T1, P1], [T2, P2], ...]
        Please input the IDT_condition in APART Module.
        save_path: the path to save the figure
        **kwargs: the parameters of heatmap
    return:
        None
    """
    format_settings()
    from common_func.common_functions import save_pkl
    IDT = {
        "detail": detail_data.reshape(len(range_phi), len(range_T), len(range_P)),
        "reduced": reduced_data.reshape(len(range_phi), len(range_T), len(range_P)),
        "optimal": optimal_data.reshape(len(range_phi), len(range_T), len(range_P))
    }
    # [print(probe_point_phi, probe_point_P, probe_point_T)]
    # [print(range_phi, range_P, range_T)]
    if not concat_Pressure:
        fig, axes = plt.subplots(len(range_phi), len(range_P), figsize=(4* len(range_P),4* len(range_phi)), dpi = 300, sharex='col', sharey = False, squeeze = False)
        plt.subplots_adjust(hspace=0.5, wspace=0.2)
        # 每个phi + p对应一个子图
        for i, phi in enumerate(range_phi): 
            for j, P in enumerate(range_P): 
                tmp_detail_IDT = IDT['detail'][i,:,j]
                tmp_reduced_IDT = IDT['reduced'][i,:,j]
                tmp_optimal_IDT = IDT['optimal'][i,:,j]         
                axes[i,j].plot(1000 / range_T, tmp_optimal_IDT, label = 'Optimized', c = '#8DC0C8', ls = '-', lw = 3, zorder = 3)
                axes[i,j].plot(1000 / range_T, tmp_reduced_IDT, c = '#293b68', ls = '--', lw = 2,  label = 'Original', zorder = 2)
                if probe_point_P is not None and probe_point_T is not None and probe_point_phi is not None:
                    if np.isin(P, probe_point_P) and np.any((probe_point_T >= np.amin(range_T)) & (probe_point_T <= np.amax(range_T))) and np.isin(phi, probe_point_phi):
                        ## 提取 rangeT 里面 probe_point_T 的 index
                        index = np.array([np.where(range_T == item)[0][0] for item in probe_point_T])
                        
                        print(f'probe index: {index}; range_T: {range_T}; probe_point_T: {probe_point_T}')
                        axes[i,j].scatter(1000 / range_T[index], tmp_detail_IDT[index], marker = '^', edgecolors = 'red', facecolors = 'none', s = 80,  linewidth = 2.5, label = 'Alignment point', zorder = 5)
                        ## 找到 Index 的补集
                        index_complement = np.setdiff1d(np.arange(len(range_T)), index)
                        axes[i,j].scatter(1000 / range_T[index_complement], tmp_detail_IDT[index_complement],  marker = 'o', c = 'red', s = 50,   label = 'Benchmark', zorder = 4)
                    else:
                        axes[i].scatter(1000 / range_T, tmp_detail_IDT,  marker = 'o', c = 'red', s = 50,  label = 'Benchmark', zorder = 4)
                    lengend_ncol = 4
                else:
                    axes[i,j].scatter(1000 / range_T, tmp_detail_IDT,  marker = 'o', c = 'red', s = 50,   label = 'Benchmark', zorder = 4)
                    lengend_ncol = 3

                ylim1 = [0.1*np.min(tmp_optimal_IDT), 10*np.max(tmp_optimal_IDT)]
                ylim2 = [0.1*np.min(tmp_reduced_IDT), 10*np.max(tmp_reduced_IDT)]
                ylim = [np.min([ylim1[0], ylim2[0]]), np.max([ylim1[1], ylim2[1]])]
                xlim = [1000 / max(range_T) - 0.1, 1000 / min(range_T) + 0.1]   
                axes[i,j].set(yscale='log', ylim = ylim, xlim = xlim, )  # 确定子图的范围
                if max(ylim) / 10 < 1e-3:
                    axes[i,j].set_yticks([1e-3, 1e-5]) # 设置刻度
                    axes[i,j].set_yticklabels(['$10^{-3}$', '$10^{-5}$'], fontsize = 16) # 设置刻度标签
                elif 1e-3 < max(ylim) / 10  < 1e-1:
                    axes[i,j].set_yticks([1e-1, 1e-3, 1e-5])
                    axes[i,j].set_yticklabels(['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'], fontsize = 16)
                elif 1e-1 < max(ylim) / 10 :
                    axes[i,j].set_yticks([1e1, 1e-1, 1e-3])
                    axes[i,j].set_yticklabels(['$10^{1}$', '$10^{-1}$', '$10^{-3}$'], fontsize = 16)
                axes[i,j].tick_params(axis='x', labelsize=16)
                if i == 0:
                    axes[i,j].set_xlabel(f"$p = {P}$ atm", fontsize = 16)
            # 将 phi 标注在子图边框外的左上角
            axes[i,0].set_title(f"$\phi = {phi}$", fontsize = 16, loc = 'left')
            # axes[i,0].plot([], [], c = '#E9BD27', ls = '-.', lw = 2,  label = 'Intermediate', zorder = 1)
            lines, labels = axes[0,0].get_legend_handles_labels()
    else:
        # 将 pressure 相同的曲线画在一张图中并注明 pressure
        lengend_ncol = 3
        fig, axes = plt.subplots(1, len(range_phi), figsize=(5 * len(range_phi), 4), dpi = 300, sharex='col', sharey = False, squeeze = True)
        format_settings()
        plt.subplots_adjust(hspace=0.2, wspace=0.2, left = 0.12, bottom = 0.17)
        # 每个phi + p对应一个子图
        for i, phi in enumerate(range_phi):
            for j, P in enumerate(range_P[0:2]):
                tmp_detail_IDT = IDT['detail'][i,:,j]
                tmp_reduced_IDT = IDT['reduced'][i,:,j]
                tmp_optimal_IDT = IDT['optimal'][i,:,j]   
                
                # 获取 tmp_detail_IDT < 0.95 的数据点，> 0.95 的数据舍弃
                index_095 = np.where(tmp_detail_IDT >= 0.95)[0]
                if index_095.size == 0:
                    tmp_range_T = range_T
                    temperature_095 = 0
                else:
                    temperature_095 = range_T[index_095].max()
                    tmp_range_T = range_T[range_T > temperature_095]
                
                axes[i].plot(1000 / tmp_range_T, tmp_optimal_IDT[range_T > temperature_095], label = 'Optimized', c = '#8DC0C8', ls = '-', lw = 3, zorder = 2)
                axes[i].plot(1000 / tmp_range_T, tmp_reduced_IDT[range_T > temperature_095], c = '#293b68', ls = '--', lw = 2,  label = 'Original', zorder = 1)
                # axes[i].scatter(1000 / range_T, tmp_optimal_IDT,  marker = 'o', c = '#8DC0C8', s = 50,  label = 'Optimized', zorder = 3)
                if (probe_point_P is not None) and (probe_point_T is not None) and (probe_point_phi is not None):
                    lengend_ncol = 4
                    if np.isin(P, probe_point_P) and np.any((probe_point_T >= np.amin(range_T)) & (probe_point_T <= np.amax(range_T))) and np.isin(phi, probe_point_phi):
                        ## 提取 rangeT 里面 probe_point_T 的 index
                        index = np.array([np.where(range_T == item)[0][0] for item in probe_point_T])
                        
                        probe_x = 1000 / range_T[index]
                        probe_y = tmp_detail_IDT[index]
                        if index_095.size != 0:
                            probe_y = probe_y[probe_x <= 1000 / temperature_095]
                            probe_x = probe_x[probe_x <= 1000 / temperature_095]
                            
                        
                        axes[i].scatter(probe_x, probe_y, marker = '^', edgecolors = 'red', facecolors = 'none', s = 80,  linewidth = 2.5,  label = 'Alignment point', zorder = 5)
                        ## 找到 Index 的补集
                        index_complement = np.setdiff1d(np.arange(len(range_T)), index)
                        
                        probe_x = 1000 / range_T[index_complement]
                        probe_y = tmp_detail_IDT[index_complement]
                        if index_095.size != 0:
                            probe_y = probe_y[probe_x < 1000 / temperature_095]
                            probe_x = probe_x[probe_x < 1000 / temperature_095]
                        
                        axes[i].scatter(probe_x, probe_y,  marker = 'o', c = 'red', s = 50,   label = 'Benchmark', zorder = 5)
                        print(f'probe index: {index}; range_T: {range_T}; probe_point_T: {probe_point_T}; Index_complement: {index_complement}')
                    else:
                        axes[i].scatter(1000 / tmp_range_T, tmp_detail_IDT[range_T > temperature_095],  marker = 'o', c = 'red', s = 50,  label = 'Benchmark', zorder = 5)
                else:
                    axes[i].scatter(1000 / tmp_range_T, tmp_detail_IDT[range_T > temperature_095],  marker = 'o', c = 'red', s = 50,  label = 'Benchmark', zorder = 5)
                    lengend_ncol = 3
                # 在每条曲线的末端标注 pressure; 要求标注的位置在末端的左下方,不需要箭头且字体加粗
                # optimal_list = IDT["optimal"][i,:,j]
                # annotation_index = np.argmin(np.abs(optimal_list - np.mean(optimal_list)))
                # axes[i].annotate(f"$P = {P}$ atm", xy=(1000 / range_T[0], IDT["optimal"][i,:,j][0]),
                #                 xytext=(1000 / range_T[annotation_index], IDT["optimal"][i,:,j][annotation_index] / 4),
                #                 # arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 3, headlength = 3),
                #                 horizontalalignment='left', verticalalignment='top', fontsize = 16, fontweight = 'bold')
                # 特殊对待
                if j == 0:
                    # 使用轴坐标 (transform=axes[i].transAxes)
                    axes[i].text(
                        1000 / tmp_range_T[-2] + 0.15 * (1000 / tmp_range_T[-1] - 1000 / tmp_range_T[-2]),  # 基于x范围微调
                        1e-1,  # 固定的y值
                        f"$P = {P}$ atm",
                        # transform=axes[i].transAxes,  # 使用轴坐标系统
                        horizontalalignment='left', 
                        verticalalignment='top',
                        fontsize=14, 
                        fontweight='bold'
                    )
                else:
                    # 使用数据坐标 (默认变换)
                    axes[i].text(
                        1000 / tmp_range_T[0] + 0.35 * (1000 / tmp_range_T[-1] - 1000 / tmp_range_T[0]),  # 基于x范围微调
                        1e-4,  # 固定的y值
                        f"$P = {P}$ atm",
                        horizontalalignment='left', 
                        verticalalignment='top',
                        fontsize=14, 
                        fontweight='bold'
                    )
                # if j == 0:
                #     axes[i].text(f"$P = {P}$ atm", xy=(1000 / range_T[0], IDT["optimal"][i,:,j][0]),
                #                 xytext=(0.6, 0.3),
                #                 # arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 3, headlength = 3),
                #                 horizontalalignment='left', verticalalignment='top', fontsize = 14, fontweight = 'bold')

                
                # else:
                #     axes[i].text(f"$P = {P}$ atm", xy=(1000 / range_T[0], IDT["optimal"][i,:,j][0]),
                #                     xytext=(1.15, 1e-4),
                #                 # arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 3, headlength = 3),
                #                 horizontalalignment='left', verticalalignment='top', fontsize = 14, fontweight = 'bold')
                # 在图像右下角标注 phi
                axes[i].text(0.05, 0.98, f"$\phi = {phi}$", transform=axes[i].transAxes, fontsize = 14, fontweight = 'bold', verticalalignment='top')
                axes[i].set_xlabel(r'1000 / T ($\mathrm{K}^{-1}$)', fontsize = 16)
                # axes[i].text(0.05, 0.95, f"$\phi = {phi}$", transform=axes[i].transAxes, fontsize = 14, fontweight = 'bold', verticalalignment='top')
            ylim1 = [0.1*np.min(IDT['optimal'][i,:,:]), 10*np.max(IDT['optimal'][i,:,:])]
            ylim2 = [0.1*np.min(IDT['reduced'][i,:,:]), 10*np.max(IDT['reduced'][i,:,:])]
            ylim = [np.min([ylim1[0], ylim2[0]]), np.max([ylim1[1], ylim2[1]])]
            xlim = [1000 / max(range_T) - 0.1, 1000 / min(range_T) + 0.1]   
            axes[i].set(yscale='log', ylim = ylim, xlim = xlim, )
            if max(ylim) / 10 < 1e-3:
                axes[i].set_yticks([1e-3, 1e-5]) # 设置刻度
                axes[i].set_yticklabels(['$10^{-3}$', '$10^{-5}$'], fontsize = 14) # 设置刻度标签
            elif 1e-3 < max(ylim) / 10  < 1e-1:
                axes[i].set_yticks([1e-1, 1e-3, 1e-5])
                axes[i].set_yticklabels(['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'], fontsize = 14)
            elif 1e-1 < max(ylim) / 10 :
                axes[i].set_yticks([1e1, 1e-1, 1e-3, 1e-5])
                axes[i].set_yticklabels(['$10^{1}$', '$10^{-1}$', '$10^{-3}$', '$10^{-5}$'], fontsize = 14)
            # axes[0].plot([], [], c = '#E9BD27', ls = '-.', lw = 2,  label = 'Intermediate', zorder = 1)
            # axes[i].set_ylabel(f"$\phi = {phi}$", loc='center', fontsize = 14)
            lines, labels = axes[0].get_legend_handles_labels()
        # xticklabel 字体大小
        for ax in axes:
            ax.tick_params(axis='x', labelsize=16)
    # fig.supxlabel('1000 K / Temperature', fontsize = 16)
    fig.supylabel('Ignition delay time (s)', fontsize = 16,)
    # 增加一个空的紫色的 -. 曲线为了绘制 label
    
    # legend() 之前去掉重复的 label
    tmp_legend = dict(zip(labels, lines))
    lines, labels = tmp_legend.values(), tmp_legend.keys()
    # show the legend on the top of x figure, and let it to be flattened
    fig.legend(lines, labels, loc='lower center', ncol = lengend_ncol, borderaxespad=0.5, bbox_to_anchor=(0.5, 0.98), fontsize = 16, frameon = False,
               columnspacing=1, handlelength=1)
    fig.tight_layout()
    save_pkl((fig, axes), save_path.replace('.png', '.pkl'))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    
def CompareDRO_IDT_lineplot2(
                detail_data:np.ndarray,
                reduced_data:np.ndarray,
                optimal_data:np.ndarray,
                base_data:np.ndarray,
                IDT_condition:np.ndarray,
                benchmark_color = 'red',
                probe_point:np.ndarray = None,
                xH2 = None,
                save_path:str = None,
                sources_dict:str = None,
                smooth_window = None,
                uncertainty: np.ndarray = None,
                pressure_group:list = None,
                **kwargs):
    """
    与上一个函数不同的是直接接受整个 IDT_condition 的输入同时按照温度作为横坐标绘制
    因此请输入的数据中的 phi 和 P 是差不多范围的
    可以输入多组不同来源的实验数据

    Args:
        smooth_window (int): 通过​**kwargs传递的平滑窗口长度（需为奇数）。例如smooth_window=5
        uncertainty (np.ndarray): 可选参数，表示数据的不确定性范围，形状应与 detail_data 相同。用于绘制 error bar
        pressure_group (list): 可选参数，表示不同压力组的列表。用于在图例中标识不同压力组的数据。
    """
    from scipy.signal import savgol_filter  # 新增导入
    from common_func.common_functions import save_pkl
    
    # 从kwargs获取平滑参数
    
        # 数据平滑处理（新增代码段）
    def apply_smoothing(data, window):
        if smooth_window and (len(data) > smooth_window >= 3):
            # 自动调整窗口为奇数
            window = smooth_window if smooth_window % 2 == 1 else smooth_window - 1
            return savgol_filter(data, window_length=window, polyorder=2)
        return data

    # 对需要绘制的数据进行平滑处理
    reduced_data = apply_smoothing(reduced_data, smooth_window)
    optimal_data = apply_smoothing(optimal_data, smooth_window)
    if base_data is not None:
        base_data = apply_smoothing(base_data, smooth_window)

    # 以下为原始绘图代码（仅修改数据源为平滑后的数据）
    lengend_ncol = 3
    fig, axes = plt.subplots(1, 1, figsize=(6, 3.2), dpi = 300, sharex='col', sharey = False, squeeze = True)
    format_settings()
    plt.subplots_adjust(hspace=0.2, wspace=0.2, left = 0.12, bottom = 0.17)
    if pressure_group is None:
        range_T = IDT_condition[:, 1]
        
        axes.plot(1000 / range_T, optimal_data,  c = '#8DC0C8', ls = '-', lw = 2,  label = 'Optimized', zorder = 1.5)
        axes.plot(1000 / range_T, reduced_data, c = '#293b68', ls = '--', lw = 2, label = 'Original', zorder = 1)
        
        if base_data is not None:
            axes.plot(1000 / range_T, base_data, label = 'Base', c = '#017301', ls = '-', lw = 2, zorder = 3)
        # for ind, tmp_IDT_condition in enumerate(IDT_condition):
        #     tmp_detail_IDT = detail_data[ind]
        #     tmp_reduced_IDT = reduced_data[ind]
        #     tmp_optimal_IDT = optimal_data[ind] 

        if probe_point is not None:
            ## 提取 rangeT 里面 probe_point_T 的 index
            index = np.array(probe_point)
            axes.scatter(1000 / range_T[index], detail_data[index],  marker = '^', edgecolors = benchmark_color, facecolors = 'none', s = 80,  linewidth = 1.5,  label = 'Alignment point', zorder = 3)
            ## 找到 Index 的补集
            index_complement = np.setdiff1d(np.arange(len(range_T)), index)
            # axes.plot(1000 / range_T[index_complement], optimal_data[index_complement], c = 'red', ls = '-', lw = 2,  label = 'Optimized', zorder = 3)
            axes.scatter(1000 / range_T[index_complement], detail_data[index_complement], label = 'Benchmark', marker = 'o', c = benchmark_color, s = 50, zorder = 2)
            lengend_ncol = 4
            print(f'probe index: {probe_point}; increment: {index_complement}')
        else:
            axes.scatter(1000 / range_T, detail_data, label = 'Benchmark', marker = 'o', c = benchmark_color, s = 50, zorder = 3, lw = 2.5, )
            lengend_ncol = 3
            
        # 如果有误差棒数据，单独添加误差棒（不包含散点）
        # if uncertainty is not None:
        #     axes.errorbar(1000 / range_T, detail_data, yerr=uncertainty, fmt='none', ecolor=benchmark_color, 
        #                  capsize=5, capthick=1, elinewidth=1, zorder=2)
    else:
        assert probe_point is None, "暂不支持在不同压力组中使用探针点"
        assert len(pressure_group) < 3, "pressure_group暂不支持超过2组"
        print(f'Pressure group: {pressure_group}')
        for i, indices in enumerate(pressure_group):
            tmp_range_T = IDT_condition[indices, 1]
            tmp_range_P = IDT_condition[indices, 2]
            tmp_optimal_data = optimal_data[indices]
            tmp_reduced_data = reduced_data[indices]
            tmp_detail_data = detail_data[indices]
            axes.plot(1000 / tmp_range_T, tmp_optimal_data,  c = '#8DC0C8', ls = '-', lw = 2,  label = 'Optimized', zorder = 1.5)
            axes.plot(1000 / tmp_range_T, tmp_reduced_data, c = '#293b68', ls = '-', lw = 2, label = 'Original', zorder = 1)
            axes.scatter(1000 / tmp_range_T, tmp_detail_data, label = 'Benchmark', marker = 'o', c = benchmark_color, s = 50, zorder = 3, lw = 2.5, )
            if base_data is not None:
                axes.plot(1000 / tmp_range_T, base_data[indices], label = 'Base', c = '#017301', ls = '-', lw = 2, zorder = 3)
            
            if i == 0:
                text_position = (
                    1000 / tmp_range_T[-2] + 0.15 * (1000 / tmp_range_T[-1] - 1000 / tmp_range_T[-2]),  # 基于x范围微调
                    1e-1  # 固定的y值
                )
            elif i == 1:
                text_position = (
                    1000 / tmp_range_T[0] + 0.35 * (1000 / tmp_range_T[-1] - 1000 / tmp_range_T[0]),  # 基于x范围微调
                    1e-4  # 固定的y值
                )
            axes.text(
                text_position[0], text_position[1],
                f"$P = {np.mean(tmp_range_P):.1f}$ atm",
                horizontalalignment='left', 
                verticalalignment='top',
                fontsize=14, 
                fontweight='bold'
            )
    
    label_text = f'{sources_dict}\n' if sources_dict is not None else ''
    
    print(sources_dict)
    P_min = np.min(IDT_condition[:, 2]); P_max = np.max(IDT_condition[:, 2])
    phi_min = np.min(IDT_condition[:, 0]); phi_max = np.max(IDT_condition[:, 0])
    P = np.mean(IDT_condition[:, 2]); phi = np.mean(IDT_condition[:, 0])
    if phi_max - phi_min <= 0.5:
        label_text += f"$\phi = {phi:.1f}$\n"
    else:
        label_text += f"$\phi \in [{phi_min:.1f}, {phi_max:.1f}]$\n"
    if P_max - P_min <= 1:
        label_text += f"$P = {P:.1f}$ " + r"$\bf{atm}$"
    else:
        label_text += f"$P\in [{P_min:.1f}, {P_max:.1f}]$ " + r"$\bf{atm}$"
    if xH2 is not None:
        label_text += "\n" + f"{xH2}% " + r"$\bf{H_2}$"  # 注意换行符位置


    axes.text(0.05, 1 - 0.05, label_text, 
            #   f"$P = {P:.1f}$ atm\n$\phi = {phi:.1f}$",
              transform=axes.transAxes, horizontalalignment='left', verticalalignment='top', fontsize = 14, fontweight = 'bold')

    # 在图像左上角标注 phi
    # axes.set_title(f"$\phi = {phi}$", fontsize = 14, loc = 'left')
    axes.set_xlabel(r'1000 / T ($\mathrm{K}^{-1}$)', fontsize = 16)
    # axes.text(0.05, 0.95, f"$\phi = {phi}$", transform=axes.transAxes, fontsize = 14, fontweight = 'bold', verticalalignment='top')
    ylim1 = [0.1*np.min(optimal_data), 10*np.max(optimal_data)]
    ylim2 = [0.1*np.min(reduced_data), 10*np.max(reduced_data)]
    ylim = [np.min([ylim1[0], ylim2[0]]), np.max([ylim1[1], ylim2[1]])]
    xlim = [1000 / max(IDT_condition[:, 1]) - 0.02, 1000 / min(IDT_condition[:, 1]) + 0.02]   
    axes.set(yscale='log', ylim = ylim, xlim = xlim, )
    if max(ylim) / 10 < 1e-3:
        axes.set_yticks([1e-3, 1e-5]) # 设置刻度
        axes.set_yticklabels(['$10^{-3}$', '$10^{-5}$'], fontsize = 14) # 设置刻度标签
    elif 1e-3 < max(ylim) / 10  < 1e-2:
        axes.set_yticks([1e-4, 1e-2])
        axes.set_yticklabels(['$10^{-4}$', '$10^{-2}$'], fontsize = 14)
    elif 1e-2 < max(ylim) / 10  < 1e-1:
        axes.set_yticks([1e-1, 1e-3, 1e-5])
        axes.set_yticklabels(['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'], fontsize = 14)
    elif 1e-1 < max(ylim) / 10 :
        axes.set_yticks([1e1, 1e-1, 1e-3])
        axes.set_yticklabels(['$10^{1}$', '$10^{-1}$', '$10^{-3}$',], fontsize = 14)

    # axes.set_ylabel(f"$\phi = {phi}$", loc='center', fontsize = 14)
    lines, labels = axes.get_legend_handles_labels()
    # xticklabel 字体大小
    axes.tick_params(axis='x', labelsize=16)
    # fig.supxlabel('1000 K / Temperature', fontsize = 16)
    fig.supylabel('Ignition delay time (s)', fontsize = 16)

    # legend() 之前去掉重复的 label
    tmp_legend = dict(zip(labels, lines))
    lines, labels = tmp_legend.values(), tmp_legend.keys()
    # show the legend on the top of x figure, and let it to be flattened
    fig.legend(lines, labels, loc='lower center', ncol = lengend_ncol + 1, borderaxespad=0., bbox_to_anchor=(0.5, 1), fontsize = 14, frameon = False,
               columnspacing=0.5, handlelength=1)
    # fig.tight_layout(h_pad=1.2)
    # 设置绘图边框为默认值
    for spine in axes.spines.values():
        spine.set_linewidth(1)  # 设置边框线宽为默认值
        spine.set_visible(True)  # 确保边框可见
    save_path = save_path.replace(" ", ",")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    save_pkl((fig, axes), save_path.replace('.png', '.pkl'))
    plt.close(fig)


def CompareDRO_IDT_lineplot2_multiP(
    detail_data: np.ndarray,
    reduced_data: np.ndarray,
    optimal_data: np.ndarray,
    base_data: np.ndarray,
    IDT_condition: np.ndarray,
    benchmark_color='red',
    probe_point: np.ndarray = None,
    xH2=None,
    save_path: str = None,
    sources_dict: str = None,
    smooth_window=None,
    uncertainty: np.ndarray = None,
    **kwargs
):
    """
    支持多压力分组画曲线,风格与原CompareDRO_IDT_lineplot2完全一致。
    """
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # 数据平滑处理
    def apply_smoothing(data, window):
        if window and (len(data) > window >= 3):
            window = window if window % 2 == 1 else window - 1
            return savgol_filter(data, window_length=window, polyorder=2)
        return data

    # 分组：按压力
    group_dict = defaultdict(list)
    for i, cond in enumerate(IDT_condition):
        phi, T, P = cond
        group_dict[P].append(i)
    pressure_list = sorted(group_dict.keys())

    # 画图
    fig, axes = plt.subplots(1, 1, figsize=(6, 3.2), dpi=300)
    format_settings = lambda: None  # 可根据你的原代码替换
    # plt.subplots_adjust(hspace=0.2, wspace=0.2, left=0.12, bottom=0.17)

    color_map = plt.cm.get_cmap('tab10', len(pressure_list))
    line_styles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(1,1)), (0,(5,2)), (0, (7,2))]

    for idx, P in enumerate(pressure_list):
        inds = np.array(group_dict[P])
        T_group = IDT_condition[inds,1]
        sort_idx = np.argsort(T_group)
        inds_sorted = inds[sort_idx]
        color = color_map(idx)
        ls = line_styles[idx % len(line_styles)]
        x = 1000 / IDT_condition[inds_sorted,1]

        # Benchmark
        if probe_point is not None:
            index = np.array(probe_point)
            if np.isin(P, IDT_condition[index, 2]):
                axes.scatter(x[index], detail_data[inds_sorted][index], marker='^', edgecolors=benchmark_color, facecolors='none', s=80, linewidth=1.5, label='Alignment point', zorder=3)
                index_complement = np.setdiff1d(np.arange(len(inds_sorted)), index)
                axes.scatter(x[index_complement], detail_data[inds_sorted][index_complement], label='Benchmark', marker='o', c=benchmark_color, s=80, zorder=2)
            else:
                axes.scatter(x, detail_data[inds_sorted], label='Benchmark', marker='o', c=benchmark_color, s=50, zorder=2)
        else:
            axes.scatter(x, detail_data[inds_sorted], label=f'Benchmark-{P:.1f}atm', marker='o', c=[color], s=50, zorder=2)
        # Reduced
        if reduced_data is not None:
            y_red = apply_smoothing(reduced_data[inds_sorted], smooth_window)
            axes.plot(x, y_red, c=color, ls=ls, lw=2, label=f'Original-{P:.1f}atm', zorder=1)
        # Base
        if base_data is not None:
            y_base = apply_smoothing(base_data[inds_sorted], smooth_window)
            axes.plot(x, y_base, label=f'Base-{P:.1f}atm', c=color, ls=ls, lw=2, zorder=3, alpha=0.6)
        # Optimal
        if optimal_data is not None:
            y_opt = apply_smoothing(optimal_data[inds_sorted], smooth_window)
            axes.plot(x, y_opt,  c=color, ls=ls, lw=2,  label=f'Optimized-{P:.1f}atm', zorder=3, alpha=0.85)
        
        if uncertainty is not None:
            y_uncertainty = uncertainty[inds_sorted]
            axes.errorbar(x, detail_data[inds_sorted], yerr=y_uncertainty, fmt='none', ecolor=benchmark_color, capsize=5, capthick=1, elinewidth=1, zorder=2)
        
    # 标签（保持与原版一致）
    label_text = f'{sources_dict}\n' if sources_dict is not None else ''
    phi_min = np.min(IDT_condition[:, 0])
    phi_max = np.max(IDT_condition[:, 0])
    phi = np.mean(IDT_condition[:, 0])
    if phi_max - phi_min <= 0.5:
        label_text += f"$\phi = {phi:.1f}$\n"
    else:
        label_text += f"$\phi \in [{phi_min:.1f}, {phi_max:.1f}]$\n"
    if xH2 is not None:
        label_text += "\n" + f"{xH2}% " + r"$\bf{H_2}$"
    axes.text(0.05, 1 - 0.05, label_text, 
              transform=axes.transAxes, ha='left', va='top', fontsize=14, fontweight='bold')

    axes.set_xlabel(r'1000 / T ($\mathrm{K}^{-1}$)', fontsize=16)
    axes.set_yscale('log')
    axes.set_ylabel('Ignition delay time (s)', fontsize=16)
    axes.tick_params(axis='x', labelsize=14)
    axes.tick_params(axis='y', labelsize=14)

    # 自动设置ylim/xlim
    all_y = np.concatenate([detail_data] + [d for d in [reduced_data, optimal_data, base_data] if d is not None])
    axes.set_ylim(0.1 * np.min(all_y), 10 * np.max(all_y))
    x_all = 1000 / IDT_condition[:,1]
    axes.set_xlim(np.min(x_all) - 0.02, np.max(x_all) + 0.02)

    # 图例去重
    handles, labels = axes.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    axes.legend(legend_dict.values(), legend_dict.keys(), loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5,1.01))

    for spine in axes.spines.values():
        spine.set_linewidth(1)
        spine.set_visible(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def CompareDRO_PSR_lineplot(
                 detail_psr:np.ndarray,
                reduced_psr:np.ndarray,
                optimal_psr:np.ndarray,
                 detail_res_time:np.ndarray,
                reduced_res_time:np.ndarray,
                optimal_res_time:np.ndarray,                
                PSR_condition: np.ndarray,
                save_path:str = None,
                n_col = None, n_row = None,
                ylim_increment = None,
                middle_result_psr = None,
                figsize = None, 
                if_legend = True,
                xlim_right = None,
                extinction_time_dict = None,
                **kwargs):
    """
    Compare the result between detail mechanism, reduced mechanism and optimal reduced mechanism.
    Only difference with CompareDRO_heatmap is that we use lineplot to show the result. So the phi could be changed.
    params:
        detail_data: the detail mechanism data
        reduced_data: the reduced mechanism data
        optimal_data: the optimal reduced mechanism data
        range_T & range_P & range_phi: the IDT condition of the detail mechanism. We need to reshape the result according to them.
        save_path: the path to save the figure
        **kwargs: the parameters of heatmap
    return:
        None
    """
    from common_func.common_functions import save_pkl
    optim_scatter_numbers = kwargs.pop('optim_scatter_numbers', None)
    print("PSR_condition: ", PSR_condition)
    n_col = 3 if n_col is None else n_col
    n_row = int(np.ceil(len(PSR_condition) / n_col)) if n_row is None else n_row
    figsize = (4.5* n_col, 2.8* n_row) if figsize is None else figsize
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, dpi = 300, sharex = False, sharey = False, squeeze = False)
    format_settings()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, bottom = 0.05, top = 0.95, left = 0.15, right = 0.9)
    # 每个phi + p对应一个子图
    index = 0
    for i in range(n_row): 
        for j in range(n_col):  
            
            tmp_optimal_res_time = np.array(optimal_res_time[index])
            tmp_reduced_res_time = np.array(reduced_res_time[index])
            tmp_detail_res_time =  np.array(detail_res_time[index])
            tmp_optimal_psr =      np.array(optimal_psr[index])
            tmp_reduced_psr =      np.array(reduced_psr[index])
            tmp_detail_psr =       np.array(detail_psr[index])
            
            for spine in axes[i, j].spines.values():
                spine.set_linewidth(2)
                spine.set_visible(True)   
            legend_ncol = 0
            if extinction_time_dict is not None:
                true_extinction_time = extinction_time_dict['true_extinction_time'][index]
                reduced_extinction_time = extinction_time_dict['reduced_extinction_time'][index]
                optimal_extinction_time = extinction_time_dict['optimal_extinction_time'][index]
                # 绘制竖线
                axes[i,j].axvline(x=true_extinction_time, color='orange', linestyle='-', linewidth=2, label='True extinction time')
                axes[i,j].axvline(x=reduced_extinction_time, color='orange', linestyle='--', linewidth=2, label='Original extinction time')
                axes[i,j].axvline(x=optimal_extinction_time, color='green', linestyle='--', linewidth=2, label='Optimal extinction time')
                legend_ncol += 2
                optimal_index = optimal_res_time[index] >= true_extinction_time
                reduced_index = reduced_res_time[index] >= true_extinction_time
                detail_index = detail_res_time[index] >= true_extinction_time
                
                tmp_optimal_psr = tmp_optimal_psr[optimal_index][:-1]
                tmp_reduced_psr = tmp_reduced_psr[reduced_index][:-1]
                tmp_detail_psr =       tmp_detail_psr[detail_index][:-1]
                tmp_optimal_res_time = tmp_optimal_res_time[optimal_index][:-1]
                tmp_reduced_res_time = tmp_reduced_res_time[reduced_index][:-1]
                tmp_detail_res_time =  tmp_detail_res_time[detail_index][:-1]
                    
            axes[i,j].plot(tmp_optimal_res_time, tmp_optimal_psr, label = 'Optimized', c = '#8DC0C8', lw = 3, ls = '-', zorder = 2)
            axes[i,j].plot(tmp_reduced_res_time, tmp_reduced_psr, c = '#293b68', ls = '--', lw = 2,  label = 'Original', zorder = 1)
            if optim_scatter_numbers is not None:
                # 散点相隔 optim_scatter_numbers 个点
                tmp_detail_res_time = tmp_detail_res_time[::optim_scatter_numbers].tolist() + [tmp_detail_res_time[-1]]
                tmp_detail_psr = tmp_detail_psr[::optim_scatter_numbers].tolist() + [tmp_detail_psr[-1]]
                axes[i,j].scatter(tmp_detail_res_time, tmp_detail_psr,  marker = 'o', c = 'red', s = 50,  label = 'Benchmark', zorder = 3)
            else:
                axes[i,j].scatter(tmp_detail_res_time, tmp_detail_psr,  marker = 'o', c = 'red', s = 50,  label = 'Benchmark', zorder = 3)
            if middle_result_psr is not None:
                legend_ncol += 4
                axes[i,j].scatter(reduced_res_time[index][:-1], middle_result_psr[index][:-1], edgecolors = '#E9BD27', marker = 'o', facecolors = 'none', linewidth = 2.5, s = 50,  label = 'Without PSRT', zorder = 3)
            else:
                legend_ncol += 3

                
                
            phi, T, P = PSR_condition[index]
            axes[i,j].text(0.95, 0.02, f"$\phi = {phi:.1f}$" + f"\n$T = {T} $ " + r"$\bf{K}$" + f"\n$p = {P} $ " + r"$\bf{atm}$", 
                           transform=axes[i,j].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')
            xlim_right = 1.2 * max(np.max(detail_res_time[index]), np.max(reduced_res_time[index]), np.max(optimal_res_time[index])) if xlim_right is None else xlim_right
            xlim = [0.8 * min(np.min(detail_res_time[index]), np.min(reduced_res_time[index]), np.min(optimal_res_time[index])), xlim_right]
            
            if ylim_increment is not None:
                ylim = np.mean([np.mean(detail_psr[index]), np.mean(reduced_psr[index]), np.mean(optimal_psr[index])])
                ymax = np.max([np.max(detail_psr[index]), np.max(reduced_psr[index]), np.max(optimal_psr[index])])
                axes[i,j].set(ylim = (ylim - ylim_increment, ymax + 100))
            index += 1
            # ticklabel 字体大小
            axes[i,j].set(xscale='log', xlim = xlim, )  # 确定子图的范围
            axes[i,j].tick_params(axis='x', labelsize=16)
            axes[i,j].tick_params(axis='y', labelsize=16)

            axes[0, j].set_xlabel('Residence time (s)', fontsize = 16)
        axes[i, 0].set_ylabel('Temperature (K)', fontsize = 16, y = 0.5)

    lines, labels = axes[0,0].get_legend_handles_labels()
    # show the legend on the top of x figure, and let it to be flattened
    bbox1 = axes[0, 0].get_position()
    bbox2 = axes[0, 1].get_position()
    # 计算两个子图的中心位置
    center_x = (bbox1.x1 + bbox2.x0) / 2
    print(f'center_x: {center_x}')
    if if_legend:
        fig.legend(lines, labels, loc='center', ncol = legend_ncol, borderaxespad=0., bbox_to_anchor=(center_x, 1), fontsize = 16, frameon = False,
                   columnspacing=1.5, handlelength=1.1)
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # fig.tight_layout()
    save_pkl((fig, axes), save_path.replace('.png', '.pkl'))
    print(f'save_path: {save_path}')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def CompareDRON_PSR_lineplot(
                 detail_psr:np.ndarray,
                reduced_psr:np.ndarray,
                optimal_psr:np.ndarray,
                network_psr:np.ndarray,
                detail_res_time:np.ndarray,
                reduced_res_time:np.ndarray,
                optimal_res_time:np.ndarray,                
                PSR_condition: np.ndarray,
                save_path:str = None,
                n_col = None, n_row = None,
                **kwargs):
    """
    Compare the result between detail mechanism, reduced mechanism and optimal reduced mechanism and neural network.
    Only difference with CompareDRO_PSR_lineplot is that we use NN to show the result. 
    params:
        detail_psr: the detail mechanism psr
        reduced_psr: the reduced mechanism psr
        optimal_psr: the optimal reduced mechanism psr
        network_psr: the neural network psr
        range_T & range_P & range_phi: the IDT condition of the detail mechanism. We need to reshape the result according to them.
        save_path: the path to save the figure
        **kwargs: the parameters of heatmap
    return:
        None
    """
    n_col = 3 if n_col is None else n_col
    n_row = int(np.ceil(len(PSR_condition) / n_col)) if n_row is None else n_row
    fig, axes = plt.subplots(n_row, n_col, figsize=(4* n_col,4* n_row), dpi = 300, sharex = False, sharey = False, squeeze = False)
    format_settings()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    # 每个phi + p对应一个子图
    index = 0
    for i in range(n_row): 
        for j in range(n_col):         
            axes[i,j].plot(detail_res_time[index], detail_psr[index], label = 'Benchmark', c = '#011627', ls = '-', lw = 3, zorder = 3)
            axes[i,j].plot(reduced_res_time[index], reduced_psr[index], c = '#293b68', ls = '--', lw = 2,  label = 'Original', zorder = 2)
            axes[i,j].scatter(optimal_res_time[index], optimal_psr[index],  marker = 'o', c = '#8DC0C8', s = 50,   label = 'Optimized', zorder = 2.5)
            axes[i,j].scatter(reduced_res_time[index], network_psr[index],  marker = 'o', edgecolors = '#FF9F1C', facecolors = 'w', s = 70,  label = 'Network', zorder = 1)
            phi, T, P = PSR_condition[index]
            axes[i,j].set_xlabel(f"$\phi = {phi}$, $T = {T}$ K, $p = {P}$ atm", fontsize = 16)
            
            xlim = [0.8 * min(np.min(detail_res_time[index]), np.min(reduced_res_time[index]), np.min(optimal_res_time[index]), np.min(reduced_res_time[index])), 
                    1.2 * max(np.max(detail_res_time[index]), np.max(reduced_res_time[index]), np.max(optimal_res_time[index]), np.max(reduced_res_time[index])), ] 
            axes[i,j].set(xlim = xlim, xscale='log')  # 确定子图的范围
            index += 1
            if index == len(PSR_condition):
                break 

    fig.supxlabel('Residence time (s)', fontsize = 16)
    fig.supylabel('Temperature (K)', fontsize = 16)
    
    lines, labels = axes[0,0].get_legend_handles_labels()
    # show the legend on the top of x figure, and let it to be flattened
    fig.legend(lines, labels, loc='lower center', ncol = 3, borderaxespad=0., bbox_to_anchor=(0.5, 1), fontsize = 16, frameon = False)
        
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


def CompareDRO_LFS_lineplot_Tcombined(
                 detail_lfs:np.ndarray,
                reduced_lfs:np.ndarray,
                optimal_lfs:np.ndarray,
                range_T:np.ndarray,
                range_P:np.ndarray,
                range_phi:np.ndarray,
                save_path:str = None,
                **kwargs):
    """
    Compare the One Dimension Laminar Flame Speed result between detail mechanism, reduced mechanism and optimal reduced mechanism.
    Parameters same with CompareDRO_LFS_lineplot
    """
    from common_func.common_functions import save_pkl
    detail_lfs =   detail_lfs.reshape(len(range_phi), len(range_T), len(range_P))
    reduced_lfs = reduced_lfs.reshape(len(range_phi), len(range_T), len(range_P))
    optimal_lfs = optimal_lfs.reshape(len(range_phi), len(range_T), len(range_P))
    fig, axes = plt.subplots(1, len(range_P), figsize=(8* len(range_P), 4), dpi = 600, sharey = False, squeeze = False)
    format_settings()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    # 每个phi + p对应一个子图
    i = 0
    for j, P in enumerate(range_P):
        for k, T in enumerate(range_T):
            axes[i,j].plot(range_phi, detail_lfs[:, k, j], label = 'Benchmark', c = '#011627', ls = '-', lw = 5, zorder = 3)
            axes[i,j].plot(range_phi, reduced_lfs[:, k, j], c = '#293b68', ls = '--', lw = 4,  label = 'Original', zorder = 2)
            axes[i,j].scatter(range_phi, optimal_lfs[:, k, j],  marker = 'o', c = '#8DC0C8', s = 50,   label = 'Optimized', zorder = 1)
            
            # 在每个线的右上角使用 text 标注温度 T 
            axes[i,j].annotate(f"T = {T} K", xy=(range_phi[-1], optimal_lfs[:, k, j][-1]), 
                               xycoords='data', fontsize = 12, xytext=(1.5, 1.5), textcoords='offset points', color = '#3C486B')

        # ylim = [0.1*np.min(IDT['optimal'][i,:,j]), 10*np.max(IDT['optimal'][i,:,j])]
        # xlim = [1000 / max(range_T) - 0.1, 1000 / min(range_T) + 0.1]   
        # axes[i,j].set( ylim = ylim, xlim = xlim, )  # 确定子图的范围
        axes[i,j].set_xlim(np.min(range_phi) - 0.2, np.max(range_phi) + 0.4)
        axes[i,j].set_title(f"$p = {P}$ atm", loc='left')

    fig.supxlabel('Equivalence ratio', fontsize = 16)
    fig.supylabel('Laminar Flame Speed (m/s)', fontsize = 16)

    lines, labels = axes[0,0].get_legend_handles_labels()
    # show the legend on the below of x axis, and let it to be flattened
    fig.legend(lines, labels, loc='lower center', ncol = 3, borderaxespad=0., bbox_to_anchor=(0.5, 1), fontsize = 16, frameon = False)
    fig.tight_layout()
    save_pkl((fig, axes), save_path.replace('.png', '.pkl'))
    plt.savefig(save_path)
    plt.close(fig)


def CompareDRO_LFS_lineplot(
                detail_lfs:np.ndarray,
                reduced_lfs:np.ndarray,
                optimal_lfs:np.ndarray,
                base_lfs:np.ndarray,
                range_T:np.ndarray = None,
                range_P:np.ndarray = None,
                range_phi:np.ndarray = None,
                save_path:str = None,
                relerror:tuple = None,
                FS_condition:np.ndarray = None,
                n_col = None,
                probe_point:np.ndarray = None,
                uncertainty: np.ndarray = None,
                **kwargs):
    """
    和 CompareDRO_LFS_lineplot_Tcombined 不同的是,这里的 T 和 P 是分开的,所以需要分开绘制。
    relerror: tuple, (optim_relative_error, reduced_relative_error)
    20231031: 增加 FS_condition 参数,用于绘制 probe point
    """
    from common_func.common_functions import save_pkl
    if FS_condition is not None:
        ## 提取 FS_condition 中的 phi, T, P; 将detail_lfs, reduced_lfs, optimal_lfs 按照这个顺序排列
        T = np.unique(FS_condition[:, 1]); P = np.unique(FS_condition[:, 2]); phi = FS_condition[:, 0]
        n_col = max(len(P), 3) if n_col is None else n_col
        n_row = int(np.ceil(len(T) * len(P) / n_col))
        print(f'n_row: {n_row}, n_col: {n_col}')
        axes_index_tuple_list = [(i, j) for i in range(n_row) for j in range(n_col)]
        fig, axes = plt.subplots(n_row, n_col, figsize=(4* n_col,2.5* n_row), dpi = 300, squeeze = False, sharex='col', sharey='row')
        format_settings()

        fig.subplots_adjust(hspace=0.2, wspace=0.1)
        for index in range(len(T) * len(P)):
            row_index = index // n_col; col_index = index % n_col
            tmp_T = T[index // len(P)]; tmp_P = P[index % len(P)]
            tmp_index = np.where((FS_condition[:, 1] == tmp_T) & (FS_condition[:, 2] == tmp_P))
            tmp_phi = FS_condition[tmp_index, 0][0]
            tmp_detail_lfs  = 100 * detail_lfs[tmp_index]; 
            tmp_reduced_lfs = 100 * reduced_lfs[tmp_index]; 
            tmp_optimal_lfs = 100 * optimal_lfs[tmp_index]
            
            axes[row_index, col_index].plot(tmp_phi, tmp_reduced_lfs, c = '#293b68', ls = '--', lw = 4,  label = 'Original', zorder = 2)
            if base_lfs is not None:
                tmp_base_lfs = 100 * base_lfs[tmp_index]
                axes[row_index, col_index].plot(tmp_phi, tmp_base_lfs, label = 'Base', c = 'purple', ls = '-', lw = 3, zorder = 4)
            # axes[row_index, col_index].scatter(tmp_phi, tmp_optimal_lfs,  marker = 'o', c = '#8DC0C8', s = 50,  label = 'Optimized', zorder = 3)
            if not probe_point is None:
                probe_point_phi, probe_point_lfs = [], []
                for condition in probe_point:
                    if condition[1] == tmp_T and condition[2] == tmp_P:
                        # axes[row_index, col_index].scatter(condition[0], tmp_optimal_lfs[np.where(tmp_phi == condition[0])], marker = '^', edgecolors = '#0700e6', facecolors = 'none', s = 50,  linewidth = 2.5, label = 'Alignment point', zorder = 6)
                        probe_point_phi.append(condition[0])
                        probe_point_lfs.append(tmp_optimal_lfs[np.where(tmp_phi == condition[0])])
                        
                        # 获取 detail_lfs 中对应的 probe_point_phi 和非对应的 probe_point_phi
                        detailed_with_probe = tmp_detail_lfs[np.where(tmp_phi == condition[0])]
                        detailed_wo_probe = np.delete(tmp_detail_lfs, np.where(tmp_phi == condition[0]))
                        
                        tmp_phi = np.delete(tmp_phi, np.where(tmp_phi == condition[0]))
                        
                print(f'len of probe_point_phi: {len(probe_point_phi), len(probe_point_lfs)}; remained: {len(tmp_phi), len(tmp_optimal_lfs)}')
                axes[row_index, col_index].scatter(probe_point_phi, detailed_with_probe, marker = '^', edgecolors = '#0700e6', facecolors = 'none', s = 80,  linewidth = 2.5, label = 'Alignment point', zorder = 6)
                axes[row_index, col_index].plot(tmp_phi, detailed_wo_probe, label = 'Benchmark', c = '#011627', ls = '-', lw = 5, zorder = 3)
            else:
                axes[row_index, col_index].plot(tmp_phi, tmp_detail_lfs, label = 'Benchmark', c = '#011627', ls = '-', lw = 5, zorder = 3)
                
            axes[row_index, col_index].scatter(tmp_phi, tmp_optimal_lfs,  marker = 'o', c = '#8DC0C8', s = 80,  label = 'Optimized', zorder = 5)
            
            if uncertainty is not None:
                tmp_uncertainty = uncertainty[tmp_index]
                axes[row_index, col_index].errorbar(tmp_phi, tmp_detail_lfs, yerr=tmp_uncertainty, fmt='none', ecolor='#011627', capsize=5, capthick=1, elinewidth=1, zorder=2)

            axes[row_index, col_index].set_xlim(np.min(tmp_phi) - 0.2, np.max(tmp_phi) + 0.4)
            # 在图像左上角标注 "$T = {tmp_T}$ K, $p = {tmp_P}$ atm"
            axes[row_index, col_index].text(0.05, 0.95, f"$T={tmp_T}$ " + r"$\bf{K}$" + f"\n$P = {tmp_P}$ " + r"$\bf{atm}$", transform=axes[row_index, col_index].transAxes, 
                                            fontsize = 14, fontweight = 'bold', verticalalignment='top')
            axes_index_tuple_list.remove((row_index, col_index))
            # 调整 xticklabel 和 yticklabel 的字体
            axes[row_index, col_index].tick_params(axis='x', labelsize=16)
            axes[row_index, col_index].tick_params(axis='y', labelsize=16)
            # 在每一列最底层的图上标注 xlabel
            if row_index == n_row - 1:
                axes[row_index, col_index].set_xlabel('Equivalence ratio', fontsize = 16)
            # 调整 ylim 增加 5
            ylim = axes[row_index, col_index].get_ylim()
            axes[row_index, col_index].set_ylim(ylim[0], ylim[1] + 5)

            # 绘制所有的边框 spline
            for spine in axes[row_index, col_index].spines.values():
                spine.set_linewidth(2)
                spine.set_visible(True)
                
        fig.supylabel('Laminar Flame Speed (cm/s)', fontsize = 16, verticalalignment='center', y=0.6, x = 0.05)
        # 删除没有绘制图像的子图
        for row_index, col_index in axes_index_tuple_list:
            fig.delaxes(axes[row_index, col_index])
        # 调整 xlim 为所有数据的最大值
        xlim = [np.min(phi) - 0.1, np.max(phi) + 0.1]
        for i in range(n_row):
            for j in range(n_col):
                axes[i,j].set_xlim(xlim)
        
    else:    
        detail_lfs =  100 * detail_lfs.reshape(len(range_phi), len(range_T), len(range_P))
        reduced_lfs = 100 * reduced_lfs.reshape(len(range_phi), len(range_T), len(range_P))
        optimal_lfs = 100 * optimal_lfs.reshape(len(range_phi), len(range_T), len(range_P))
        fig, axes = plt.subplots(len(range_T), len(range_P), figsize=(4* len(range_P), 4* len(range_T)), dpi = 300, sharey = False, squeeze = False)
        format_settings()
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        # 每个phi + p对应一个子图
        for j, P in enumerate(range_P):
            for k, T in enumerate(range_T):
                axes[k,j].plot(range_phi, detail_lfs[:, k, j], label = 'Benchmark', c = '#011627', ls = '-', lw = 5, zorder = 3)
                axes[k,j].plot(range_phi, reduced_lfs[:, k, j], c = '#293b68', ls = '--', lw = 4,  label = 'Original', zorder = 2)
                axes[k,j].scatter(range_phi, optimal_lfs[:, k, j],  marker = 'o', c = '#8DC0C8', s = 80, label = 'Optimized', zorder = 1)
                
                # 在每个线的右上角使用 text 标注温度 T 
                axes[k,j].annotate(f"T = {T} K", xy=(range_phi[-1], optimal_lfs[:, k, j][-1]), 
                                xycoords='data', fontsize = 12, xytext=(1.5, 1.5), textcoords='offset points', color = '#3C486B')

                axes[k,j].set_xlim(np.min(range_phi) - 0.2, np.max(range_phi) + 0.4)
                # 调整 xticklabel 和 yticklabel 的字体
                axes[row_index, col_index].tick_params(axis='x', labelsize=16)
                axes[row_index, col_index].tick_params(axis='y', labelsize=16)
                if j == 0:
                    axes[k,j].set_ylabel(f"$T = {T}$ K", loc='center')
                if not probe_point is None:
                    for condition in probe_point:
                        if condition[1] == T and condition[2] == P:
                            axes[k,j].scatter(condition[0], optimal_lfs[:, k, j][np.where(range_phi == condition[0])], marker = '^', edgecolors = '#0700e6', facecolors = 'none', s = 50,  linewidth = 2.5,  label = 'Alignment point', zorder = 6)
            axes[0,j].set_title(f"$p = {P}$ atm", loc='left')

    
        if relerror is not None:
            fig.supxlabel(f'Equivalence ratio \n' + f"DeePMO Relative Error: {relerror[0]:.2f}%, Original Relative Error: {relerror[1]:.2f}%", fontsize = 16, verticalalignment='center')
        else:
            fig.supxlabel('Equivalence ratio', fontsize = 16, verticalalignment='center')
        fig.supylabel('Laminar Flame Speed (cm/s)', fontsize = 16, verticalalignment='center')

    lines, labels = axes[0,0].get_legend_handles_labels()
    # delete the repeat legend
    tmp_legend = dict(zip(labels, lines))
    lines, labels = tmp_legend.values(), tmp_legend.keys()
    # show the legend on the below of x axis, and let it to be flattened
    fig.legend(lines, labels, loc='lower center', ncol = len(labels),
               borderaxespad=0., bbox_to_anchor=(0.51, 0.95), fontsize = 16, frameon = False, 
               columnspacing=0.8, handlelength=0.6)
    # fig.subplots_adjust(left = 0.12, hspace=0.17, wspace=0.2)
    fig.tight_layout()
    # savefig 内调宽上部分留白
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.15, dpi = 300)
    save_pkl((fig, axes), save_path.replace('.png', '.pkl'))
    plt.close(fig)


def CompareDRO_LFS_lineplot2(
                detail_lfs:np.ndarray,
                reduced_lfs:np.ndarray,
                optimal_lfs:np.ndarray,
                base_lfs:np.ndarray,
                FS_condition:np.ndarray = None,
                index_group:np.ndarray = None,
                group_labels:np.ndarray = None,
                save_path:str = None,
                probe_point:np.ndarray = None,
                concat_phi:bool = False,
                xH2=None,
                xN2=None,
                benchmark_color = 'black',
                smooth_window=None,
                uncertainty: np.ndarray = None,
                **kwargs):
    """
    20211125 增加
    为了使用更灵活的分组绘图方法,同时将所有的数据绘制到一个大图中
    """
    from common_func.common_functions import save_pkl
    from scipy.signal import savgol_filter  # 新增导入

    # 数据平滑处理函数（新增）
    def apply_smoothing(data, window):
        if window and (len(data) > window >= 3):
            window = window if window % 2 == 1 else window - 1
            return savgol_filter(data, window_length=window, polyorder=2)
        return data
    
    ## 提取 FS_condition 中的 phi, T, P; 将detail_lfs, reduced_lfs, optimal_lfs 按照这个顺序排列
    # n_col 是温度的个数
    T_list = np.unique(FS_condition[:, 1])
    n_col = len(T_list) 
    format_settings()
    if not concat_phi:
        fig, axes = plt.subplots(1, n_col, figsize=(8, 6), dpi = 300, squeeze = False)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for tmp_index_group, tmp_group in enumerate(index_group):
            tmp_FS_condition = FS_condition[tmp_group]
            tmp_phi = tmp_FS_condition[:, 0]
            tmp_detail_lfs = 100 * detail_lfs[tmp_group]
            tmp_reduced_raw = reduced_lfs[tmp_group]
            tmp_reduced_smoothed = apply_smoothing(tmp_reduced_raw, smooth_window)
            tmp_reduced_lfs = 100 * tmp_reduced_smoothed  # 应用平滑
            tmp_optimal_raw = optimal_lfs[tmp_group]
            tmp_optimal_smoothed = apply_smoothing(tmp_optimal_raw, smooth_window)
            tmp_optimal_lfs = 100 * tmp_optimal_smoothed  # 应用平滑
            tmp_T = tmp_FS_condition[0, 1]; tmp_P = tmp_FS_condition[0, 2]
            col_index = np.where(T_list == tmp_T)[0][0]
            #axes[0, col_index].plot(tmp_phi, tmp_detail_lfs, label = 'Benchmark', c = benchmark_color, ls = '-', lw = 3, zorder = 3)
    
            axes[0, col_index].plot(tmp_phi, tmp_reduced_lfs, c = '#293b68', ls = '--', lw = 2,  label = 'Original', zorder = 2)
            if base_lfs is not None:
                tmp_base_raw = base_lfs[tmp_group]
                tmp_base_smoothed = apply_smoothing(tmp_base_raw, smooth_window)
                tmp_base_lfs = 100 * tmp_base_smoothed  # 应用平滑
                axes[0, col_index].plot(tmp_phi, tmp_base_lfs, label = 'Base', c = '#017301', ls = '-', lw = 3, zorder = 4)
                
            # axes[0, col_index].scatter(tmp_phi, tmp_optimal_lfs,  marker = 'o', c = '#8DC0C8', s = 50,  label = 'Optimized',zorder = 3)
            if not probe_point is None:
                for condition in probe_point:
                    if condition[1] == tmp_T and condition[2] == tmp_P:
                        axes[0, col_index].scatter(condition[0], tmp_detail_lfs[np.where(tmp_phi == condition[0])], marker = '^', edgecolors = benchmark_color,
                                                   facecolors = 'none', s = 80,  linewidth = 2.5, label = 'Alignment point', zorder = 6)
                        # 删除 tmp_detail_lfs 中的 target 点
                        tmp_detail_lfs = np.delete(tmp_detail_lfs, np.where(tmp_phi == condition[0]))
                        tmp_phi = np.delete(tmp_phi, np.where(tmp_phi == condition[0]))
            
            axes[0, col_index].scatter(tmp_phi, tmp_detail_lfs,  marker = 'o', c = benchmark_color, s = 50,  label = 'Benchmark',zorder = 3)
            if uncertainty is not None:
                # 绘制 detail_lfs 的不确定度
                tmp_uncertainty = uncertainty[tmp_group]
                # tmp_uncertainty_smoothed = apply_smoothing(tmp_uncertainty, smooth_window)
                axes[0, col_index].errorbar(tmp_phi, tmp_detail_lfs, yerr=tmp_uncertainty, fmt='o', 
                                             ecolor='lightgray', elinewidth=2, capsize=3, label='Uncertainty', zorder=1, alpha=0.5)
            
            #axes[0, col_index].scatter(tmp_phi, tmp_optimal_lfs,  marker = 'o', c = '#8DC0C8', s = 50,  label = 'Optimized',zorder = 3)
            axes[0, col_index].plot(tmp_phi, tmp_optimal_lfs, label = 'Optimized', c = 'red', ls = '-', lw = 3, zorder = 3)

            # 在图像左上角标注 "$T = {tmp_T}$ K, $p = {tmp_P}$ atm"
            if group_labels is not None:
                tmp_group_labels = group_labels[row_index]
            else:
                tmp_group_labels = f"$T={tmp_T}$ "+r"$\bf{K}$"+f"   $P={tmp_P}$ " +r"$\bf{atm}$"
            if xH2 is not None:  # 添加掺氢比
                tmp_group_labels += f"\n$\mathbf{{{xH2}\%\ H_2}}$"
            if xN2 is not None:  # 添加掺氮比
                if xN2 == 'Air':
                    tmp_group_labels += f"\n$\\mathbf{{{100 - 78}\\%O_2\\ {78}\\%N_2}}$(Air)"
                else:
                    tmp_group_labels += f"\n$\\mathbf{{{100 - xN2}\\%O_2\\ {xN2}\\%N_2}}$"
            
            # 将 detail_lfs 最右边(当量比最大)的点的值标注在图像右上角
            axes[0, col_index].annotate(tmp_group_labels, xy=(tmp_phi[-1], tmp_detail_lfs[-1]), 
                                        xycoords='data', fontsize = 12, xytext=(1.5, 1.5), textcoords='offset points', color = '#3C486B')
            # axes[0, col_index].text(0.05, 0.95, tmp_group_labels, transform=axes[0, col_index].transAxes, 
            #                                 fontsize = 14, fontweight = 'bold', horizontalalignment='left', verticalalignment='top')
            # 调整 xticklabel 和 yticklabel 的字体
            axes[0, col_index].tick_params(axis='x', labelsize=16)
            axes[0, col_index].tick_params(axis='y', labelsize=16)
            # 在每一列最底层的图上标注 xlabel
            axes[0, col_index].set_xlabel('Equivalence ratio', fontsize = 16)
            
            # ylim 增加 6
            ylim = axes[row_index, 0].get_ylim(); margin_y_original = 6 / 20 * (ylim[1] - ylim[0])
            axes[row_index, 0].set_ylim(ylim[0], ylim[1] + margin_y_original)
            
        fig.supylabel('Laminar Flame Speed (cm/s)', fontsize = 16, verticalalignment='center', y=0.6)

        lines, labels = axes[0,0].get_legend_handles_labels()
        # delete the repeat legend
        tmp_legend = dict(zip(labels, lines))
        lines, labels = tmp_legend.values(), tmp_legend.keys()
        # show the legend on the below of x axis, and let it to be flattened
        fig.legend(lines, labels, loc='lower center', ncol = len(labels),
                borderaxespad=0., bbox_to_anchor=(0.51, 0.95), fontsize = 14, frameon = False, 
                columnspacing=0.8, handlelength=0.6)
        fig.subplots_adjust(left = 0.12, hspace=0.17, wspace=0.2)
    else:
        n_row = len(index_group)
        # 将所有曲线花在不一样的子图中同时将子图的 x 轴合并
        fig, axes = plt.subplots(n_row, 1, figsize=(6, 2.5 * n_row), dpi = 300, squeeze = False, sharex=True)
        fig.subplots_adjust(hspace=0, wspace=0.1)
        for row_index, tmp_group in enumerate(index_group):
            tmp_FS_condition = FS_condition[tmp_group]
            tmp_phi = tmp_FS_condition[:, 0]
            tmp_detail_lfs = 100 * detail_lfs[tmp_group]
            tmp_reduced_raw = reduced_lfs[tmp_group]
            tmp_reduced_smoothed = apply_smoothing(tmp_reduced_raw, smooth_window)
            tmp_reduced_lfs = 100 * tmp_reduced_smoothed  # 应用平滑
            tmp_optimal_raw = optimal_lfs[tmp_group]
            tmp_optimal_smoothed = apply_smoothing(tmp_optimal_raw, smooth_window)
            tmp_optimal_lfs = 100 * tmp_optimal_smoothed  # 应用平滑
            tmp_T = tmp_FS_condition[0, 1]; tmp_P = tmp_FS_condition[0, 2]
            # col_index = np.where(T_list == tmp_T)[0][0]
            axes[row_index, 0].plot(tmp_phi, tmp_optimal_lfs, label = 'Optimized', c = '#8DC0C8', ls = '-', lw = 3, zorder = 3)
            axes[row_index, 0].plot(tmp_phi, tmp_reduced_lfs, c = '#293b68', ls = '--', lw = 2.5,  label = 'Original', zorder = 2)
            if base_lfs is not None:
                tmp_base_raw = base_lfs[tmp_group]
                tmp_base_smoothed = apply_smoothing(tmp_base_raw, smooth_window)
                tmp_base_lfs = 100 * tmp_base_smoothed  # 应用平滑
                axes[row_index, 0].plot(tmp_phi, tmp_base_lfs, label = 'Base', c = '#017301', ls = '-', lw = 3, zorder = 4)
            
            if not probe_point is None:
                probe_point_phi = []; probe_point_lfs = []
                for condition in probe_point:
                    if condition[1] == tmp_T and condition[2] == tmp_P:
                        probe_point_phi.append(condition[0])
                        probe_point_lfs.append(tmp_detail_lfs[np.where(tmp_phi == condition[0])])
                        
                        # 删除 tmp_detail_lfs 中的 target 点
                        tmp_detail_lfs = np.delete(tmp_detail_lfs, np.where(tmp_phi == condition[0]))  
                        tmp_phi = np.delete(tmp_phi, np.where(tmp_phi == condition[0]))
                        # print(f'condition[0]: {condition[0]}, tmp_phi: {tmp_phi}; len(tmp_phi): {len(tmp_phi)}; len(tmp_detail_lfs): {len(tmp_detail_lfs)}')

                axes[row_index, 0].scatter(probe_point_phi, probe_point_lfs, marker = '^', edgecolors = benchmark_color, facecolors = 'none', s = 80,  linewidth = 2.5, label = 'Alignment point', zorder = 6)
            axes[row_index, 0].scatter(tmp_phi, tmp_detail_lfs, label = 'Benchmark', zorder = 3, s = 50, marker = 'o', c = benchmark_color, lw = 2.5)


            # 在图像左上角标注 "$T = {tmp_T}$ K, $p = {tmp_P}$ atm"
            if group_labels is not None:
                tmp_group_labels = group_labels[row_index]
            else:
                tmp_group_labels = f"$T={tmp_T}$ "+r"$\bf{K}$"+f"   $P={tmp_P}$ " +r"$\bf{atm}$"
            if xH2 is not None:  # 添加掺氢比
                tmp_group_labels += f"\n$\mathbf{{{xH2}\%\ H_2}}$"
            if xN2 is not None:  # 添加掺氮比
                if xN2 == 'Air':
                    tmp_group_labels += f"\n$\\mathbf{{{100 - 78}\\%O_2\\ {78}\\%N_2}}$(Air)"
                else:
                    tmp_group_labels += f"\n$\\mathbf{{{100 - xN2}\\%O_2\\ {xN2}\\%N_2}}$"
                
            # 将 detail_lfs 最右边(当量比最大)的点的值标注在图像右上角
            axes[row_index, 0].text(0.05, 0.95, tmp_group_labels, transform=axes[row_index, 0].transAxes,
                                            fontsize = 14, fontweight = 'bold', horizontalalignment='left', verticalalignment='top')
        
            # 调整 xticklabel 和 yticklabel 的字体
            axes[row_index, 0].tick_params(axis='x', labelsize=16)
            axes[row_index, 0].tick_params(axis='y', labelsize=16)
            # 在每一列最底层的图上标注 xlabel
            if row_index == n_row - 1:
                axes[row_index, 0].set_xlabel('Equivalence ratio', fontsize = 16)
            # ylim 增加 6
            ylim = axes[row_index, 0].get_ylim(); margin_y_original = 6 / 20 * (ylim[1] - ylim[0])
            axes[row_index, 0].set_ylim(ylim[0], ylim[1] + margin_y_original)
        
        fig.supylabel('Laminar Flame Speed (cm/s)', fontsize = 16, verticalalignment='center', y=0.5)
        lines, labels = axes[0,0].get_legend_handles_labels()
        # delete the repeat legend
        tmp_legend = dict(zip(labels, lines))
        lines, labels = tmp_legend.values(), tmp_legend.keys()
        # show the legend on the below of x axis, and let it to be flattened
        fig.legend(lines, labels, loc='lower center', ncol = len(labels),
                borderaxespad=0.1, bbox_to_anchor=(0.51, 0.9), fontsize = 14, frameon = False, 
                columnspacing=0.22, handlelength=1)
        # fig.subplots_adjust(left = 0.12, hspace=0.17, wspace=0.2)
        
    # fig.tight_layout()
    # savefig 内调宽上部分留白
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.15, dpi = 300)
    save_pkl((fig, axes), save_path.replace('.png', '.pkl'))
    plt.close(fig)



def CompareDRO_PSR_concentration_lineplot(
                detail_psr_concentration:np.ndarray,
                reduced_psr_concentration:np.ndarray,
                optimal_psr_concentration:np.ndarray,
                PSR_concentration_condition:np.ndarray,
                save_path:str = None,
                probe_point:np.ndarray = None,
                species = ['CO', 'CO2'],
                **kwargs):
    """
    20250704 增加
    """
    from common_func.common_functions import save_pkl

    colors = ['#011627', "#BD7304", '#8DC0C8']
    classified_index = classify_array_phi_P(PSR_concentration_condition)
    print(classified_index)
    n_col = np.ceil(len(classified_index) ** 1/2).astype(int).item()
    n_row = np.ceil(len(classified_index) / n_col).astype(int).item()
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row), dpi = 200, sharey = False, squeeze = False)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    colors = ['#011627', "#BD7304", '#8DC0C8']
    interval = len(detail_psr_concentration) // len(species)
    for tmp_ax, index in zip(ax.flatten()[:len(classified_index)], classified_index):
        tmp_condition = PSR_concentration_condition[index]
        phi = tmp_condition[0, 0]; P = tmp_condition[0, 2]
        for k, sp in enumerate(species):
            tmp_T = tmp_condition[:, 1]
            tmp_true_data =       detail_psr_concentration[k * interval:(k+1)*interval][index]
            tmp_cantera_data = optimal_psr_concentration[k * interval:(k+1)*interval][index]
            tmp_reduced_data = reduced_psr_concentration[k * interval:(k+1)*interval][index]
            
            if not probe_point is None:
                probe_point_T = []; probe_point_value = []
                for condition in probe_point:
                    if condition[0] == phi and condition[2] == P:
                        for T, value in zip(tmp_T, detail_psr_concentration):
                            if T == condition[1]:
                                probe_point_T.append(T)
                                probe_point_value.append(value[k * interval + index])

                tmp_ax.scatter(probe_point_T, probe_point_value, marker = '^', edgecolors = tmp_color, facecolors = 'none', s = 80,  linewidth = 2.5,  label = 'Alignment point', zorder = 6)
                optimized = np.setdiff1d(tmp_condition[:, 1], probe_point_T)
                tmp_detail_psr_concentration = np.delete(tmp_detail_psr_concentration, np.where(np.isin(tmp_condition[:, 1], probe_point_T)))
                tmp_ax.scatter(optimized, tmp_detail_psr_concentration, label = 'Benchmark', marker = '.',  c = tmp_color, s = 80,  linewidth = 2.5, zorder = 6)
            else:

                tmp_ax.scatter(tmp_T, tmp_true_data, edgecolors = colors[k], marker = 'o', lw = 2, label = sp + 'detail', zorder = 5, s = 100, facecolors = 'none')
            
            tmp_ax.plot(tmp_T, tmp_reduced_data, c = colors[k], ls = '--', lw = 2.5,  label = sp + 'reduced', zorder = 2)
            tmp_ax.plot(tmp_T, tmp_cantera_data, c = colors[k],ls = '-', lw = 2.5,  label = sp + 'optimized', zorder = 2.5)
            
        tmp_ax.text(0.05, 0.95, f"$P={P}$ " +r"$\bf{atm}$" + f" $\phi$ = {phi}", transform=tmp_ax.transAxes,
                        fontsize = 14, fontweight = 'bold', horizontalalignment='left', verticalalignment='top')
      
        tmp_ax.set_xlim(np.min(tmp_condition[:, 1]) - 50, np.max(tmp_condition[:, 1]) + 50)
        current_ylim = tmp_ax.get_ylim()
        # 设置 y 轴范围为当前范围 + 5
        tmp_ax.set_ylim(current_ylim[0], current_ylim[1] + 0.1 * current_ylim[1])
        # 调整 xticklabel 和 yticklabel 的字体
        tmp_ax.tick_params(axis='x', labelsize=16)
        tmp_ax.tick_params(axis='y', labelsize=16)
        tmp_ax.set_ylabel(f"Mole Fraction", loc='center')

    
    legend_elements = []
    for i, species_name in enumerate(species):
        tmp_color = colors[i % len(colors)]
        # 圆点为 benchmark
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                      markeredgecolor=tmp_color, markeredgewidth=2, markersize=8, label=f'{species_name} (Benchmark)'))
        # 实线为 optimized
        legend_elements.append(Line2D([0], [0], color=tmp_color, lw=2.5, label=f'{species_name} (Optimized)'))
        # 虚线为 original
        legend_elements.append(Line2D([0], [0], color=tmp_color, lw=2.5, linestyle='--', label=f'{species_name} (Original)'))
    
    # show the legend on the below of x axis
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               bbox_to_anchor=(0.51, 1.05), fontsize=16, frameon=False, handlelength=1.5)
    # fig.subplots_adjust(left = 0.12, hspace=0.17, wspace=0.2)
    fig.tight_layout()
    # savefig 内调宽上部分留白
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3, dpi = 300)
    save_pkl((fig, ax), save_path.replace('.png', '.pkl'))
    plt.close(fig)
        

def CompareOPTIM_SA(optimal_dict:dict,
                    SA_dict:dict,
                    optim_SA_dict:dict,
                    save_dirpath:str = None,
                    reaction_num = None,
                    figname = "CompareOPTIM_SA",
                    IDT_condition = None,
                    figsize = None,
                    labels = ['Optim', 'reduced_SA', 'optim_SA'],
                    color = ['b', 'r', 'g'],
                    **kwargs):
    """
    将灵敏度最高的几个反应和调整幅度最大的几个反应提取出来,绘制所有反应的优化结果与SA结果的对比图。
    绘制出的图像是一个 barplot, 第一象限内展示调整幅度, 第四象限内展示灵敏度。
    要求在输入前必须就对 optim_dict 中的值排好顺序。
    params:
        optimal_dict: the optimal result of the reduced mechanism. Format:
            {'reaction_name': abs(optimal_value - original_value)}
        SA_dict: the SA result of the reduced mechanism. Format:
            {'reaction_name': SA_value}
        save_path: the path to save the figure
        reaction_num: the number of reactions to plot
        labels: the labels of the barplot
        **kwargs: the parameters of barplot
    return:
        None
    """
    # from kwargs get title of the plot
    title = kwargs.pop('title', None)
    
    sns.set_theme(style="whitegrid")
    reaction_num = len(optimal_dict) if reaction_num is None else reaction_num
    # 将 optimal_dict 按照调整幅度的大小进行排序
    optimal_dict = dict(sorted(optimal_dict.items(), key = lambda item: item[1], reverse = True))
    # 保留前 nums 个反应和最后 nums 个反应
    optimal_dict = dict(list(optimal_dict.items())[:reaction_num] + list(optimal_dict.items())[-reaction_num:])
    # 按照 optimal_dict 的顺序对 SA_dict 进行排序
    SA_dict = {key: SA_dict[key] for key in optimal_dict.keys()}
    optim_SA_dict = {key: optim_SA_dict[key] for key in optimal_dict.keys()}
    
    # plot the grouped barplot of optimal_dict and SA_dict
    figsize = (2 / 5 * reaction_num, 6) if figsize is None else figsize
    fig, ax = plt.subplots(figsize = figsize); ax.grid(False)
    format_settings()
    sa_ax = ax.twiny(); sa_ax.grid(False); sa_ax.invert_xaxis()
    # get the reaction names
    reaction_names = list(optimal_dict.keys()); optimal_values = np.array(list(optimal_dict.values()))
    # get the SA values
    SA_values = np.array(list(SA_dict.values()))
    print("SA_values", SA_values)
    for i, (opt, sa) in enumerate(zip(optimal_values, SA_values)):
        print(f"reaction {i}: {reaction_names[i]}, opt = {opt}, sa = {sa}")
        handle1 = ax.barh(i, opt, height=0.2, color=color[0], zorder = 1)
        handle2 = sa_ax.barh(i+0.2, sa, height=0.2, color=color[1],  zorder = 1)
        handle3 = sa_ax.barh(i-0.2, optim_SA_dict[reaction_names[i]], height=0.2, color=color[2], zorder = 1)
        if opt > 0:
            ax.text(-1, i, reaction_names[i], ha='right', va='center', color='b', fontsize = 8, zorder = 3)
        else:
            ax.text(1, i, reaction_names[i], ha='left', va='center', color='b', fontsize = 8, zorder = 3)
    ax.set_yticks([]); sa_ax.set_yticks([])
    # 设置其他属性
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    sa_ax.spines['right'].set_visible(False)
    sa_ax.spines['top'].set_visible(False)
    sa_ax.spines['left'].set_visible(False)
    # 设置 xlim
    ax.set_xlim(-1.1 * np.amax(np.abs(optimal_values)), 1.1 * np.amax(np.abs(optimal_values)))
    sa_ax.set_xlim(-1.1 * np.amax(np.abs(SA_values)), 1.1 * np.amax(np.abs(SA_values)))
    if IDT_condition is not None:
        phi, T, P = IDT_condition
        ax.set_xlabel('Adjust Range for Reactions \n' + f'T={int(T)}_P={P}_phi={phi}')
    else:
        ax.set_xlabel('Adjust Range for Reactions')
    # set the title
    if title is not None:
        ax.set_title(title)
    # set the legend on the bottom of x axis
    ax.legend([handle1, handle2, handle3], labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon = False)
    # save the figure
    if save_dirpath is not None:
        fig.savefig(save_dirpath + f'/{figname}', bbox_inches='tight')
    # show the figure
    plt.close(fig)


def CompareOPTIM_SA_with_Detail(optimal_dict:dict,
                    SA_dict:dict,
                    optim_SA_dict:dict,
                    detail_dict:dict,
                    save_dirpath:str = None,
                    reaction_num = None,
                    figname = "CompareOPTIM_SA",
                    figsize = None,
                    labels = ['Optim', 'Original_SA', 'Optim_SA', 'Detailed_SA'],
                    color = ['b', 'r', 'g'],
                    **kwargs):
    """
    将灵敏度最高的几个反应和调整幅度最大的几个反应提取出来,绘制所有反应的优化结果与SA结果的对比图。
    绘制出的图像是一个 barplot, 第一象限内展示调整幅度, 第四象限内展示灵敏度。
    要求在输入前必须就对 optim_dict 中的值排好顺序。
    params:
        optimal_dict: the optimal result of the reduced mechanism. Format:
            {'reaction_name': abs(optimal_value - original_value)}
        SA_dict: the SA result of the reduced mechanism. Format:
            {'reaction_name': SA_value}
        save_path: the path to save the figure
        reaction_num: the number of reactions to plot
        labels: the labels of the barplot
        **kwargs: the parameters of barplot
    return:
        None
    """
    # from kwargs get title of the plot
    title = kwargs.pop('title', None)
    
    sns.set_theme(style="whitegrid")
    reaction_num = len(optimal_dict) if reaction_num is None else reaction_num
    # 将 optimal_dict 按照调整幅度的大小进行排序
    optimal_dict = dict(sorted(optimal_dict.items(), key = lambda item: item[1], reverse = True))
    # 保留前 nums 个反应和最后 nums 个反应
    optimal_dict = dict(list(optimal_dict.items())[:reaction_num] + list(optimal_dict.items())[-reaction_num:])
    # 按照 optimal_dict 的顺序对 SA_dict 进行排序
    SA_dict = {key: SA_dict[key] for key in optimal_dict.keys()}
    optim_SA_dict = {key: optim_SA_dict[key] for key in optimal_dict.keys()}
    detail_dict = {key: detail_dict[key] for key in optimal_dict.keys()}
    
    # plot the grouped barplot of optimal_dict and SA_dict
    figsize = (3 / 5 * reaction_num, 8) if figsize is None else figsize
    fig, ax = plt.subplots(figsize = figsize); ax.grid(False)
    format_settings()
    sa_ax = ax.twiny(); sa_ax.grid(False); sa_ax.invert_xaxis()
    # get the reaction names
    reaction_names = list(optimal_dict.keys()); optimal_values = np.array(list(optimal_dict.values()))
    # get the SA values
    SA_values = np.array(list(SA_dict.values()))
    detail_values = np.array(list(detail_dict.values()))
    print("SA_values", SA_values)
    for ind, (opt, sa, detail) in enumerate(zip(optimal_values, SA_values, detail_values)):
        print(f"reaction {ind}: {reaction_names[ind]}, opt = {opt}, sa = {sa}, detail = {detail}")
        i = 2 * ind
        handle1 = ax.barh(i+0.45, opt, height=0.45, color=color[0], zorder = 1, align = 'edge')
        handle2 = sa_ax.barh(i, sa, height=0.45, color=color[1],  zorder = 1, align = 'edge')
        handle3 = sa_ax.barh(i-0.45, optim_SA_dict[reaction_names[ind]], height=0.45, color=color[2], zorder = 1, align = 'edge')
        handle4 = sa_ax.barh(i-0.9, detail, height=0.45, color=color[3], zorder = 1, align = 'edge')
        # 我们想要将 text 紧接着 bar 的边缘,所以需要根据 opt, sa, detail 以及 reaction_names[ind] 的正负来确定 text 的位置
        text_x_location = 1.1 * np.amax(np.abs([opt, sa, detail]))
        if opt > 0:
            ax.text(-text_x_location, i, reaction_names[ind], ha='right', va='center', color='b', fontsize = 12, zorder = 3)
        else:
            ax.text(text_x_location, i, reaction_names[ind], ha='left', va='center', color='b', fontsize = 12, zorder = 3)
    ax.set_yticks([]); sa_ax.set_yticks([])
    # 设置其他属性
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    sa_ax.spines['right'].set_visible(False)
    sa_ax.spines['bottom'].set_visible(False)
    sa_ax.spines['left'].set_visible(False)
    # 设置 xlim
    ax.set_xlim(-1.1 * np.amax(np.abs(optimal_values)), 1.1 * np.amax(np.abs(optimal_values)))
    sa_ax.set_xlim(-1.1 * np.amax(np.abs(SA_values)), 1.1 * np.amax(np.abs(SA_values)))
    if title is not None:
        ax.set_xlabel('Adjust Range for Reactions \n' + title)
    else:
        ax.set_xlabel('Adjust Range for Reactions')
    ax.set_title('Sensitivity')
    # set the legend on the bottom of x axis
    ax.legend([handle1, handle2, handle3, handle4], labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon = False)
    # save the figure
    if save_dirpath is not None:
        fig.savefig(save_dirpath + f'/{figname}', bbox_inches='tight')
    # show the figure
    plt.close(fig)
    

def FatherSampleCompareWithNN(true_idt_data:np.ndarray, 
                              reduced_idt_data:np.ndarray, 
                              father_sample_data:np.ndarray, 
                              IDT_conditions:np.ndarray, 
                              save_path = None,
                              **kwargs):
    """
    father sample 版本的 compare_with_nn 函数
    绘制一张 father sample 中 IDT 分布与真实值和简化值的对比图,类似于 compare_with_nn 函数,但是将优化值使用箱线图代替。
    params:
        true_idt_data: the IDT data of the true mechanism
        reduced_idt_data: the IDT data of the reduced mechanism
        father_sample_data: the IDT data of the father sample
        save_path: the path to save the figure
        **kwargs: the parameters of compare_with_nn
    return:
        None
    """
    reduced_idt_data = reduced_idt_data - true_idt_data; father_sample_data = father_sample_data - true_idt_data
    # 使用 seaborn 根据 father_sample_data 绘制箱线图
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(20, 6)); ax.grid(False)
    format_settings()
    # 绘制的箱线图调整箱线图之间的间距
    sns.boxplot(data = father_sample_data, width=.6, palette="vlag", **kwargs)
    # 将 reduced_idt_data 画在箱线图上
    ax.plot(np.arange(father_sample_data.shape[1]), reduced_idt_data, '^', color = '#E53A40', label = 'Original', markersize = 10) 
    # 在 y = 0 的位置画一条水平线
    ax.axhline(y = 0, color = 'black', linestyle = '--')
    # 隐藏 xticks
    ax.set_xticks([])
    # 设置 ylim 的范围
    ylim_top = np.amax(np.abs(father_sample_data))
    ylim_top = max([ylim_top, np.amax(np.abs(reduced_idt_data))]) * 1.1; ylim_bottom = -ylim_top
    ax.set_ylim(ylim_bottom, ylim_top)
    # 设置 ylabel
    ax.set_ylabel(r'$\log$ IDT - $\log$ IDT$_{true}$')
    # 设置 xlabel
    ax.set_xlabel('Alignment points')
    # 设置 legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def CompareDRO_IDT_heatmap_NN(detail_data:np.ndarray,
                            reduced_data:np.ndarray,
                            optimal_data:np.ndarray, 
                            origin_detail_data:np.ndarray,
                            origin_reduced_data:np.ndarray,
                            origin_optimal_data:np.ndarray,
                            range_T:np.ndarray,
                            range_P:np.ndarray,
                            probe_point:np.ndarray = None,
                            save_path:str = '.',
                            labels = None,
                            colors = None,
                            Tlist = None, philist = None, Plist = None,
                            **kwargs):
    """
    将 ValidationIDT_heatmap 获得的图与 compare_nn_2 放在同一张图内展示
    所有参数与 ValidationIDT_heatmap 相同
    params:
        detail_data: the IDT data of the detailed mechanism, not use the logscale data
        reduced_data: the IDT data of the reduced mechanism, not use the logscale data
        optimal_data: the IDT data of the optimal mechanism, not use the logscale data
        origin_detail_data: the IDT data of the detailed mechanism, use the logscale data
        origin_reduced_data: the IDT data of the reduced mechanism, use the logscale data
        origin_optimal_data: the IDT data of the optimal mechanism, use the logscale data
        range_T: the range of the temperature
        range_P: the range of the pressure
        probe_point: the probe point
        save_path: the path to save the figure
        labels: the labels of the lines
        markers: the markers of the lines
        colors: the colors of the lines
        title: the title of the figure
        Tlist: the list of the temperature
        **kwargs: the parameters of the function
    """
    # font set to be 'Times New Roman'
    plt.rcParams['font.family'] = 'Times New Roman'
    # from kwargs get title of the plot
    cmap = kwargs.pop('cmap', 'Blues')

    optimal_data = np.abs(optimal_data - detail_data) / detail_data; reduced_data = np.abs(reduced_data - detail_data) / detail_data
    optimal_data = optimal_data.reshape(len(range_T), len(range_P)); reduced_data = reduced_data.reshape(len(range_T), len(range_P)); detail_data = detail_data.reshape(len(range_T), len(range_P))
    print(optimal_data, reduced_data)
    # optimal_data, reduced_data 沿着 T 轴反向排列
    optimal_data = np.flip(optimal_data, axis = 0); reduced_data = np.flip(reduced_data, axis = 0); detail_data = np.flip(detail_data, axis = 0)
    range_T = np.flip(range_T, axis = 0)

    vmin = min(np.min(optimal_data), np.min(reduced_data)); vmin = np.sign(vmin) * np.abs(vmin) * 1.1
    vmax = max(np.max(optimal_data), np.max(reduced_data)); vmax = np.sign(vmax) * np.abs(vmax) * 1.1

    fig = plt.figure(figsize = (12, 12))
    # combine two first row subplots into one
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1.75, 1], figure = fig, wspace = 0.05, hspace = 0.3)
    ax_heatmap1 = plt.subplot(gs[0, 0])
    ax_heatmap2 = plt.subplot(gs[0, 1])
    ax_heatmap_cbar = plt.subplot(gs[0, 2])
    ax_nn = plt.subplot(gs[1, :])
    
    sns.heatmap(reduced_data, ax = ax_heatmap1, vmin =vmin, vmax = vmax, cmap = cmap, cbar = False, **kwargs)
    sns.heatmap(optimal_data, ax = ax_heatmap2, vmin =vmin, vmax = vmax, cmap = cmap, cbar = False, **kwargs)

    # set the xticks and yticks of seaborn
    ax_heatmap1.set_xticks(np.arange(len(range_P))+0.5)
    ax_heatmap1.set_yticks(np.arange(len(range_T))[1::3]+0.5)
    ax_heatmap1.set_xticklabels(range_P)
    ax_heatmap1.set_yticklabels(range_T[1::3])
    ax_heatmap2.set_xticks(np.arange(len(range_P))+0.5)
    # ax_heatmap2.set_yticks(np.arange(len(range_T))+0.5)
    ax_heatmap2.set_xticklabels(range_P)
    ax_heatmap2.set_yticklabels([])
    ax_heatmap1.set_ylabel('Temperature (K)', fontsize = 16)
    # ax_heatmap2.set_ylabel('Temperature(K)', fontsize = 16)
    ax_heatmap1.set_xlabel('Pressure (atm)' + '\n' + 'Original reduced mechanism', fontsize = 16)
    ax_heatmap2.set_xlabel('Pressure (atm)' + '\n' + 'DeePMO reduced mechanism', fontsize = 16)
    # set ticklabels font size
    ax_heatmap1.tick_params(axis='x', which='major', labelsize=16)
    ax_heatmap2.tick_params(axis='x', which='major', labelsize=16)
    # ticklabels of axis y set to be horizontal
    ax_heatmap1.tick_params(axis='y', labelrotation=0, labelsize=14)    
    ax_heatmap2.tick_params(axis='y', labelrotation=0, labelsize=14)



    # only show one color bar of the whole plot
    # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.92, 0.45, 0.025, 0.5])
    cb = fig.colorbar(ax_heatmap1.collections[0], cax=ax_heatmap_cbar)
    cb.outline.set_visible(False)
    cb.set_label('Relative error', loc = 'center', fontsize = 12)

    if probe_point is not None:
        # plot a star marker on the probe point
        for working_condition in probe_point:
            try:
                # find the index of the probe point
                print(np.where(range_T == working_condition[0]))
                index_T = np.where(range_T == working_condition[0])[0][0]
                index_P = np.where(range_P == working_condition[1])[0][0]
                # plot the marker
                ax_heatmap1.plot(index_P + 1/2, index_T + 1/2, marker='^', markersize = 25 /2 , color="red", label = 'probe point')
                ax_heatmap2.plot(index_P + 1/2, index_T + 1/2, marker='^', markersize = 25 /2 , color="red", label = 'probe point')
                # show the label of marker on the bottom of the axis x; omit redundant legend
                lines, labels = ax_heatmap1.get_legend_handles_labels(); tmp_legend = dict(zip(labels, lines))
                lines, labels = tmp_legend.values(), tmp_legend.keys()
                # put the legend on the top center of the whole plot; use box to ancher the legend
                fig.legend(lines, labels, loc = 'upper center', bbox_to_anchor=(0.5, 0.925), ncol = 2, fontsize = 16, frameon = False)
                
            except:
                print('The probe point is not in the IDT condition.')
        
    # 调整数据维度,最后一个维度放温度 T
    origin_optimal_data = origin_optimal_data - origin_detail_data; origin_reduced_data = origin_reduced_data - origin_detail_data
    zorders = np.arange(2) * 0.1 + 1; zorders = zorders[::-1]
    # 获得上下界
    x_lim_left = np.min([np.amin(origin_reduced_data), np.amin(origin_optimal_data)]); x_lim_r = np.max([np.amax(origin_reduced_data), np.amax(origin_optimal_data)])
    count_num = len(origin_detail_data) // len(Tlist); major_tick_location = np.arange(len(origin_detail_data) + 1)[::count_num]
    tickerlocation = np.array(major_tick_location + count_num / 2, dtype = np.float16)[:-1]
    tickercontent = [rf"$T_0 = {T}K$" for T in Tlist]

    print("tickercontent:", tickercontent)

    ax_nn.axhline(y = 0, ls = '--', lw = 1, color = 'black', zorder = 1)
    ax_nn.scatter(np.arange(len(origin_optimal_data)), origin_optimal_data, c = colors[0], s = 45, marker = '^', label = "DeePMO", zorder = zorders[0])
    # ax_nn.scatter(np.arange(len(origin_reduced_data)), origin_reduced_data, c = colors[1], s = 40, marker = markers[1], label = "Reduced", zorder = zorders[1])
    ax_nn.scatter(np.arange(len(origin_reduced_data)), origin_reduced_data, c = colors[1], s = 45,  label = "Original", zorder = zorders[1], facecolors = 'w', marker = 'o',)
    lines, labels = ax_nn.get_legend_handles_labels()
    print(f"{labels=}")
    ax_nn.legend(lines, labels, loc='upper right', bbox_to_anchor = (1, 0.9), fontsize = 14, frameon=False)
    ax_nn.set_xlim(left = -1, right = len(origin_reduced_data) + 1)
    ax_nn.set_ylim(ymin = x_lim_left - 0.2, ymax = x_lim_r + 0.2)
    ax_nn.set_xlabel(r'Alignment points', fontsize = 16); ax_nn.set_ylabel(r'$\log$' + 'Absolute Error', fontsize = 16)
    ax_nn.xaxis.set_major_locator(ticker.FixedLocator(major_tick_location))
    ax_nn.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_nn.xaxis.set_minor_locator(ticker.FixedLocator(tickerlocation))
    ax_nn.xaxis.set_minor_formatter(ticker.FixedFormatter(tickercontent))
    for tick in ax_nn.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
        # 设置 xtick 的 fontsize 为 14
        tick.label1.set_fontsize(14)
    # 设置 xtick的 labels 的 fontsize 为 14
    ax_nn.tick_params(axis='x', labelsize = 14); ax_nn.tick_params(axis='y', labelsize = 14)
    # 在当前图像的右上角添加 phi 和 P 的值
    if philist is not None and Plist is not None:
        ax_nn.text(0.98, 0.98, rf"$\phi \in [{min(philist)}, {max(philist)}]$" + "\n" + rf"$P \in [{min(Plist)}, {max(Plist)}]$", 
               fontsize = 14, transform=ax_nn.transAxes, horizontalalignment='right', verticalalignment='top')
    fig.tight_layout(h_pad = 1.08, pad = 1.08)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


"""==============================================================================================================="""
"""                                              Network Verification                                             """
"""==============================================================================================================="""

def NV_sample_distribution(
        true_idt_data:np.ndarray, reduced_idt_data:np.ndarray,
        apart_idt_data: np.ndarray = None, network_predict_idt_data:np.ndarray = None, 
        apart_data_list = [], network_json_list = [], network_pth_list = [],
        probe_point_index = 1, sample_nums = 10000,
        save_path:str = None, **kwargs
    ):
    """
    抽取某个工况的检查采样的数据点绘制小提琴图 + 有网络输出分布的比较,确认覆盖真实值的分布
    params:
        total_circ: 总循环次数
        true_idt_data: 真实 IDT 数据, shape = (total_circ,)
        reduced_idt_data: 简化机理 IDT 数据, shape = (total_circ,)
        apart_idt_data: apart IDT 数据, shape = (total_circ, sample_nums)
        network_predict_idt_data: 网络预测的 IDT 数据, shape = (total_circ, sample_nums)
        apart_data_list: apart 数据的路径列表
        network_json_list: 网络 json 文件的路径列表
        network_pth_list: 网络 pth 文件的路径列表
        probe_point_index: 采样点的索引
        sample_nums: 采样点的数量
        save_path: 保存路径
    return:
        None 
    """       
    raise DeprecationWarning("This function is deprecated.")
    np.set_printoptions(suppress = True, precision = 1)
    if apart_idt_data is None or network_predict_idt_data is None:
        apart_data_idt = []; network_idt_data = []
        for apart_data_path, network_json_path, network_pth_path in zip(apart_data_list, network_json_list, network_pth_list):
            apart_data = np.load(apart_data_path, allow_pickle = True)
            apart_data_idt.append(apart_data['all_idt_data'][:,probe_point_index].tolist()[0:sample_nums])
            model = load_best_dnn(ANET_132, network_json_path, device = 'cpu',
                                model_pth_path = network_pth_path)
            data = model.forward_IDT(
                        torch.tensor(apart_data['Alist'], dtype = torch.float32)
                ).detach().numpy()[0:sample_nums, probe_point_index].tolist()
            network_idt_data.append(data)
        total_circ = len(apart_data_idt)
        # apart_data_idt = np.array(apart_data_idt)
        # for tmp in apart_data_idt:
        #     print(len(tmp), sep = '\n')
        apart_idt_data = np.log10(np.array(apart_data_idt)).reshape(-1, total_circ)
        network_predict_idt_data = np.array(network_idt_data).reshape(-1, total_circ)
        print(f"apart_data_idt.shape = {apart_idt_data.shape}")
        print(f"network_idt_data.shape = {network_predict_idt_data.shape}")
    else:
        total_circ = len(apart_idt_data)
    print("total_circums = ", total_circ)
    pd_trainDNN = pd.DataFrame(network_predict_idt_data, columns = range(total_circ), index = range(len(network_predict_idt_data))
    , dtype = float)
    pd_trainDNN['resource'] = 'DNN_predict'
    pd_sample = pd.DataFrame(apart_idt_data, columns = range(total_circ), index = range(len(apart_idt_data))
    , dtype = float)
    print(pd_sample.iloc[0:100,0:5])
    pd_sample['resource'] = 'Sample_Distribution'
    data = pd.concat([pd_sample, pd_trainDNN])
    data = pd.melt(data, var_name =  'TotalCirc', value_name = 'IDT', id_vars = 'resource')
    data['TotalCirc'] = data['TotalCirc'].astype('int64')
    print(data.info())
    fig = plt.figure(figsize = (total_circ * 0.75, 6),dpi = 300)
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    trueplot = ax.axhline(y = true_idt_data, ls = '--', c = 'maroon')
    idtnplot = ax.axhline(y = reduced_idt_data,  ls = '--',  c = 'teal')
    sns.violinplot(data = data, x = 'TotalCirc', y = 'IDT', hue = 'resource', 
                    split = True , ax = ax, bw = 0.1)
    # 将 sns 的 legend 和 trueplot 和 idtnplot 的 legend 合并
    handles, labels = ax.get_legend_handles_labels()
    handles = [trueplot, idtnplot] + handles
    labels = ['Detail IDT', 'Original IDT'] + labels
    # legend 放置在 x axis 的下方
    legend = ax.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0.05), ncol = 4, frameon = False)
    ax.set_xlabel('Iterations' + '\n' + 'Samples and Test Data Distribution Along Circulation'); ax.set_ylabel('IDT in Log scale')
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)   


def NetSampleIter_Loss(
            average_loss:np.ndarray, max_loss:np.ndarray, 
            sample_range:np.ndarray, 
            average_loss_std:np.ndarray = None, sample_range_std:np.ndarray = None,
            save_path:str = None,
):
    """
    生成 Net-Sample 过程中的 loss 变化曲线; 涉及到对于每个工况的平均 loss 和最大 loss 的绘制
    同时利用 twinx 绘制筛选过程中的采样范围变化情况
    params:
        average_loss: 平均 loss, shape = (total_circ,)
        max_loss: 最大 loss, shape = (total_circ,)
        sample_range: 采样范围, shape = (total_circ,)
        average_loss_std: 平均 loss 的标准差, shape = (total_circ,)
        sample_range_std: 采样范围的标准差, shape = (total_circ,)
        save_path: 保存路径
    return: 
        None
    """
    fig = plt.figure(figsize = (6, 4), dpi = 300)
    ax = fig.add_axes([0.15,0.15,0.75,0.75])
    ax.plot(average_loss, label = 'average loss', c = 'teal')
    ax.plot(max_loss, label = 'max loss', c = 'maroon')
    if average_loss_std is not None:
        ax.fill_between(range(len(average_loss)), average_loss - average_loss_std, average_loss + average_loss_std, color = 'teal', alpha = 0.2)
    ax.set_xlabel('Iterations' + '\n' + "Loss Curve"); ax.set_ylabel('Log Absolute Loss')
    ax2 = ax.twinx()
    ax2.plot(sample_range, label = 'sample range', c = 'darkorange')
    if sample_range_std is not None:
        ax2.fill_between(range(len(max_loss)), max_loss - sample_range_std, max_loss + sample_range_std, color = 'maroon', alpha = 0.2)
    ax2.set_ylabel('Sample Range')
    # 合并 ax 和 ax2 的 legend
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles + handles2
    labels = labels + labels2
    legend = ax.legend(handles, labels, loc = 'center', bbox_to_anchor = (0.5, 1.05), ncol = 4, frameon = False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def CompareNetworkCanteraSensitivity(network:nn.Module, father_sample: np.ndarray, 
                                     reduced_chem: str, phi, T, P, fuel, oxidizer, probe_point_index,
                                     delta = 0.05, rea_keywords = None, save_dirpath = "."):
    """
    比较训练好的替代模型网络和 Cantera 的敏感度分析结果
    params:
        network: 训练好的替代模型网络
        father_sample: 父机理的采样数据, shape = (sample_nums, total_circ)
        reduced_chem: 简化机理的路径
        phi, T, P: 工况
        fuel, oxidizer: 燃料和氧化剂的组分
        rea_keywords: 简化机理不考虑的 core 之外的元素
        save_dirpath: 保存路径
    return:
        None
    """
    # 使用简化机理和 father_sample 生成当前优化中心的机理
    path = mkdirplus(save_dirpath + "/NetworkCanteraSensitivity")
    optim_mech = Adict2yaml(original_chem_path = reduced_chem, chem_path = save_dirpath + "/NetworkCanteraSensitivity/optim_mech.yaml", Alist = father_sample
                            , rea_keywords = rea_keywords)
    father_sample, eq_dict = yaml_key2A(optim_mech, rea_keywords = rea_keywords)
    optim_sensitivity = yaml2idt_sensitivity(
        optim_mech,
        phi, T, P,
        fuel, oxidizer,
        delta,
        save_path = path + "/optim_mech_cantera_sensitivity.json",
    )
    optim_sensitivity = {key: optim_sensitivity[key] for key in eq_dict.keys()}
    # 计算网络在 father_sample 上的敏感度
    ## 首先生成网络的敏感度分析数据; 要求为对每一个维度都进行 delta 大小的扰动
    network_sensitivity = {}
    base_idt = network(torch.tensor(father_sample, dtype = torch.float32)).detach().numpy()
    base_idt = base_idt.reshape(2, 4, 3)
    base_idt = 10 ** base_idt[probe_point_index[0], probe_point_index[1], probe_point_index[2]]
    for i, equation in enumerate(eq_dict.keys()):
        sample = father_sample.copy()
        sample[i] += delta
        idt = network(torch.tensor(sample, dtype = torch.float32)).detach().numpy()
        idt = idt.reshape(2, 4, 3)
        idt = 10 ** idt[probe_point_index[0], probe_point_index[1], probe_point_index[2]]
        # sample[i] -= 2 * delta
        # idt2 = network(torch.tensor(sample, dtype = torch.float32)).detach().numpy()
        # idt2 = idt2.reshape(2, 4, 3)
        # idt2 = idt2[probe_point_index[0], probe_point_index[1], probe_point_index[2]]
        # idt -= idt2
        network_sensitivity.update({equation: (idt - base_idt) / (delta * base_idt)})
    # 保存 network_sensitivity
    
    # 生成敏感度分析的图像; 形式为金字塔图,按照敏感度从大到小排列
    ## 生成敏感度的排序
    print("network_sensitivity" ,list(network_sensitivity.items()))
    optim_sensitivity_sorted = sorted(list(optim_sensitivity.items()), key = lambda x: x[1], reverse = True)
    network_sensitivity_sorted = sorted(list(network_sensitivity.items()), key = lambda x: x[1], reverse = True)
    ## 先按照 optim_sensitivity_sorted 的顺序生成金字塔图; 即生成一张横置的 barplot,第一象限为 cantera 的敏感度,第二象限为 network 的敏感度。纵坐标为反应方程
    fig = plt.figure(figsize = (6, 4), dpi = 300)
    ax = fig.add_axes([0.15,0.15,0.75,0.75])
    ax.barh(range(len(optim_sensitivity_sorted)), [i[1] for i in optim_sensitivity_sorted], color = 'teal', label = 'Cantera')
    ax.barh(range(len(optim_sensitivity_sorted)), [-i[1] for i in network_sensitivity_sorted], color = 'maroon', label = 'Network')
    ax.set_yticks(range(len(optim_sensitivity_sorted)))
    ax.set_yticklabels([i[0] for i in optim_sensitivity_sorted])
    ax.set_xlabel('Sensitivity'); ax.set_ylabel('Reaction')
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 0.05), ncol = 2, frameon = False)
    fig.tight_layout()
    fig.savefig(path + '/NetworkCanteraSensitivity.png')
    plt.close(fig)
    ## 再按照 network_sensitivity_sorted 的顺序生成金字塔图; 即生成一张横置的 barplot,第一象限为 cantera 的敏感度,第二象限为 network 的敏感度。纵坐标为反应方程
    fig = plt.figure(figsize = (6, 4), dpi = 300)
    ax = fig.add_axes([0.15,0.15,0.75,0.75])
    ax.barh(range(len(network_sensitivity_sorted)), [i[1] for i in network_sensitivity_sorted], color = 'maroon', label = 'Network')
    ax.barh(range(len(network_sensitivity_sorted)), [-i[1] for i in optim_sensitivity_sorted], color = 'teal', label = 'Cantera')
    ax.set_yticks(range(len(network_sensitivity_sorted)))
    ax.set_yticklabels([i[0] for i in network_sensitivity_sorted])
    ax.set_xlabel('Sensitivity'); ax.set_ylabel('Reaction')
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 0.95), ncol = 2, frameon = False)
    fig.tight_layout()
    fig.savefig(path + '/CanteraNetworkSensitivity.png')
    plt.close(fig)


"""==============================================================================================================="""
"""                                              逐反应测试                                                        """
"""==============================================================================================================="""

def drawReactionAdjustScale_with_Circ(best_sample_list, father_samples, reaction_name:str = None, reaction_index:int = None, 
                                      eq_dict = None, alpha_dict_list = None, save_path = "./drawReactionAdjustScale_with_Circ.png", **kwargs):
    """
    绘制在整个流程中,对于一个指定反应的调整大小变化情况; 用于详细分析一个反应在调整过程中的具体变化情况
    以期获得这个反应在整个体系中扮演的角色。与此同时绘制的还有最优 15 个样本中这个反应最终值的方差情况
    """
    # 若 reaction_name 不为 None, 使用 eq_dict 获得 reaction_index
    if reaction_name is not None:
        reaction_index = list(eq_dict.keys()).index(reaction_name)
    else:
        reaction_name = list(eq_dict.keys())[reaction_index]
    if alpha_dict_list is not None:
        alpha_value = [
            tmp_alpha_dict[reaction_name] for tmp_alpha_dict in alpha_dict_list
            ]
        alpha_value = np.array(alpha_value).T
    else:
        alpha_value = None
    # 将 reaction_index 修正为 Alist 中的真实索引; 如果有多个对象,统一选择第一个对象
    reaction_index = sum(
        [len(np.array(eq_dict[key]).flatten()) for i, key in enumerate(eq_dict.keys()) if i < reaction_index]
    )
    
    # 获得 best_sample_list 中所有样本的 reaction_index 反应的值
    # best_sample_list = [np.load(i) for i in best_sample_list]
    # 如果 best_sample_list 中的元素是 2 维的,则取第一个元素
    if len(best_sample_list[0].shape) == 2:
        best_sample_list = [i[0] for i in best_sample_list]
    best_sample_reaction_value = np.array([i[reaction_index] for i in best_sample_list])
    print(best_sample_reaction_value.shape, best_sample_reaction_value)

    # 获得 father_samples 中所有样本的 reaction_index 反应的值
    father_samples = father_samples[:, :, reaction_index]
    # 计算其 std
    father_samples_std = []; new_father_samples = []
    for father_sample, tmp_best_sample_reaction_value in zip(father_samples, best_sample_reaction_value):
        father_sample = father_sample[father_sample != 0]
        father_samples_std.append([
            np.amin(father_sample - tmp_best_sample_reaction_value),
            np.amax(father_sample - tmp_best_sample_reaction_value)
        ])
        new_father_samples.append(father_sample)
    father_samples_std = np.abs(np.array(father_samples_std).T)
    father_samples = new_father_samples
    print(father_samples_std.shape, father_samples_std)
    # 绘制图像
    fig, ax = plt.subplots(figsize = (6, 4), dpi = 200)
    format_settings()
    ax.plot(np.array(range(len(best_sample_reaction_value))), best_sample_reaction_value, label = 'Best Sample')
    # ax.plot(np.array(range(1, len(best_sample_reaction_value) + 1)), father_samples_std[0], label = 'Best Sample')
    # 将 father_samples_std 以误差轴的格式绘制在图像上
    ax.errorbar(x = np.array(range(len(best_sample_reaction_value))) - 0.05, y = best_sample_reaction_value, 
                yerr = father_samples_std, label = 'Father Sample Min-Max', fmt='-', elinewidth = 2, linestyle = "")
    sns.violinplot(data = father_samples, ax = ax, color = 'teal', label = 'Father Sample')
    if alpha_value is not None:
        alpha_value = np.abs(alpha_value)
        print(alpha_value)
        # alpha_value = np.abs(np.repeat(alpha_value, len(best_sample_reaction_value), axis = 1))
        # print(alpha_value)
        ax.errorbar(np.array(range(len(best_sample_reaction_value))) + 0.05, best_sample_reaction_value, yerr = alpha_value, label = 'Alpha Value',  fmt='-', elinewidth = 2, ls = "")
    ax.set_xlabel('Iterations'); ax.set_ylabel('Reaction Value')
    ax.set_title(f"Reaction {reaction_name} Value Adjust Scale")
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1), ncol = 3, frameon = False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
