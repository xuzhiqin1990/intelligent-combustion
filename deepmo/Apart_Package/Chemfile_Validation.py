# -*- coding:utf-8 -*-
import seaborn as sns
import pandas as pd
import sys, traceback, time
from concurrent.futures import ProcessPoolExecutor
import matplotlib.cm as cm
from multiprocessing import Pool
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import FuncFormatter, ScalarFormatter

sys.path.append('..')
from Apart_Package.utils import *
from Apart_Package.utils.cantera_multiprocess_utils import *
from Apart_Package.utils.setting_utils import *
from Apart_Package.utils.yamlfiles_utils import *
from Apart_Package.APART_plot.Result_plot import *
from Apart_Package.APART_base import _GenOneLFS, _GenOneIDT_HRR
from Apart_Package.APART_plot.APART_plot import classify_array_phi_P, classify_array_T_P

from matplotlib.colors import LinearSegmentedColormap

# 定义颜色列表
colors = ['#831A21', '#A13D3B', '#D5E490', '#C16D58', '#ECD0B4', '#F2EBE5', '#E8EDF1', '#C8D6E7', '#9EBCDB', '#7091C7', '#4E70AF', '#375093']
colors = ['#4e62ab', '#469eb4', '#87cfa4', '#cbed9d', '#f5fbb1'][:-1]
cmap_name = 'BlueToRed'
format_settings()
# 创建色系
cm01 = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)


class Chemfile_Validation(object):
    """
    用于验证最终机理的结果并进行进一步分析的类，目的是彻底替换原 APART132.validation，使用方式

    1. 继承了 object 类, 因此不需要输入任何内容，简化了调用方式。在 __init__ 中实现了以下功能：
        1.1 在当前路径下生成 validation 文件夹
        1.2 加载 self.detail_mech, self.reduced_mech, self.optim_mech, self.experimet_dataset


    2. 使用 Validation_xxx 等方法进行验证，目前实现了以下方法：
        2.1 ValidationIDT_heatmap: 绘制 IDT 的 heatmap, 只考虑 range_T 和 range_P, range_phi 只能取单点值
        2.2 ValidationIDT_heatmap_NN: 绘制 IDT 的 heatmap, 只考虑 range_T 和 range_P, range_phi 只能取单点值 + compare_nn_plot 的图像合并
        2.3 ValidationIDT_lineplot: 绘制 IDT 关于 1000 / 初值温度 的图像
        2.4 ValidationPSR_lineplot: 绘制 PSR 温度关于 res time 的图像
        2.5 ValidationFS_lineplot: 绘制 FS 关于当量比的图像
        2.6 ValidationTimeTemperature: 绘制 IDT 的时间温度曲线
        2.7 ValidationTimeSpecies: 绘制 IDT 的时间物质mole分数曲线

    """
    def __init__(self, 
                 dir_path = ".",
                 detail_mech: str = None,
                 reduced_mech: str = None,
                 optim_mech: str = None,
                 IDT_mode: int = None, 
                 fuel: str = None,
                 oxidizer: str = None,
                 setup_file: str = None,
                 **kwargs) -> None:
        ct.suppress_thermo_warnings()
        # 在当前路径下生成 validation 文件夹
        self.vlidpath = mkdirplus(f'{dir_path}/validation')
        # 加载 self.detail_mech, self.reduced_mech, self.optim_mech, self.experiment_dataset
        self.detail_mech = detail_mech if detail_mech is not None else dir_path + "/settings/chem.yaml"
        self.reduced_mech = reduced_mech if reduced_mech is not None else dir_path + "/settings/reduced_chem.yaml"
        self.optim_mech = optim_mech if optim_mech is not None else dir_path + "/optim_chem.yaml"
        if setup_file is not None:
            self.setup_file = setup_file
            chem_args = get_yaml_data(setup_file)
            self.fuel, self.oxidizer = chem_args.get('fuel', None), chem_args.get('oxidizer', None)
            self.IDT_mode = chem_args['IDT_mode'] if 'IDT_mode' in chem_args.keys() else IDT_mode
            self.PSR_mode = chem_args['PSR_mode'] if 'PSR_mode' in chem_args.keys() else False
            self.PSR_exp_factor = 2 ** chem_args['PSRex_decay_exp'] if 'PSRex_decay_exp' in chem_args.keys() else 2 ** (1/2)
            if self.fuel is None or self.oxidizer is None:
                self.fuel, self.oxidizer = fuel, oxidizer
        else:
            self.fuel, self.oxidizer = fuel, oxidizer   
        self.IDT_mode = IDT_mode if IDT_mode is not None else 1
        # assert self.fuel is not None and self.oxidizer is not None, "fuel and oxidizer can't be None"
        format_settings()


    def GenTrue_RES_TIME_LIST(self, PSR_condition = None, save_path = None, init_res_time = 1, exp_factor = None, **kwargs):
        """
        生成 PSR 对应的 true RES_TIME_LIST 列表，注意，一个工况就对应着 PSR 的一个列表
        param:
            PSR_condition: PSR 的工况，格式为 [(phi, T, p), ...]
            save_path: 保存路径
            init_res_time: 初始的 res time
            exp_factor: 每次迭代的 res time 的缩减因子
        add:
            2021.10.19: 生成 self.True_RES_TIME_LIST, self.True_PSR_T_list
        """
        exp_factor = self.PSR_exp_factor if exp_factor is None else exp_factor
        if exp_factor == 1: 
            raise ValueError("exp_factor can't be 1")   
        PSR_condition = self.PSR_condition if PSR_condition is None else PSR_condition
        if save_path is not None and os.path.exists(save_path):
            tmp_npz_file = np.load(save_path, allow_pickle = True)
            self.True_RES_TIME_LIST = tmp_npz_file['True_RES_TIME_LIST']
            self.True_PSR_T_list = tmp_npz_file['True_PSR_T_list']
        else:
            self.True_RES_TIME_LIST, self.True_PSR_T_list = [], []
            for condition_item in PSR_condition:
                gas = ct.Solution(self.detail_mech)
                phi, T, p = condition_item
                gas.TP = T, p * ct.one_atm
                gas.set_equivalence_ratio(phi, self.fuel, self.oxidizer)
                RES_TIME_LIST, PSR_T_LIST = solve_psr_true(gas, exp_factor = exp_factor, ini_res_time=init_res_time, **kwargs)
                self.True_RES_TIME_LIST.append(RES_TIME_LIST); self.True_PSR_T_list.append(PSR_T_LIST)
            # if save_path is not None: np.savez(save_path, True_RES_TIME_LIST = self.True_RES_TIME_LIST, True_PSR_T_list = self.True_PSR_T_list, PSR_condition = PSR_condition)
        return self.True_RES_TIME_LIST, self.True_PSR_T_list


    def Sample_Quality_Index_heatmap(self, apart_data_path_list: list[str], weight_dict:dict[str: float], true_data_dict, divide_bins = 10, 
                             save_dirpath = ".", loss_func_ord = 2, prepared_loss_bins = None, vmin = None, vmax = None, **kwargs):
        """
        判断在过程中的样本质量变化情况，顺序读取 apart_data_path_list 中的内容后按照 weight 计算与真实值的加权差距大小；
        将总 loss 按照 divide_bins 分段，统计每个分段的样本数目
        params:
            apart_data_path_list: apart_data 的路径列表
            weights_dict: 权重列表; 以 target: weight 的形式输入
            true_data_dict: 真实值列表; 以 target: true_data 的形式输入
            divide_bins: 分段数目
            Prepared_Loss: 事先准备好的 Loss，如果不为 None 则直接使用该 Loss 进行分析
        """
        mkdirplus(save_dirpath)
        target = list(weight_dict.keys()); file_target = ['all_' + i + '_data' for i in target]
        example_file = np.load(apart_data_path_list[0], allow_pickle = True)
        # 检查 file_target 是否在 example_file.files 中
        assert all([i in example_file.files for i in file_target]), f"file_target {file_target} is not in {example_file.files}"
        Loss_bin_num, Loss = [], []
        for file in apart_data_path_list:
            data = np.load(file, allow_pickle = True)
            loss = np.zeros(len(data['Alist']))
            for target_item in target:
                tmp_weight = weight_dict[target_item]
                tmp_data = data['all_' + target_item + '_data']
                if target_item == 'idt' or target_item == 'hrr':
                    tmp_data = np.log10(tmp_data)
                elif target_item == 'psr_extinction_time':
                    tmp_data = np.log2(tmp_data)
                loss += tmp_weight * np.linalg.norm(tmp_data - true_data_dict[target_item], axis = 1, ord = loss_func_ord)
            # 将 loss 按照 divide_bins 分段，统计每个分段的样本数目
            loss_bins = np.linspace(0, np.max(loss), divide_bins + 1); loss_bins = np.append(loss_bins, 1e2)
            ## 统计分段样本数
            loss_bins_num = np.zeros(len(loss_bins) - 1)
            for i in range(len(loss_bins) - 1):
                loss_bins_num[i] = np.sum((loss >= loss_bins[i]) & (loss < loss_bins[i + 1]))
            Loss_bin_num.append(loss_bins_num); Loss.append(loss)
            print(f"file {file} loss_bins_num is {loss_bins_num}")
            # 绘制样本 loss 的 histplot
            if save_dirpath is not None:
                fig, ax = plt.subplots()
                sns.histplot(loss, bins = loss_bins, kde = True, stat = 'density', ax = ax)
                plt.savefig(save_dirpath + f"/{file.split('/')[-1].split('.')[0]}_loss_histplot.png")
                plt.close(fig)
                
        Loss_bin_num = np.array(Loss_bin_num)
        # 绘制 Loss 的 heatmap
        ## 对 Loss 的第 0 维所有元素进行统一的按照一个相同的 bin 分段
        if prepared_loss_bins is None:
            loss_bins = np.linspace(0, np.max([np.amax(data) for data in Loss]), divide_bins * 2) 
            loss_bins = np.append(loss_bins, 1e2)
        else:
            loss_bins = prepared_loss_bins
        loss_bins_num = []
        ## 统计每个维度的分段样本数
        for i in range(len(Loss)):
            # print("shape", loss_bins_num[:, i].shape)
            # print(np.histogram(Loss[i], bins = loss_bins)[0].shape)
            loss_bins_num.append(np.histogram(Loss[i], bins = loss_bins)[0].tolist())
        loss_bins_num = np.array(loss_bins_num) / np.sum(loss_bins_num, axis = 1)[:, np.newaxis]
        ## 绘制 heatmap
        if save_dirpath is not None:
            fig, ax = plt.subplots(figsize = (7.5, 8 * 0.75))
            loss_bins_num = np.flip(np.transpose(loss_bins_num), axis = 0)
            sns.heatmap(loss_bins_num, cmap = cm.get_cmap('Blues'), ax = ax, cbar = True, cbar_kws = {'shrink': 0.8, 'label': 'Density'}, vmin = vmin, vmax = vmax)
            ax.set_ylabel("Average Sample Loss", fontsize = 18); ax.set_xlabel("Iteration", fontsize = 18)
            diff_loss_bins = np.flip(np.array(
                [(loss_bins[i + 1] + loss_bins[i]) /2 for i in range(len(loss_bins) - 1)]
            ))
            ax.yaxis.set_major_formatter(ScalarFormatter())  # 交换 xaxis 和 yaxis 的 formatter
            ax.set_yticks(np.arange(len(diff_loss_bins))[::2], fontsize = 16)  # 交换 xticks 和 yticks
            # yticklabel 使用科学计数法表示
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.yaxis.offsetText.set_visible(False)
            
            ax.set_yticklabels([f"{diff_loss_bins[i]:.1e}" for i in range(len(diff_loss_bins))][::2], rotation='horizontal', fontsize = 14)  # 交换 xticklabels 和 yticklabels
            ax.xaxis.set_major_formatter(ScalarFormatter())  # 交换 xaxis 和 yaxis 的 formatter
            ax.set_xticks(np.arange(len(Loss))[::2])  # 交换 xticks 和 yticks
            ax.set_xticklabels([f"{i}" for i in range(len(Loss))[::2]], fontsize = 16)  # 交换 xticklabels 和 yticklabels
            # ax.set_title("Loss Distribution Density heatmap", fontsize = 16)
            fig.tight_layout()
            plt.savefig(save_dirpath + f"/Loss_heatmap.png", dpi = 300, pad_inches = 0.1)
            plt.close(fig)
        return Loss_bin_num, Loss, loss_bins_num, loss_bins


    def Sample_Quality_Index(self, apart_data_path_list: list[str], weight_dict:dict[str: float], true_data_dict, loss_func_ord = 2, ):
        """
        判断在过程中的样本质量变化情况，顺序读取 apart_data_path_list 中的内容后按照 weight 计算与真实值的加权差距大小；
        返回样本质量的分布情况
        params:
            apart_data_path_list: apart_data 的路径列表
            weights_dict: 权重列表; 以 target: weight 的形式输入
            true_data_dict: 真实值列表; 以 target: true_data 的形式输入
        """
        target = list(weight_dict.keys()); file_target = ['all_' + i + '_data' for i in target]
        example_file = np.load(apart_data_path_list[0], allow_pickle = True)
        # 检查 file_target 是否在 example_file.files 中
        assert all([i in example_file.files for i in file_target]), f"file_target {file_target} is not in {example_file.files}"
        Loss = []; slice1 = int(4e4)
        for file in apart_data_path_list:
            data = np.load(file, allow_pickle = True)
            loss = np.zeros(len(data['Alist']))
            for target_item in target:
                tmp_weight = weight_dict[target_item]
                tmp_data = data['all_' + target_item + '_data']
                # if len(weight_dict) == 1:
                #     loss += np.mean(np.abs((tmp_data - true_data_dict[target_item]) / true_data_dict[target_item])) * 100
                # if target_item == 'idt' or target_item == 'hrr':
                #     tmp_data = np.log10(tmp_data)
                # elif target_item == 'psr_extinction_time':
                #     tmp_data = np.log2(tmp_data)
                # loss += tmp_weight * np.linalg.norm(tmp_data - true_data_dict[target_item], axis = 1, ord = loss_func_ord)
                loss += np.mean(np.abs((tmp_data - true_data_dict[target_item]) / true_data_dict[target_item]), axis = 1) * 100
            # print(loss[:slice1].shape)
            Loss.append(loss[:slice1])
        return np.array(Loss)


    def Sample_Frequency_Index(self, apart_data_QoI: list[np.ndarray], apart_data: list[np.ndarray], QoI_name: str = None, save_dirpath = None, **kwargs):
        """
        绘制在整个过程中样本 QoI 值的频率变化情况，频率的定义为最大 QoI - 最小 QoI / 最大样本与最小样本的最大、最小和欧式距离
        对于多工况的情况，具体选择参考 loss_mode 参数
        params:
            apart_data_QoI: apart_data 中的 QoI 数据
            QoI_name: QoI 名字
            loss_mode: loss 的计算方式; 可选 average, max, min
        """
        average_frequency = []; max_frequency = []; min_frequency = []; max_center_frequency = []
        Dist_center = []
        for apart_data_QoI_percirc, apart_data_percirc in zip(apart_data_QoI, apart_data):
            apart_data_QoI_percirc = np.linalg.norm(apart_data_QoI_percirc, axis = 1)
            max_index = np.argmax(apart_data_QoI_percirc)
            min_index = np.argmin(apart_data_QoI_percirc)
            max_alist = apart_data_percirc[max_index]
            min_alist = apart_data_percirc[min_index]
            max_qoi = apart_data_QoI_percirc[max_index]
            min_qoi = apart_data_QoI_percirc[min_index]
            # 计算 apart_data_percirc 中的重心坐标
            center_alist = np.mean(apart_data_percirc, axis = 0)
            max_center_distance = np.linalg.norm(apart_data_percirc - center_alist, ord = 2)
            # 找到 Qoi 最大的样本和最小的样本位置
            average_frequency.append((max_qoi - min_qoi) / np.linalg.norm(max_alist - min_alist, ord = 2))
            max_frequency.append((max_qoi - min_qoi) / np.linalg.norm(max_alist - min_alist, ord = np.inf))
            min_frequency.append((max_qoi - min_qoi) / np.linalg.norm(max_alist - min_alist, ord = -np.inf))
            max_center_frequency.append((max_qoi - min_qoi) / max_center_distance)
            # experiment_time = apart_data_percirc.shape[0]
            # # 样本点之间的平均距离
            # dist_center = 0
            # for i in range(experiment_time):
            #     for j in range(i+1, experiment_time):
            #         dist_center += np.linalg.norm(apart_data_percirc[i, :] - apart_data_percirc[j, :], ord = 2)
            # dist_center /= (experiment_time ** 2 - experiment_time) / 2  
            # Dist_center.append(dist_center) 
        average_frequency = np.array(average_frequency); max_frequency = np.array(max_frequency); min_frequency = np.array(min_frequency)
        fig, ax = plt.subplots(); ax2 = ax.twinx()
        ax.semilogy(average_frequency, label = "average_frequency")
        ax.semilogy(max_frequency, label = "max_frequency")
        # ax.semilogy(min_frequency, label = "min_frequency")
        ax.semilogy(max_center_frequency, label = "max_center_frequency")
        # ax2.plot(Dist_center, label = "Dist_center")
        # 填充这三条曲线之间的区域
        ax.fill_between(np.arange(len(average_frequency)), average_frequency, max_frequency, alpha = 0.2)
        # ax.fill_between(np.arange(len(average_frequency)), average_frequency, min_frequency, alpha = 0.2)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Frequency")
        ax.set_title(f"Frequency of {QoI_name}")
        fig.legend(ncol = 3, bbox_to_anchor = (0.5, 1), loc = 'center', frameon = False)
        fig.tight_layout(pad = 1.1)
        plt.savefig(save_dirpath + f"/{QoI_name}_frequency.png", dpi = 300)
        plt.close(fig)
        return average_frequency, max_frequency, min_frequency
                
    
    def BestSample_Loss_Along_Iter(self, best_sample_list: list[np.ndarray], true_data: np.ndarray, loss_ord = 2, figsave_path = None, **kwargs):
        """
        计算 best_sample_list 中的每个样本与 true_data 的 loss，返回一个列表
        params:
            best_sample_list: best_sample 的列表
            true_data: 真实值
            loss_ord: loss 的计算方式
        """
        loss_list = []
        max_loss_list, min_loss_list = [], []
        for best_sample in best_sample_list:
            loss_list.append(np.linalg.norm(best_sample - true_data, ord = loss_ord) / len(true_data))
            max_loss_list.append(np.linalg.norm(best_sample - true_data, ord = np.inf))
            min_loss_list.append(np.linalg.norm(best_sample - true_data, ord = -np.inf))
        # 选择三个 sci 颜色
        
        if figsave_path is not None:
            fig, ax = plt.subplots()
            ax.plot(loss_list, label = "loss", c = 'r')
            # ax.plot(max_loss_list, label = "max_loss", lw = 0.5, ls = '--', c = '#FFA511')
            # ax.plot(min_loss_list, label = "min_loss", lw = 0.5, ls = '--', c = '#FFA490')
            ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
            ax.set_title("Loss Along Iteration")
            # 填充这三条曲线之间的区域
            # ax.fill_between(np.arange(len(loss_list)), loss_list, max_loss_list, alpha = 0.2)
            # ax.fill_between(np.arange(len(loss_list)), loss_list, min_loss_list, alpha = 0.2)
            fig.legend()
            fig.tight_layout(pad = 1.1)
            plt.savefig(figsave_path, dpi = 300)
            plt.close(fig)
        return loss_list, max_loss_list, min_loss_list
    
    
    def Get_Reaction_Ajustment_Range(self, best_sample_list, sensitivity_list = None, save_dirpath = '.'):
        """
        
        """
        mkdirplus(save_dirpath)
        original_A0, eq_dict = yaml_key2A(self.reduced_mech)
        Eq_dicts = [[] for _ in range(len(best_sample_list))]
        for i, best_chem in enumerate(best_sample_list):
            best_A0, _ = yaml_key2A(best_chem)
            # 计算两者的差值
            diff_A0 = best_A0 - original_A0
            diff_eq_dict = Alist2eq_dict(diff_A0, eq_dict)
            # 按照绝对值大小排序
            diff_eq_dict = dict(sorted(diff_eq_dict.items(), key = lambda x: np.mean(np.abs(x[1])).item(), reverse = True))
            # 保存到文件中
            with open(save_dirpath + f"/{i}_diff_A0.json ", 'w') as f:
                json.dump(diff_eq_dict, f)
            tmp_sensitive = sensitivity_list[i]
            if not os.path.exists(tmp_sensitive):
                print(f"File {tmp_sensitive} does not exist.")
                continue
            with open(tmp_sensitive, 'r') as f:
                tmp_sensitive = json.load(f)
            # 找到 sensitivity 大于 5 的反应
            tmp_sensitive = {k: v for k, v in tmp_sensitive.items() if np.abs(v) > 5}
            # 筛选出前 10 个 diff_eq_dict
            diff_eq_dict = {k: v for k, v in diff_eq_dict.items() if k in tmp_sensitive.keys()}
            # 按照 diff_eq_dict 的大小排序
            diff_eq_dict = dict(sorted(diff_eq_dict.items(), key = lambda x: np.mean(np.abs(x[1])).item(), reverse = True))
            # 打印前 10 个
            print(f'Iteration {i} len of diff_eq_dict is {len(diff_eq_dict)}')
            kk = 0
            for k, v in diff_eq_dict.items():
                # print(f"Reaction {k} Sensitivity {tmp_sensitive[k]}; Diff {v}")
                Eq_dicts[i].append(
                    k
                )
                kk += 1
                if kk > 10:
                    print('='*50)
                    break
            
        # 统计 Eq_dicts 中的每个反应出现次数
        Eq_dicts_flatten = [item for sublist in Eq_dicts for item in sublist]
        from collections import Counter
        Eq_dicts_counter = Counter(Eq_dicts_flatten)
        Eq_dicts_counter = dict(sorted(Eq_dicts_counter.items(), key = lambda x: x[1], reverse = True))
        print(Eq_dicts_counter)  
            
            
        return diff_eq_dict
    """==============================================================================================================="""
    """                                  Validation FOR Detail Mechanism                                              """
    """==============================================================================================================="""


    def ValidationIDT_heatmap(self, range_phi:float, range_T:np.ndarray, range_P, 
                              optim_mech, probe_point = None, save_path = ".", 
                              multiprocess:int = False, cut_time = 1, yaml2idt_kwargs:dict = {},
                              **kwargs):
        """
        20230327 增加
        绘制 IDT 的 heatmap, 只考虑 range_T 和 range_P, range_phi 只能取单点值
        实现了以下的步骤：
        1. 使用单进程/多进程计算给定机理 + detail/reduced 的 IDT
        2. 使用 Result_plot 中的 Compare 系

        params:
            range_phi, T, P: 绘图中确定的工况, 其中 phi 只能取单点值
            optim_mech: 优化后的机理文件名
            multiprocess: 是否使用多进程; 默认不使用, 若输入为整数则指定为核数
            save_path: 保存所有生成文件的文件夹名字
            probe_point: 在 APART 过程中用于训练的工况
            sharey: plt.subplots 中是否 sharey
            yaml2idt_kwargs: 传递给 yaml2idt 的参数
        """
        mechs = {
            "optimal": optim_mech,
            "detail": self.detail_mech,
            "reduced": self.reduced_mech
        }
        IDT = {}
        if multiprocess:
            with ProcessPoolExecutor(max_workers = multiprocess) as exec:
                for mech in mechs.keys():
                    def callback(future):
                        IDT.update({mech: future.result()[0]})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    try:
                        tmp_save_path = save_path + f"/{mech}_ValidationIDT_heatmap_IDT.npz"
                        future = exec.submit(
                                yaml2idt, 
                                mechs[mech],
                                mode = self.IDT_mode,
                                IDT_phi = [range_phi], 
                                IDT_T = range_T, 
                                IDT_P = range_P, 
                                fuel = self.fuel,
                                oxidizer = self.oxidizer,
                                save_path = tmp_save_path,
                                cut_time = cut_time,
                                **yaml2idt_kwargs
                                ).add_done_callback(callback)
                    except Exception as r:
                        print(f'Multiprocess error; error reason:{r}')
        else:
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_heatmap_IDT.npz"
                if os.path.exists(tmp_save_path):
                    tmp_idt = np.load(tmp_save_path)['IDT']
                    IDT.update({mech: tmp_idt})
                else:
                    try:
                        t0 = time.time()
                        tmp_idt, _ = yaml2idt(mechs[mech],
                                        mode = self.IDT_mode,
                                        IDT_phi = [range_phi], 
                                        IDT_T = range_T, 
                                        IDT_P = range_P, 
                                        fuel = self.fuel,
                                        oxidizer = self.oxidizer,
                                        save_path = tmp_save_path,
                                        cut_time = cut_time,
                                        **yaml2idt_kwargs
                                        )
                        IDT.update({mech: tmp_idt})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    
                    except Exception as r:
                        traceback.print_exc()       

        optimal_error = 100 * np.mean(np.abs(IDT["optimal"] - IDT["detail"]) / IDT["detail"])
        reduced_error = 100 * np.mean(np.abs(IDT["reduced"] - IDT["detail"]) / IDT["detail"])
        CompareDRO_IDT_heatmap(
            detail_data =  IDT["detail"],
            reduced_data = IDT["reduced"],
            optimal_data = IDT["optimal"],
            range_T = range_T,
            range_P = range_P,
            save_path = save_path + f"/Validation_IDT_heatmap_optim={optimal_error:1e}_reduced={reduced_error:1e}.png",
            probe_point = probe_point,
            **kwargs
        )


    def ValidationIDT_heatmap_NN(self, range_phi:float, range_T:np.ndarray, range_P, 
                              origin_IDT_condition, probe_point = None, save_path = ".", 
                              multiprocess:int = False, cut_time = 1, labels = None,
                              colors = None, yaml2idt_kwargs:dict = {},**kwargs):
        """
        20230412 增加
        绘制 IDT 的 heatmap, 只考虑 range_T 和 range_P, range_phi 只能取单点值 + compare_nn_plot 的图像合并
        实现了以下的步骤：
        1. 使用单进程/多进程计算给定机理 + detail/reduced 的 IDT
        2. 使用 Result_plot 中的 Compare 系

        params:
            range_phi, T, P: 绘图中确定的工况, 其中 phi 只能取单点值
            optim_mech: 优化后的机理文件名
            multiprocess: 是否使用多进程; 默认不使用, 若输入为整数则指定为核数
            save_path: 保存所有生成文件的文件夹名字
            probe_point: 在 APART 过程中用于训练的工况
            sharey: plt.subplots 中是否 sharey
        """
        mechs = {
            "optimal": self.optim_mech,
            "detail": self.detail_mech,
            "reduced": self.reduced_mech
        }
        IDT = {}
        if multiprocess:
            with ProcessPoolExecutor(max_workers = multiprocess) as exec:
                for mech in mechs.keys():
                    def callback(future):
                        IDT.update({mech: future.result()[0]})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    try:
                        tmp_save_path = save_path + f"/{mech}_ValidationIDT_heatmap_NN_IDT.npz"
                        future = exec.submit(
                                yaml2idt, 
                                mechs[mech],
                                mode = self.IDT_mode,
                                IDT_phi = [range_phi], 
                                IDT_T = range_T, 
                                IDT_P = range_P, 
                                fuel = self.fuel,
                                oxidizer = self.oxidizer,
                                save_path = tmp_save_path,
                                cut_time = cut_time,
                                **yaml2idt_kwargs
                                ).add_done_callback(callback)
                    except Exception as r:
                        print(f'Multiprocess error; error reason:{r}')
        else:
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_heatmap_NN_IDT.npz"
                if os.path.exists(tmp_save_path):
                    tmp_idt = np.load(tmp_save_path)['IDT']
                    IDT.update({mech: tmp_idt})
                else:
                    try:
                        t0 = time.time()
                        tmp_idt, _ = yaml2idt(mechs[mech],
                                        mode = self.IDT_mode,
                                        IDT_phi = [range_phi], 
                                        IDT_T = range_T, 
                                        IDT_P = range_P, 
                                        fuel = self.fuel,
                                        oxidizer = self.oxidizer,
                                        cut_time = cut_time,
                                        **yaml2idt_kwargs
                                        )
                        np.savez(tmp_save_path, IDT = tmp_idt)
                        IDT.update({mech: tmp_idt})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    except Exception as r:
                        print(f'Multiprocess error; error reason:{r}')            

        optimal_error = 100 * np.mean(np.abs(IDT["optimal"] - IDT["detail"]) / np.abs(IDT["detail"]))
        reduced_error = 100 * np.mean(np.abs(IDT["reduced"] - IDT["detail"]) / np.abs(IDT["detail"]))
        origin_idts = {}
        #  将 origin_IDT_condition 按照第二列升序排列
        origin_IDT_condition = origin_IDT_condition[origin_IDT_condition[:, 1].argsort()]
        # Tlist 改为 origin_IDT_condition 的第二列去重升序排列
        Tlist = np.sort(np.unique(origin_IDT_condition[:, 1]))
        philist = np.sort(np.unique(origin_IDT_condition[:, 0]))
        Plist = np.sort(np.unique(origin_IDT_condition[:, 2]))
        for mech in mechs.keys():
            tmp_save_path = save_path + f"/{mech}_ValidationIDT_heatmap_NN_origin_IDT.npz"
            # 若 tmp_save_path 存在 则直接加载
            if os.path.exists(tmp_save_path):
                tmp_idt = np.load(tmp_save_path)['IDT']
            else:
                tmp_idt, _ = yaml2idt(mechs[mech],
                            mode = self.IDT_mode,
                            IDT_condition = origin_IDT_condition,
                            fuel = self.fuel,
                            oxidizer = self.oxidizer,
                            save_path = tmp_save_path,
                            cut_time = cut_time,
                            **yaml2idt_kwargs
                            )
            origin_idts.update({mech: tmp_idt})
        # print(f"The IDT data is {IDT['detail']}, {IDT['reduced']}, {IDT['optimal']}")
        # print(f"The origin IDT data is {origin_idts['detail']}, {origin_idts['reduced']}, {origin_idts['optimal']}")
        CompareDRO_IDT_heatmap_NN(
            detail_data = IDT["detail"],
            reduced_data = IDT["reduced"],
            optimal_data = IDT["optimal"],
            range_T = range_T,
            range_P = range_P,
            save_path = save_path + f"/Validation_IDT_heatmap_NN_optim={optimal_error:1f}_reduced={reduced_error:1f}.png",
            probe_point = probe_point,
            labels = labels,
            colors = colors,
            Tlist = Tlist, philist = philist, Plist = Plist,
            origin_detail_data = np.log10(origin_idts['detail']),
            origin_reduced_data = np.log10(origin_idts['reduced']),
            origin_optimal_data = np.log10(origin_idts['optimal']),
            # **kwargs
        )


    def ValidationIDT_lineplot(self, range_phi:list, range_T:np.ndarray, range_P, save_path = ".", cut_time = 1, concat_Pressure:bool = False,
                               multiprocess:int = False, yaml2idt_kwargs = {}, probe_point = None, **kwargs):
        """
        20230327 增加
        绘制 IDT 关于初值温度倒数的图像
        实现了以下的步骤：
        1. 使用单进程/多进程计算给定机理 + detail/reduced 的 IDT
        2. 使用 Result_plot 中的 Compare 系

        params:
            range_phi, T, P: 绘图中确定的工况
            optim_mech: 优化后的机理文件名
        kwargs:
            multiprocess: 是否使用多进程; 默认不使用, 若输入为整数则指定为核数
            save_path: 保存所有生成文件的文件夹名字
            logger: logger 名字
            trigger_wc: 在 APART 过程中用于训练的工况
            sharey: plt.subplots 中是否 sharey
        """
        mechs = {
            "optimal": self.optim_mech,
            "detail": self.detail_mech,
            "reduced": self.reduced_mech
        }
        IDT = {}
        if multiprocess:
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_lineplot_IDT.npz"
                idt = yaml2idt_Mcondition(
                    chem_file = mechs[mech],
                    mode = self.IDT_mode,
                    IDT_phi = range_phi, 
                    IDT_T = range_T, 
                    IDT_P = range_P, 
                    fuel = self.fuel,
                    oxidizer = self.oxidizer,
                    cut_time = cut_time,
                    cpu_process = multiprocess,
                    save_dirpath = save_path,
                    **yaml2idt_kwargs
                )
                IDT.update({mech: idt})
                np.savez(tmp_save_path, IDT = idt)
        else:
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_lineplot_IDT.npz"
                if os.path.exists(tmp_save_path):
                    tmp_idt = np.load(tmp_save_path)['IDT']
                    IDT.update({mech: tmp_idt})
                else:
                    try:
                        t0 = time.time()
                        tmp_idt, _ = yaml2idt(mechs[mech],
                                        mode = self.IDT_mode,
                                        IDT_phi = range_phi, 
                                        IDT_T = range_T, 
                                        IDT_P = range_P, 
                                        fuel = self.fuel,
                                        oxidizer = self.oxidizer,
                                        save_path = tmp_save_path,
                                        cut_time = cut_time,
                                        **yaml2idt_kwargs
                                        )
                        IDT.update({mech: tmp_idt})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    
                    except Exception as r:
                        print(f'error reason:{r}')            
        

        CompareDRO_IDT_lineplot(
            detail_data =  IDT["detail"],
            reduced_data = IDT["reduced"],
            optimal_data = IDT["optimal"],
            range_T = range_T,
            range_P = range_P,
            range_phi = range_phi,
            save_path = save_path + f"/Validation_IDT_lineplot_concat_pressure_{concat_Pressure}.png",
            concat_Pressure = concat_Pressure,
            **kwargs
        )


    def ValidationIDT_lineplot_ProbePoint(self, range_phi:list, range_T:np.ndarray, range_P, probe_point_phi, probe_point_T, probe_point_P,
                                          save_path = ".", cut_time = 1, concat_Pressure:bool = False,
                               multiprocess:int = False, yaml2idt_kwargs = {}, **kwargs):
        """
        20231128 增加
        绘制 IDT 关于初值温度倒数的图像
        额外实现了 probe point 的加入; probe point 应该包含在 range 中

        params:
            range_phi, T, P: 绘图中确定的工况
            optim_mech: 优化后的机理文件名
            probe_point_phi, probe_point_T, probe_point_P: probe point 的 phi, T, P
        """
        mechs = {
            "optimal": self.optim_mech,
            "detail": self.detail_mech,
            "reduced": self.reduced_mech
        }
        IDT = {}
        os.makedirs(save_path, exist_ok=True)
        if multiprocess:
            multiprocess = os.cpu_count() - 1 if multiprocess == True else multiprocess
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_lineplot_IDT.npz"
                if os.path.exists(tmp_save_path):
                    tmp_idt = np.load(tmp_save_path)['IDT']
                    IDT.update({mech: tmp_idt})
                else:
                    idt = yaml2idt_Mcondition(
                        chem_file = mechs[mech],
                        mode = self.IDT_mode,
                        IDT_phi = range_phi, 
                        IDT_T = range_T, 
                        IDT_P = range_P, 
                        fuel = self.fuel,
                        oxidizer = self.oxidizer,
                        cut_time = cut_time,
                        cpu_process = multiprocess,
                        save_dirpath = save_path,
                        **yaml2idt_kwargs
                    )
                    IDT.update({mech: idt})
                    np.savez(tmp_save_path, IDT = idt)
        else:
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_lineplot_IDT.npz"
                if os.path.exists(tmp_save_path):
                    tmp_idt = np.load(tmp_save_path)['IDT']
                    IDT.update({mech: tmp_idt})
                else:
                    try:
                        t0 = time.time()
                        tmp_idt, _ = yaml2idt(mechs[mech],
                                        mode = self.IDT_mode,
                                        IDT_phi = range_phi, 
                                        IDT_T = range_T, 
                                        IDT_P = range_P, 
                                        fuel = self.fuel,
                                        oxidizer = self.oxidizer,
                                        save_path = tmp_save_path,
                                        cut_time = cut_time,
                                        **yaml2idt_kwargs
                                        )
                        IDT.update({mech: tmp_idt})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    
                    except Exception as r:
                        print(f'error reason:{r}')            
        
        print(f'keys in IDT are {IDT.keys()}')
        CompareDRO_IDT_lineplot(
            detail_data =  IDT["detail"],
            reduced_data = IDT["reduced"],
            optimal_data = IDT["optimal"],
            range_T = range_T,
            range_P = range_P,
            range_phi = range_phi,
            save_path = save_path + f"/Validation_IDT_lineplot_concat_pressure_{concat_Pressure}.png",
            concat_Pressure = concat_Pressure,
            probe_point_phi=probe_point_phi, probe_point_T=probe_point_T, probe_point_P=probe_point_P,
            **kwargs
        )


    def ValidationPSR_lineplot(self, range_phi:list, range_T:np.ndarray, range_P, fuel, oxidizer, save_path = ".", n_col = None, n_row = None, exp_factor = 2 ** (1/2),
                               ylim_increment = None, PSR_condition = None, optim_scatter_numbers = None, middle_result_psr_chem = None, extinction_time_dict = None, **kwargs):
        """
        类似于 ValidationIDT_lineplot, 这里进行 PSR 的检验; 参数内容基本等同于前者
        在使用之前需要先调用 GenTrue_RES_TIME_LIST 方法
        """
        mkdirplus(save_path)
        import pickle
        mechs = {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech,
            "middle_result": middle_result_psr_chem
        }
        PSR_condition = np.array(
            [[phi, T, P] for phi in range_phi for T in range_T for P in range_P]
        ) if PSR_condition is None else PSR_condition
        tmp_save_path = save_path + f"/detail_ValidationPSR_lineplot_PSR.pkl"
        if os.path.exists(tmp_save_path):
            with open(tmp_save_path, 'rb') as f:
                Detail_PSR_T = pickle.load(f)
            RES_TIME_LIST = Detail_PSR_T['RES_TIME_LIST']
            Detail_PSR_T = Detail_PSR_T['PSR']
        else:
            RES_TIME_LIST, Detail_PSR_T = self.GenTrue_RES_TIME_LIST(PSR_condition, save_path=save_path+'/true_res_time_list', exp_factor=exp_factor)
            tmp_dict = {
                'PSR': Detail_PSR_T,
                'RES_TIME_LIST': RES_TIME_LIST
            }
            with open(tmp_save_path, 'wb') as f:
                pickle.dump(tmp_dict, f)
            
        [print(f'len of res_time_list is {len(i)}') for i in RES_TIME_LIST]
        PSR = {"detail": Detail_PSR_T}
        for mech in mechs.keys():
            if mechs[mech] is None:
                continue
            tmp_save_path = save_path + f"/{mech}_ValidationPSR_lineplot_PSR.npz"
            try:
                t0 = time.time(); tmp_PSR = []
                for tmp_condition, tmp_res_time in zip(PSR_condition, RES_TIME_LIST):
                    tmp_psr = yaml2psr(mechs[mech],
                                    PSR_condition = tmp_condition,
                                    RES_TIME_LIST = tmp_res_time,
                                    fuel = fuel,
                                    oxidizer = oxidizer,
                                    save_path = tmp_save_path,
                                    error_tol = 0,
                                    )
                    tmp_PSR.append(tmp_psr)
                # np.savez(tmp_save_path, PSR = tmp_PSR)
                PSR.update({mech: tmp_PSR})
                print(f"Finished the {mech} PSR calculating! cost {time.time() - t0} s")
            
            except Exception as r:
                traceback.print_exc()
                PSR.update({mech: None})
        

        CompareDRO_PSR_lineplot(
             detail_psr =  PSR["detail"],
            reduced_psr = PSR["reduced"],
            optimal_psr = PSR["optimal"],
            detail_res_time = RES_TIME_LIST,
            reduced_res_time = RES_TIME_LIST,
            optimal_res_time = RES_TIME_LIST,
            PSR_condition = PSR_condition,
            save_path = save_path + "/Validation_PSR_lineplot.png",
            n_col = n_col,
            n_row = n_row,
            ylim_increment = ylim_increment,
            optim_scatter_numbers = optim_scatter_numbers,
            middle_result_psr=PSR['middle_result'] if 'middle_result' in PSR.keys() else None,
            extinction_time_dict = extinction_time_dict,
            **kwargs
        )          


    def ValidationPSR_lineplot_compare_middle_result(self, range_phi:list, range_T:np.ndarray, range_P, middle_result_chem, 
                                                     save_path = ".", n_col = None, n_row = None, exp_factor = 2 ** (1/2), ylim_increment = None, PSR_condition = None, optim_scatter_numbers = None,  **kwargs):
        """
        类似于 ValidationIDT_lineplot, 这里进行 PSR 的检验; 参数内容基本等同于前者
        在使用之前需要先调用 GenTrue_RES_TIME_LIST 方法
        """
        mechs = {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech,
            'middle_result': middle_result_chem
        }
        PSR_condition = np.array(
            [[phi, T, P] for phi in range_phi for T in range_T for P in range_P]
        ) if PSR_condition is None else PSR_condition
        RES_TIME_LIST, Detail_PSR_T = self.GenTrue_RES_TIME_LIST(PSR_condition, save_path = save_path + "/PSR_condition.npz", exp_factor=exp_factor)
        PSR = {"detail": Detail_PSR_T}
        for mech in mechs.keys():
            tmp_save_path = save_path + f"/{mech}_ValidationPSR_lineplot_PSR.npz"
            if os.path.exists(tmp_save_path):
                tmp_PSR = np.load(tmp_save_path, allow_pickle = True)['PSR']
                PSR.update({mech: tmp_PSR})
            else:
                try:
                    t0 = time.time(); tmp_PSR = []
                    for tmp_condition, tmp_res_time in zip(PSR_condition, RES_TIME_LIST):
                        tmp_psr = yaml2psr(mechs[mech],
                                        PSR_condition = tmp_condition,
                                        RES_TIME_LIST = tmp_res_time,
                                        fuel = self.fuel,
                                        oxidizer = self.oxidizer,
                                        save_path = tmp_save_path,
                                        error_tol = 0,
                                        )
                        tmp_PSR.append(tmp_psr)
                    np.savez(tmp_save_path, PSR = tmp_PSR)
                    PSR.update({mech: tmp_PSR})
                    print(f"Finished the {mech} PSR calculating! cost {time.time() - t0} s")
                
                except Exception as r:
                    traceback.print_exc()
            
        print(f'len of res_time_list is {len(RES_TIME_LIST[0])}; len of PSR dict are respectively {len(PSR["detail"][0])}, {len(PSR["reduced"][0])}, {len(PSR["optimal"][0])}, {len(PSR["middle_result"][0])}')
        CompareDRO_PSR_lineplot(
             detail_psr =  PSR["detail"],
            reduced_psr = PSR["reduced"],
            optimal_psr = PSR["optimal"],
            middle_result_psr=PSR['middle_result'],
            detail_res_time = RES_TIME_LIST,
            reduced_res_time = RES_TIME_LIST,
            optimal_res_time = RES_TIME_LIST,
            PSR_condition = PSR_condition,
            save_path = save_path + "/Validation_PSR_lineplot.png",
            n_col = n_col,
            n_row = n_row,
            ylim_increment = ylim_increment,
            optim_scatter_numbers = optim_scatter_numbers,
            **kwargs
        )          


    def ValidationFS_lineplot(self, range_phi:list = None, range_T:np.ndarray = None, range_P= None, 
                               save_path = ".", n_col = None, multiprocess = False,
                               experiment_data:np.array = None, FS_condition:np.array = None, probe_point = None, **kwargs):
        """
        类似于 ValidationIDT_lineplot, 这里进行 FS 的检验; 参数内容基本等同于前者
        20231030: 增加 experiment_data, FS_condition 参数, 用于绘制实验数据
        """
        mechs = {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech,
            "detail": self.detail_mech
        }; FS = {}
        for mech in mechs.keys():
            tmp_save_path = save_path + f"/{mech}_ValidationFS_lineplot_FS.npz"
            if mech == "detail" and experiment_data is not None:
                FS.update({mech: experiment_data})
                np.savez(tmp_save_path, FS = experiment_data)
                continue
            if os.path.exists(tmp_save_path):
                tmp_fs = np.load(tmp_save_path, allow_pickle = True)['FS']
                FS.update({mech: tmp_fs})
            else:
                if multiprocess:
                    t0 = time.time(); multiprocess = os.cpu_count() if not isinstance(multiprocess, int) else multiprocess
                    print(f"Start the {mech} yaml2FS_Mcondition calculating! cpu process: {multiprocess}")
                    tmp_fs = yaml2FS_Mcondition(
                                    mechs[mech],
                                    FS_T = range_T,
                                    FS_P = range_P, 
                                    FS_phi = range_phi,
                                    fuel = self.fuel,
                                    oxidizer = self.oxidizer,
                                    cpu_process = multiprocess,
                                    FS_condition = FS_condition
                                    )
                    np.savez(tmp_save_path, FS = tmp_fs)
                    FS.update({mech: tmp_fs})
                    print(f"Finished the {mech} FS calculating! cost {time.time() - t0} s")
                else:
                    try:
                        t0 = time.time()
                        tmp_fs = yaml2FS(
                                        mechs[mech],
                                        FS_T = range_T,
                                        FS_P = range_P, 
                                        FS_phi = range_phi,
                                        fuel = self.fuel,
                                        oxidizer = self.oxidizer,
                                        FS_condition = FS_condition
                                        )
                        np.savez(tmp_save_path, FS = tmp_fs)
                        FS.update({mech: tmp_fs})
                        print(f"Finished the {mech} FS calculating! cost {time.time() - t0} s")
                    
                    except Exception as r:
                        traceback.print_exc()
        # 计算 FS 的相对误差
        optim_relerror = np.mean(np.abs(FS["detail"] - FS["optimal"]) / FS["detail"]) * 100
        reduced_relerror = np.mean(np.abs(FS["detail"] - FS["reduced"]) / FS["detail"]) * 100
        CompareDRO_LFS_lineplot(
            detail_lfs =  FS["detail"],
            reduced_lfs = FS["reduced"],
            optimal_lfs = FS["optimal"],
            range_phi = range_phi,
            range_T = range_T,
            range_P = range_P,
            relerror = (optim_relerror, reduced_relerror),
            save_path = save_path + f"/Validation_FS_lineplot_optimrelerror={optim_relerror}_reducedrelerror={reduced_relerror}.png",
            FS_condition = FS_condition,
            probe_point = probe_point,
            n_col = n_col,
            **kwargs
        )


    def ValidationPSRconcentration_lineplot(self, concentration_res_time, range_phi:list = None, range_T:np.ndarray = None, range_P= None, PSR_concentration_condition = None,
                               save_path = ".", n_col = None, species = ['CO', 'CO2'], probe_point = None, fuel = None, oxidizer = None, diluent = {}, **kwargs):
        """
        类似于 ValidationIDT_lineplot, 这里进行 PSR_concentration 的检验; 参数内容基本等同于前者
        20231030: 增加 experiment_data, PSR_concentration_condition 参数, 用于绘制实验数据
        """
        mkdirplus(save_path)
        mechs = {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech,
            "detail": self.detail_mech
        }; PSR_concentration = {}
        PSR_concentration_condition = np.array([
            [phi, T, P] for phi in range_phi for T in range_T for P in range_P
        ]) if PSR_concentration_condition is None else PSR_concentration_condition
        
        for mech in mechs.keys():
            tmp_save_path = save_path + f"/{mech}_ValidationPSR_concentration_lineplot_PSR_concentration.npz"
            if os.path.exists(tmp_save_path):
                tmp_fs = np.load(tmp_save_path, allow_pickle = True)['PSR_concentration']
                PSR_concentration.update({mech: tmp_fs})
            else:
                t0 = time.time()
                tmp_fs = yaml2PSR_concentration(
                                mechs[mech],
                                concentration_species = species,
                                concentration_res_time=concentration_res_time,
                                PSR_condition = PSR_concentration_condition,
                                fuel = fuel,
                                oxidizer = oxidizer,
                                diluent=diluent,
                                )
                np.savez(tmp_save_path, PSR_concentration = tmp_fs)
                PSR_concentration.update({mech: tmp_fs})
                print(f"Finished the {mech} PSR_concentration calculating! cost {time.time() - t0} s")
                    
        # 计算 PSR_concentration 的相对误差
        optim_relerror = np.mean(np.abs(PSR_concentration["detail"] - PSR_concentration["optimal"]) / PSR_concentration["detail"]) * 100
        reduced_relerror = np.mean(np.abs(PSR_concentration["detail"] - PSR_concentration["reduced"]) / PSR_concentration["detail"]) * 100

        CompareDRO_PSR_concentration_lineplot(
            detail_psr_concentration =  PSR_concentration["detail"],
            reduced_psr_concentration = PSR_concentration["reduced"],
            optimal_psr_concentration = PSR_concentration["optimal"],
            PSR_concentration_condition = PSR_concentration_condition,
            save_path = save_path + f"/Validation_PSR_concentration_lineplot_optimrelerror={optim_relerror}_reducedrelerror={reduced_relerror}.png",
            probe_point = probe_point,
            n_col = n_col,
            **kwargs
        )


    def ValidationTimeTemperature(self, range_phi:float = None, range_T:np.ndarray = None, range_P = None, IDT_condition = None,
                                save_path = ".", cut_time = 1, yaml2idt_kwargs = {}, fuel = None, oxidizer = None,
                                sharey = True, **kwargs):
        """
        比较两种机理的时间温度变化曲线
        params:
            range_phi, T, P: 绘图中确定的工况
                此处 range_phi 只接受一个浮点数的输入; 不接受列表输入
            mechs: 机理文件名 dict
        kwargs:
            multiprocess: 是否使用多进程; 默认不使用, 若输入为整数则指定为核数
            save_path: 保存所有生成文件的文件夹名字
            logger: logger 名字
            trigger_wc: 在 APART 过程中用于训练的工况
        """
        mechs = {
            "optimal": self.optim_mech,
            "detail": self.detail_mech,
            "reduced": self.reduced_mech
        }; mkdirplus(save_path)
        IDT = {}
        if IDT_condition is None:
            IDT_condition = np.array(
                [
                    [range_phi, T, P] for T in range_T for P in range_P
                ]
            )
    
        for mech in mechs.keys():
            t0 = time.time()
            tmp_save_path = save_path + f"/{mech}_IDT_ValidationTimeTemperature.npz"
            if os.path.exists(tmp_save_path):
                tmp_timelist, tmp_tlist = np.load(tmp_save_path, allow_pickle = True)['timelist'], np.load(tmp_save_path, allow_pickle = True)['tlist']
                IDT.update({mech: [tmp_timelist, tmp_tlist]})
            else:
                try:
                    tmp_timelist, tmp_tlist = yaml2idtcurve(mechs[mech],
                                    IDT_phi = [range_phi], 
                                    IDT_T = range_T, 
                                    IDT_P = range_P, 
                                    IDT_condition = IDT_condition,
                                    fuel = self.fuel if fuel is None else fuel,
                                    oxidizer = self.oxidizer if oxidizer is None else oxidizer,
                                    # save_path = tmp_save_path,
                                    time_multiple = 10,
                                   cut_time = cut_time,
                                    )
                    IDT.update({mech: [tmp_timelist, tmp_tlist]})
                    print(f"Finished the {mech} IDT calculating! cost {time.time() - t0}s; The timelist and tlist are respectively {len(tmp_timelist)}, {len(tmp_tlist)}")
                
                except Exception as r:
                    print(f'Multiprocess error; error reason:{r}')      
                    print(traceback.format_exc())        

        keys = list(IDT.keys()); print(f"IDT keys are {keys}")
        from common_func.common_functions import save_pkl
        if IDT_condition is None:
            # for key in keys:
            #     IDT[key] = [IDT[key][0].reshape(len(range_P), len(range_T)), IDT[key][1].reshape(len(range_P), len(range_T))]
            phi = range_phi
            fig, axes = plt.subplots(len(range_P), len(range_T), figsize = (3.5 *len(range_T) ,3.2 *len(range_P)), dpi = 300, squeeze = False, sharey = sharey, sharex = False)
            fig.subplots_adjust(wspace = 0.2, hspace = 0.2, right = 0.9)
            for indp, p in enumerate(range_P):
                for indT, T in enumerate(range_T):
                    indice = indp * len(range_T) + indT
                    axes[indp, indT].plot( 100 * np.array(IDT['detail'][0][indice]),  IDT['detail'][1][indice], label = 'Benchmark', c = '#011627', ls = '-', lw = 4, zorder = 3)
                    axes[indp, indT].plot(100 * np.array(IDT['reduced'][0][indice]), IDT['reduced'][1][indice], c = '#54cfc3', ls = '--', lw = 3,  label = 'Original', zorder = 2)
                    axes[indp, indT].plot(100 * np.array(IDT["optimal"][0][indice]), IDT["optimal"][1][indice], ls = '-.', c = '#E71D36', lw = 3, label = 'Optimized', zorder = 1)
                    if indT == 0:
                        axes[indp, indT].set_ylabel("Temperature (K)",fontsize = 16,)
                    if indp == 0:
                        axes[indp, indT].set_xlabel(f"Time (s)",fontsize = 16,)
                    # if indT == 0:
                    #     axes[indp, indT].set_title(f"initial Pressure = {p} atm", loc='left',fontsize = 16,)
                    # 在图像右下角标注工况
                    axes[indp, indT].text(0.95, 0.05, f"$\phi = {phi:.1f}$\n$T = {T}$ "+r"$\bf{K}$"+ "\n$p = {p}$ " + r"$\bf{atm}$", 
                           transform=axes[indp, indT].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')
                    # axes[indp, indT].text(0.95, 0.05, f"$\phi = {phi:.1f}$\n$T = {T}$"+r"$\bf{K}$"+ "\n$p = {p}$" + r"$\bf{atm}$", 
                    #        transform=axes[indp, indT].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')
                    # axes[indp, indT].text(0.95, 0.05, f"$\phi = {phi:.1f}$\n$T = {T}$"+r"$\bf{K}$"+ "\n$p = {p}$" + r"$\bf{atm}$",
                    #        transform=axes[indp, indT].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')
                    axes[indp, indT].tick_params(labelsize = 16)
                    # axes[indp, indT].axhline(y = T + 400, c = 'r', ls = '--')
            lines, labels = axes[0,0].get_legend_handles_labels()
        else:
            fig, ax = plt.subplots(1, len(IDT_condition), figsize = (4.1 * len(IDT_condition), 3.6), dpi = 300,)
            for index, condition in enumerate(IDT_condition):
                phi, T, p = condition
                ax[index].plot( 100 * np.array(IDT['detail'][0][index]),  IDT['detail'][1][index], label = 'Benchmark', c = '#011627', ls = '-', lw = 4, zorder = 2)
                ax[index].plot(100 * np.array(IDT['reduced'][0][index]), IDT['reduced'][1][index], c = '#54cfc3', ls = '--', lw = 3,  label = 'Original', zorder = 1)
                ax[index].plot(100 * np.array(IDT["optimal"][0][index]), IDT["optimal"][1][index], ls = '-.', c = '#E71D36', lw = 3, label = 'Optimized', zorder = 3)
                ax[index].set_xlabel(f"Time (s)",fontsize = 16,)
                ax[index].text(0.95, 0.04, f"$\phi = {phi:.1f}$\n$T = {T}$ "+r"$\bf{K}$"+ f"\n$p = {p}$ " + r"$\bf{atm}$", 
                           transform=ax[index].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')
                # ax[index].text(0.95, 0.25, f"$T = {int(T)}$ K", 
                #            transform=ax[index].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')
                # ax[index].text(0.95, 0.45, f"$\phi = {phi:.1f}$", 
                #            transform=ax[index].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')
                ax[index].tick_params(labelsize = 16)
                # xlim_right = 100 * np.max(IDT['reduced'][0][index]) * 2
                # ax[index].set_xlim(0, xlim_right)   
            ax[0].set_ylabel("Temperature (K)",fontsize = 16,)
            lines, labels = ax[0].get_legend_handles_labels()    
        
        # show the legend on the top of x figure, and let it to be flattened
        fig.legend(lines, labels, loc='lower center', ncol = 3, bbox_to_anchor=(0.5, 0.9), fontsize = 16, frameon = False)
            
        fig.tight_layout(pad = 1.2)
        figpath = save_path + f"/{mech}_time_temperature.png"
        save_pkl((fig, ax), figpath.replace('.png', '.pkl'))
        plt.savefig(figpath, bbox_inches='tight', dpi = 300, pad_inches = 0.1)
        plt.close(fig)


    def ValidationTimeTemperature_WithIDT(self, experiment_data, IDT_condition = None, save_path = ".", cut_time = 1,  fuel = None, oxidizer = None, **kwargs):
        """
        比较两种机理的时间温度变化曲线
        params:
            range_phi, T, P: 绘图中确定的工况
                此处 range_phi 只接受一个浮点数的输入; 不接受列表输入
            mechs: 机理文件名 dict
        kwargs:
            multiprocess: 是否使用多进程; 默认不使用, 若输入为整数则指定为核数
            save_path: 保存所有生成文件的文件夹名字
            logger: logger 名字
            trigger_wc: 在 APART 过程中用于训练的工况
        """
        mechs = {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech
        }; mkdirplus(save_path)
        Curve = {}
        for mech in mechs.keys():
            t0 = time.time()
            tmp_save_path = save_path + f"/{mech}_IDT_ValidationTimeTemperature.npz"
            if os.path.exists(tmp_save_path):
                tmp_timelist, tmp_tlist = np.load(tmp_save_path, allow_pickle = True)['timelist'], np.load(tmp_save_path, allow_pickle = True)['tlist']
                Curve.update({mech: [tmp_timelist, tmp_tlist]})
                # IDT.update({mech: np.load(tmp_save_path, allow_pickle = True)['IDT']})
            else:
                try:
                    tmp_timelist, tmp_tlist = yaml2idtcurve(mechs[mech],
                                    IDT_condition = IDT_condition,
                                    fuel = self.fuel if fuel is None else fuel,
                                    oxidizer = self.oxidizer if oxidizer is None else oxidizer,
                                    # save_path = tmp_save_path,
                                    time_multiple = 10,
                                   cut_time = cut_time,
                                    )
                    Curve.update({mech: [tmp_timelist, tmp_tlist]})
                    # idt, _ = yaml2idt(
                    #     mechs[mech],
                    #     mode = self.IDT_mode,
                    #     IDT_condition = IDT_condition,
                    #     fuel = self.fuel if fuel is None else fuel,
                    #     oxidizer = self.oxidizer if oxidizer is None else oxidizer,
                    #     cut_time = cut_time,
                    #     time_multiple = 10,
                    # )
                    # IDT.update({mech: idt})
                    print(f"Finished the {mech} IDT calculating! cost {time.time() - t0}s; The timelist and tlist are respectively {len(tmp_timelist)}, {len(tmp_tlist)}")
                    # np.savez(tmp_save_path, timelist = tmp_timelist, tlist = tmp_tlist)
                except Exception as r:
                    print(f'Multiprocess error; error reason:{r}')      
                    print(traceback.format_exc())        

        from common_func.common_functions import save_pkl
        fig, ax = plt.subplots(1, len(IDT_condition), figsize = (4.1 * len(IDT_condition), 3.6), dpi = 300,)
        for index, condition in enumerate(IDT_condition):
            phi, T, p = condition
            ax[index].plot(100 * np.array(Curve['reduced'][0][index]), Curve['reduced'][1][index], label = 'Original', c = '#011627', ls = '-', lw = 4, zorder = 2)
            # ax[index].plot(100 * np.array(Curve['reduced'][0][index]), Curve['reduced'][1][index], c = '#54cfc3', ls = '--', lw = 3,  label = 'Original', zorder = 1)
            ax[index].plot(100 * np.array(Curve["optimal"][0][index]), Curve["optimal"][1][index], ls = '-', c = '#54cfc3', lw = 3, label = 'Optimized', zorder = 3)
            ax[index].set_xlabel(f"Time (s)",fontsize = 16,)
            ax[index].text(0.95, 0.04, f"$\phi = {phi:.1f}$\n$T = {T}$ "+r"$\bf{K}$"+ f"\n$p = {p}$ " + r"$\bf{atm}$", 
                       transform=ax[index].transAxes, fontsize = 16, fontweight = 'bold', verticalalignment='bottom', horizontalalignment='right')

            ax[index].tick_params(labelsize = 16)
            # 绘制 IDT 的竖线
            idt = experiment_data[index]
            ax[index].axvline(x = 100 * idt, c = '#E71D36', ls = '--', lw = 1.5, label = 'IDT', zorder = 4)
        
        ax[0].set_ylabel("Temperature (K)",fontsize = 16,)
        lines, labels = ax[0].get_legend_handles_labels()    
        
        # show the legend on the top of x figure, and let it to be flattened
        fig.legend(lines, labels, loc='lower center', ncol = 3, bbox_to_anchor=(0.5, 0.9), fontsize = 16, frameon = False)
            
        fig.tight_layout(pad = 1.2)
        figpath = save_path + f"/{mech}_time_temperature.png"
        save_pkl((fig, ax), figpath.replace('.png', '.pkl'))
        plt.savefig(figpath, bbox_inches='tight', dpi = 300, pad_inches = 0.1)
        plt.close(fig)




    def ValidationTimeSpecies(self, species:str, range_phi:float, range_T:np.ndarray, range_P,
                              logscale = False, save_path = ".", min_steps = 2000,
                               multiprocess:int = False,  x_left_lim = None, x_right_lim = None,  **kwargs):
        """
        比较两种机理的时间组分相对浓度变化曲线
        params:
            range_phi, T, P: 绘图中确定的工况
                此处 range_phi 只接受一个浮点数的输入; 不接受列表输入
            species: 组分
            mechs: 机理文件名 dict
        kwargs:
            multiprocess: 是否使用多进程; 默认不使用, 若输入为整数则指定为核数
            save_path: 保存所有生成文件的文件夹名字
            logger: logger 名字
            trigger_wc: 在 APART 过程中用于训练的工况
        """
        mkdirplus(save_path)
        mechs = {
            "optimal": self.optim_mech,
            "detail": self.detail_mech,
            "reduced": self.reduced_mech
        }
        mole = {}
        if multiprocess:
            with ProcessPoolExecutor(max_workers = multiprocess) as exec:
                for mech in mechs.keys():
                    def callback(future):
                        mole.update({mech: list(future.result())})
                        print(f"Finished the {mech} mole calculating! cost {time.time() - t0} s")
                    try:
                        tmp_save_path = save_path + f"/{mech}_ValidationTimeSpecies_{species}.npz"
                        exec.submit(
                                yaml2mole_curve, 
                                mechs[mech],
                                species = species,  
                                mole_phi = [range_phi], 
                                mole_T = range_T, 
                                mole_P = range_P, 
                                fuel = self.fuel,
                                oxidizer = self.oxidizer,
                                save_path = tmp_save_path,
                                min_steps = min_steps,
                                **kwargs
                                ).add_done_callback(callback)
                    except Exception as r:
                        print(f'Multiprocess error; error reason:{r}')
        else:
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationTimeSpecies_{species}.npz"
                if os.path.exists(tmp_save_path):
                    tmp_timelist, tmp_tlist = np.load(tmp_save_path, allow_pickle = True)['Timelist'], np.load(tmp_save_path, allow_pickle = True)['Mole']
                    mole.update({mech: [tmp_timelist, tmp_tlist]})
                else:
                    try:
                        t0 = time.time()
                        tmp_timelist, tmp_tlist = yaml2mole_curve(mechs[mech],
                                        species = species,  
                                        mole_phi = [range_phi], 
                                        mole_T = range_T, 
                                        mole_P = range_P, 
                                        fuel = self.fuel,
                                        oxidizer = self.oxidizer,
                                        save_path = tmp_save_path,
                                        min_steps = min_steps,
                                        **kwargs
                                        )
                        mole.update({mech: [tmp_timelist, tmp_tlist]})
                        print(f"Finished the {mech} mole calculating! cost {time.time() - t0} s")
                        
                    except Exception as r:
                        print(f'Multiprocess error; error reason:{r}')      
                        print(traceback.format_exc())        

        keys = list(mole.keys()); print(keys)
        for key in keys:
            mole[key] = [mole[key][0].reshape(len(range_T), len(range_P), -1), mole[key][1].reshape(len(range_T), len(range_P), -1)]


        fig, axes = plt.subplots(len(range_T), len(range_P), figsize = (4.5 *len(range_P) ,4.5 *len(range_T)), dpi = 250, squeeze = False, sharey = False, sharex = False)
        for indT, T in enumerate(range_T):
            for indp, p in enumerate(range_P):  
                detail_timelist, detail_molelist = 1000 * np.array(mole['detail'][0][indT, indp, :]), np.array(mole['detail'][1][indT, indp, :])
                reduced_timelist, reduced_molelist = 1000 * np.array(mole['reduced'][0][indT, indp, :]), np.array(mole['reduced'][1][indT, indp, :])
                optimal_timelist, optimal_molelist = 1000 * np.array(mole["optimal"][0][indT, indp, :]), np.array(mole["optimal"][1][indT, indp, :])
                # 删除 molelist 中小于 1e-5 的 index/
                try:
                    axes[indT, indp].plot(detail_timelist, detail_molelist, label = 'Benchmark', c = '#011627', ls = '-', lw = 2, zorder = 3)
                    axes[indT, indp].plot(reduced_timelist, reduced_molelist,  c = '#54cfc3', ls = '-.', lw = 2,  label = 'Original', zorder = 2)
                    axes[indT, indp].plot(optimal_timelist, optimal_molelist,   c = '#E71D36', ls = '--', lw = 2,  label = 'Optimized', zorder = 1)
                    if logscale:
                        axes[indT, indp].set(yscale = 'log', xscale = 'log')
                    tmp_x_left_lim = 0.9 * max([detail_timelist.min(), reduced_timelist.min(), optimal_timelist.min(), 1e-6]) if x_left_lim is None else x_left_lim
                    tmp_x_right_lim = min([detail_timelist.max(), reduced_timelist.max(), optimal_timelist.max(), 1]) if x_right_lim is None else x_right_lim
                    tmp_y_top_lim = 1.5 * max([detail_molelist.max(), reduced_molelist.max(), optimal_molelist.max()])
                    tmp_y_bottom_lim = 0.9 * max([detail_molelist.min(), reduced_molelist.min(), optimal_molelist.min(), 1e-8])
                    axes[indT, indp].set_xlim(left = tmp_x_left_lim, right = tmp_x_right_lim)
                    axes[indT, indp].set_ylim(bottom = tmp_y_bottom_lim, top = tmp_y_top_lim)
                    axes[indT, indp].tick_params(labelsize = 16)
                except:
                    traceback.print_exc()
                if indT == len(range_T) - 1:
                    axes[indT, indp].set_xlabel(f"Time (ms)",fontsize = 16,)
                if indT == 0:
                    axes[indT, indp].set_title(f"initial_P = {p} atm", loc='center',fontsize = 16,)
                if indp == 0:
                    axes[indT, indp].set_ylabel(f"initial_T = {T} K \n" + f"Temperature (K) in {species}",fontsize = 16,)    
        lines, labels = axes[0,0].get_legend_handles_labels()
        # show the legend on the top of x figure, and let it to be flattened
        fig.legend(lines, labels, loc='upper center', ncol = 3,  bbox_to_anchor=(0.5, 1.05), fontsize = 16, frameon = False)
       
        # fig.suptitle(f"Time - Species:{species} Curve")
        # fig.tight_layout(pad = 1.5)
        figpath = save_path + f"/{mech}_time_species_{species}.png"
        plt.subplots_adjust(wspace = 0.4, hspace = 0.3)
        plt.savefig(figpath, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)


    def ValidationTimeMultiSpecies(self, species_list, phi, T, P, logscale = False, save_path = ".", min_steps = 2000,
                               multiprocess:int = False,  x_left_lim = None, x_right_lim = None, n_row = 3, n_col = None, 
                               species_name_list = None, **kwargs):
        mkdirplus(save_path)
        mechs = {
            "optimal": self.optim_mech,
            "detail": self.detail_mech,
            "reduced": self.reduced_mech
        }
        mole = {}
        if multiprocess:
            for mech in mechs.keys():
                with ProcessPoolExecutor(max_workers = multiprocess) as exec:
                    Future_list = []; species_mole_list = []; species_time_list = []
                    for species in species_list:
                        try:
                            tmp_save_path = save_path + f"/{mech}_ValidationTimeSpecies_{species}.npz"
                            future = exec.submit(
                                    yaml2mole_curve, 
                                    mechs[mech],
                                    species = species,  
                                    mole_phi = [phi], 
                                    mole_T = [T], 
                                    mole_P = [P], 
                                    fuel = self.fuel,
                                    oxidizer = self.oxidizer,
                                    save_path = tmp_save_path,
                                    min_steps = min_steps,
                                    **kwargs
                                    )
                            Future_list.append(future)
                        except Exception as r:
                            print(f'Multiprocess error; error reason:{r}')
                wait(Future_list)
                for future in Future_list:
                    species_mole_list.append(future.result()[1])
                    species_time_list.append(future.result()[0])
                mole.update({mech: [species_time_list, species_mole_list]})
        else:
            for mech in mechs.keys():
                species_mole_list = []; species_time_list = []
                for species in species_list:
                    tmp_save_path = save_path + f"/{mech}_ValidationTimeSpecies_{species}.npz"
                    if os.path.exists(tmp_save_path):
                        tmp_timelist, tmp_tlist = np.load(tmp_save_path, allow_pickle = True)['Timelist'], np.load(tmp_save_path, allow_pickle = True)['Mole']
                        species_mole_list.append(tmp_tlist)
                        species_time_list.append(tmp_timelist)
                    else:
                        try:
                            t0 = time.time()
                            tmp_timelist, tmp_tlist = yaml2mole_curve(mechs[mech],
                                            species = species,  
                                            mole_phi = [phi], 
                                            mole_T = [T], 
                                            mole_P = [P], 
                                            fuel = self.fuel,
                                            oxidizer = self.oxidizer,
                                            save_path = tmp_save_path,
                                            min_steps = min_steps,
                                            **kwargs)
                            print(f"Finished the {mech} mole calculating! cost {time.time() - t0} s")
                            species_mole_list.append(tmp_tlist)
                            species_time_list.append(tmp_timelist)
                        except Exception as r:
                            print(f'Multiprocess error; error reason:{r}')      
                            print(traceback.format_exc())        
                mole.update({mech: [species_time_list, species_mole_list]})

        n_col = int(np.ceil(len(species_list) / n_row)) if n_col is None else n_col
        print(f'n_col, n_row: {n_col}, {n_row}')
        fig, axes = plt.subplots(n_row, n_col, figsize = (3.5 * n_col, 4.5 / 2 * n_row), dpi = 300, squeeze = False, sharey = False, sharex = False)
        # adjust subplots vspace
        fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, right=0.95, bottom=0.1, top=0.9)
        species_name_list = species_list if species_name_list is None else species_name_list
        for index, species in enumerate(species_list):
            indT, indp = index // n_col, index % n_col
            print(f"indT, indp: {indT}, {indp}")
            detail_timelist, detail_molelist = 1000 * np.array(mole['detail'][0][index]).flatten(), np.array(mole['detail'][1][index]).flatten()
            reduced_timelist, reduced_molelist = 1000 * np.array(mole['reduced'][0][index]).flatten(), np.array(mole['reduced'][1][index]).flatten()
            optimal_timelist, optimal_molelist = 1000 * np.array(mole["optimal"][0][index]).flatten(), np.array(mole["optimal"][1][index]).flatten()
            print(detail_molelist.shape, reduced_molelist.shape, optimal_molelist.shape, detail_timelist.shape, reduced_timelist.shape, optimal_timelist.shape)
            # raise ValueError
            axes[indT, indp].plot(detail_timelist, detail_molelist, label = 'Benchmark', c = '#011627', ls = '-', lw = 3.5, zorder = 3)
            axes[indT, indp].plot(reduced_timelist, reduced_molelist,  c = '#54cfc3', ls = '-.', lw = 2,  label = 'Original', zorder = 2)
            axes[indT, indp].plot(optimal_timelist, optimal_molelist,   c = '#E71D36', ls = '--', lw = 2.5,  label = 'Optimized', zorder = 1)
            if logscale:
                axes[indT, indp].set(yscale = 'log', xscale = 'log')
            tmp_x_left_lim = 0.9 * max([detail_timelist.min(), reduced_timelist.min(), optimal_timelist.min(), 1e-6]) if x_left_lim is None else x_left_lim
            tmp_x_right_lim = min([detail_timelist.max(), reduced_timelist.max(), optimal_timelist.max(), 1]) if x_right_lim is None else x_right_lim
            tmp_y_top_lim = 1.5 * max([detail_molelist.max(), reduced_molelist.max(), optimal_molelist.max()])
            tmp_y_bottom_lim = 0.9 * max([detail_molelist.min(), reduced_molelist.min(), optimal_molelist.min(), 1e-8])
            print(f'indT, indp: {indT}, {indp}; xlims: {tmp_x_left_lim}, {tmp_x_right_lim}; ylims: {tmp_y_bottom_lim}, {tmp_y_top_lim}')
            axes[indT, indp].set_xlim(left = tmp_x_left_lim, right = tmp_x_right_lim)
            axes[indT, indp].set_ylim(bottom = tmp_y_bottom_lim, top = tmp_y_top_lim)
            axes[indT, indp].tick_params(labelsize = 16)
            if indT == n_row - 1:
                axes[indT, indp].set_xlabel(f"Time (ms)",fontsize = 16,)
            if indp == 0:
                axes[indT, indp].set_ylabel(f"Concentration", fontsize = 16,) 
            # 采用科学计数法
            axes[indT, indp].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True) 
            axes[indT, indp].yaxis.offsetText.set_fontsize(15) 
            # 将标题移动到子图的右上角
            xlim = axes[indT, indp].get_xlim()
            ylim = axes[indT, indp].get_ylim()
            axes[indT, indp].text(xlim[1]*0.95, ylim[1]*0.95, species_name_list[index], ha='right', va='top', fontsize = 15)
        # 去掉没有绘制图像的子图
        for indT, indp in zip(range(n_row), range(n_col)):
            if indT * n_col + indp >= len(species_list):
                # 获取当前子图的位置
                pos = axes[indT, indp].get_position()
                fig.delaxes(axes[indT, indp])
        lines, labels = axes[0,0].get_legend_handles_labels()
        # show the legend on the top of x figure, and let it to be flattened
        legend = fig.legend(lines, labels, loc='lower center', ncol = len(labels), bbox_to_anchor=(0.5, 0.93), fontsize = 16, 
                            frameon = False, columnspacing=1.5, handlelength=1.5)
        # 获得 legend 下边缘位置
        # fig.canvas.draw()
        # renderer = fig.canvas.get_renderer()
        # legend_pos = legend.get_window_extent(renderer).get_points()[0]
        # # 在下方注释 PHI, T, P
        # fig.text(legend_pos[0], legend_pos[1] - 0.05, fr"phi = {phi}, T = {T} K, P = {P} atm", fontsize = 16)
        
        # fig.supxlabel(fr"phi = {phi}, T = {T} K, P = {P} atm", fontsize = 14)
        # fig.suptitle(f"Time - Species:{species} Curve")
        # fig.tight_layout()
        figpath = save_path + f"/Time_MultiSpecies.png"
        plt.savefig(figpath, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
    
    def Validation_Adiff_SA_SingleWC(self, result_eq_dict,  T, P, phi, delta, alpha_dict:dict = None,
                                    nums:int = None, reactions = None, minmax_regularization_flag = False, figsize = (10, 10), 
                                    save_dirpath:str = "./Adiff_SA_SingleWC", **kwargs):
        """
        绘制所有调整的反应的调整幅度与其灵敏度分析的对比图；只绘制调整幅度前 Nums 个反应和灵敏度前 nums 个的图
        绘制方式是金字塔图，左侧为灵敏度右侧为调整幅度

        params:
            mech: 需要展示的机理       
            nums: 展示的反应数量
            save_dirpath
            T, P, phi: 计算机理的条件
            delta: 调整幅度
            result_eq_dict: 机理的 A 值字典
            
        """
        # 通过 cantera 计算机理的A值灵敏度 sa_eq_dict
        ## 确定 core 下的 eq_dict
        mkdirplus(save_dirpath); t0 = time.time()
        result_Alist = eq_dict2Alist(result_eq_dict)

        ## 确认 A0 的值
        A0, A0_eq_dict = yaml_eq2A(
            self.reduced_mech,
            *list(result_eq_dict.keys()),
        )  
        nums = len(result_Alist) if nums is None else nums     
        
        ## 计算 IDT 的局部敏感度
        sa_json_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_reduced_mech_sensitivity.json'
        if os.path.exists(sa_json_path):
            sa_eq_dict = read_json_data(sa_json_path)
        else:
            sa_eq_dict = yaml2idt_sensitivity(
                self.reduced_mech,
                IDT_phi = [phi], IDT_T = [T], IDT_P = [P],
                fuel = self.fuel, oxidizer = self.oxidizer,
                delta = delta,
                save_path = sa_json_path,
            )
        sa_value = eq_dict2Alist(sa_eq_dict, benchmark_eq_dict = A0_eq_dict)
        t1 = time.time(); print(f"Finished the reduced mech sensitivity calculating! cost {t1 - t0} s")
        ## 计算 optim IDT 的局部敏感度
        optim_sa_json_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_optim_mech_sensitivity.json'
        if os.path.exists(optim_sa_json_path):
            optim_sa_eq_dict = read_json_data(optim_sa_json_path)
        else:
            optim_sa_eq_dict = yaml2idt_sensitivity(
                self.optim_mech,
                IDT_phi = [phi], IDT_T = [T], IDT_P = [P],
                fuel = self.fuel, oxidizer = self.oxidizer,
                delta = delta,
                save_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_optim_mech_sensitivity.json',
            )
        optim_sa_value = eq_dict2Alist(optim_sa_eq_dict, benchmark_eq_dict = A0_eq_dict)
        print(f"Finished the optim mech sensitivity calculating! cost {time.time() - t1} s")
        # 计算详细机理在相同情况下的敏感度
        detail_sa_json_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_detail_mech_sensitivity.json'
        if os.path.exists(detail_sa_json_path):
            detail_sa_eq_dict = read_json_data(detail_sa_json_path)
        else:
            detail_sa_eq_dict = yaml2idt_sensitivity(
                self.detail_mech,
                IDT_phi = [phi], IDT_T = [T], IDT_P = [P],
                fuel = self.fuel, oxidizer = self.oxidizer,
                delta = delta,
                specific_reactions = list(result_eq_dict.keys()),
                save_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_detail_mech_sensitivity.json',
            )
        detail_sa_value = eq_dict2Alist(detail_sa_eq_dict, benchmark_eq_dict = A0_eq_dict)
        
        
        # 查询整个APART过程后简化机理的变化值
        if alpha_dict is not None:
            alpha_list = eq_dict2Alist(alpha_dict, benchmark_eq_dict = A0_eq_dict)
            diff_A0 = (result_Alist - A0) / alpha_list   
        else: 
            diff_A0 = result_Alist - A0
        if minmax_regularization_flag:
            diff_A0 = (diff_A0 - np.amin(diff_A0)) / (np.amax(diff_A0) - np.amin(diff_A0))
            diff_A0 = 2 * (diff_A0 - 0.5)
            sa_value = 2 * (sa_value - 0.5)
            optim_sa_value = 2 * (optim_sa_value - 0.5)
            detail_sa_value = 2 * (detail_sa_value - 0.5)
            
        
        res_eq_dict = Alist2eq_dict(diff_A0, A0_eq_dict)
        sa_eq_dict = Alist2eq_dict(sa_value, A0_eq_dict)
        optim_sa_eq_dict = Alist2eq_dict(optim_sa_value, A0_eq_dict)
        detail_sa_eq_dict = Alist2eq_dict(detail_sa_value, A0_eq_dict)
        if reactions is not None:
            # 将 reactions 中的反应的灵敏度和调整幅度取出
            sa_eq_dict = {key: sa_eq_dict[key] for key in reactions}
            res_eq_dict = {key: res_eq_dict[key] for key in reactions}
            optim_sa_eq_dict = {key: optim_sa_eq_dict[key] for key in reactions}
            detail_sa_eq_dict = {key: detail_sa_eq_dict[key] for key in reactions}
        # 将 sa 与 res_eq_dict 里面非单值的项取最大值处理
        for key in res_eq_dict:
            if isinstance(sa_eq_dict[key], Iterable) or isinstance(res_eq_dict[key], Iterable) or isinstance(optim_sa_eq_dict[key], Iterable):
                sa_eq_dict[key] = np.amax(np.ravel(sa_eq_dict[key]))
                res_eq_dict[key] = np.amin(np.ravel(res_eq_dict[key]))
                optim_sa_eq_dict[key] = np.amax(np.ravel(optim_sa_eq_dict[key]))
                detail_sa_eq_dict[key] = np.amax(np.ravel(detail_sa_eq_dict[key]))
        # # 将 res_eq_dict 按照调整幅度的大小进行排序
        # res_eq_dict = dict(sorted(res_eq_dict.items(), key = lambda item: item[1], reverse = True))
        # # 保留前 nums 个反应和最后 nums 个反应
        # res_eq_dict = dict(list(res_eq_dict.items())[:nums] + list(res_eq_dict.items())[-nums:])
        # # 按照 res_eq_dict 的顺序对 sa_eq_dict 进行排序
        # sa_eq_dict = {key: sa_eq_dict[key] for key in res_eq_dict.keys()}
        # optim_sa_eq_dict = {key: optim_sa_eq_dict[key] for key in res_eq_dict.keys()}
        CompareOPTIM_SA_with_Detail(
            optimal_dict = res_eq_dict,
            SA_dict = sa_eq_dict,
            optim_SA_dict = optim_sa_eq_dict,
            detail_dict = detail_sa_eq_dict,
            reaction_num = nums,
            save_dirpath = save_dirpath,
            figname = f"RES_Optim_SingleWC",
            figsize = figsize,
            title = f"T = {T:.0f} K, P = {P:.1f} atm, phi = {phi:.1f}",
            labels = ['Optim', 'reduced_SA', 'optim_SA', 'Detail'],
            color = ['b', 'r', 'g', 'purple'],
        )
        CompareOPTIM_SA_with_Detail(
            optimal_dict = sa_eq_dict,
            SA_dict = res_eq_dict,
            optim_SA_dict = optim_sa_eq_dict,
            detail_dict = detail_sa_eq_dict,
            reaction_num = nums,
            save_dirpath = save_dirpath,
            figname = f"SA_Optim_SingleWC",
            figsize = figsize,
            title = f"T = {T:.0f} K, P = {P:.1f} atm, phi = {phi:.1f}",
            labels = ['reduced_SA', 'Optim', 'optim_SA', 'Detail'],
            color = ['r', 'b',  'g', 'purple'],
        )
        np.savez(save_dirpath + f'/APARTRes_SA_T={T:.0f}_P={P:.1f}_phi={phi:.1f}.npz', res_eq_dict = res_eq_dict, sa_eq_dict = sa_eq_dict)


    def Validation_Adiff_SA_MultiWC(self,  result_eq_dict,  IDT_condition:np.ndarray, delta, alpha_dict:dict = None,
                                    nums:int = None, reactions = None, minmax_regularization_flag = False, figsize = (12, 8), 
                                    save_dirpath:str = "./Adiff_SA_MultiWC", **kwargs):
        """
        计算优化后的机理中每个反应关于 IDT 的灵敏度与详细机理以及简化机理相互比较; 灵敏度分析的方式是局部灵敏度分析，公式为
            S = (IDT - IDT0) / delta_A
        其中，IDT0 为简化机理下的 IDT 值，delta_A 为微调的 A 值，IDT 为微调后的 IDT 值
        在针对不同反应的分析中，将保持其他的 A 值不变，只对单个反应进行微调

        params:
            optim_mech: 优化后的机理
            save_dirpath: 存储路径
            T, P, phi: 指定的温度、压力、等熵比
            delta: 微调的幅度
            reactors: 指定的反应器
            producers: 指定的生成器
            nums: 指定的反应数目
        """
        # 通过 cantera 计算机理的A值灵敏度 sa_eq_dict
        ## 确定 core 下的 eq_dict
        mkdirplus(save_dirpath)
        result_Alist = eq_dict2Alist(result_eq_dict)

        ## 确认 A0 的值
        A0, A0_eq_dict = yaml_eq2A(self.reduced_mech,*list(result_eq_dict.keys()),)  
        nums = len(result_Alist) if nums is None else nums     
        
        ## 计算 IDT 的局部敏感度
        sa_json_path = save_dirpath + '/Validation_Adiff_SA_MultiWC_reduced_mech_sensitivity.json'
        if os.path.exists(sa_json_path):
            sa_eq_dict = read_json_data(sa_json_path)
        else:
            sa_eq_dict = yaml2idt_sensitivity(
                self.reduced_mech,
                IDT_condition = IDT_condition,
                fuel = self.fuel, oxidizer = self.oxidizer,
                delta = delta,
                save_path = sa_json_path,
            )
        print('len of A0_eq_dict & sa_eq_dict:', len(A0_eq_dict), len(sa_eq_dict))
        sa_value = eq_dict2Alist(sa_eq_dict, benchmark_eq_dict = A0_eq_dict)

        ## 计算 IDT 的局部敏感度
        optim_sa_json_path = save_dirpath + '/Validation_Adiff_SA_MultiWC_optim_mech_sensitivity.json'
        if os.path.exists(optim_sa_json_path):
            optim_sa_eq_dict = read_json_data(optim_sa_json_path)
        else:
            optim_sa_eq_dict = yaml2idt_sensitivity(
                self.optim_mech,
                IDT_condition = IDT_condition,
                fuel = self.fuel, oxidizer = self.oxidizer,
                delta = delta,
                save_path = save_dirpath + '/Validation_Adiff_SA_MultiWC_optim_mech_sensitivity.json',
            )

        optim_sa_value = eq_dict2Alist(optim_sa_eq_dict, benchmark_eq_dict = A0_eq_dict)
        print(optim_sa_value)
        
        # 查询整个APART过程后简化机理的变化值
        if alpha_dict is not None:
            alpha_list = eq_dict2Alist(alpha_dict, benchmark_eq_dict = A0_eq_dict)
            diff_A0 = (result_Alist - A0) / alpha_list   
        else: 
            diff_A0 = result_Alist - A0
        if minmax_regularization_flag:
            diff_A0 = (diff_A0 - np.amin(diff_A0)) / (np.amax(diff_A0) - np.amin(diff_A0))
            diff_A0 = 2 * (diff_A0 - 0.5)
            sa_value = 2 * (sa_value - 0.5)
            optim_sa_value = 2 * (optim_sa_value - 0.5) 
        res_eq_dict = Alist2eq_dict(diff_A0, A0_eq_dict)
        sa_eq_dict = Alist2eq_dict(sa_value, A0_eq_dict)
        optim_sa_eq_dict = Alist2eq_dict(optim_sa_value, A0_eq_dict)
        
        if reactions is not None:
            # 将 reactions 中的反应的灵敏度和调整幅度取出
            sa_eq_dict = {key: sa_eq_dict[key] for key in reactions}
            res_eq_dict = {key: res_eq_dict[key] for key in reactions}
            optim_sa_eq_dict = {key: optim_sa_eq_dict[key] for key in reactions}
            
        # 将 sa 与 res_eq_dict 里面非单值的项取最大值处理
        for key in res_eq_dict:
            if isinstance(res_eq_dict[key], Iterable):
                res_eq_dict[key] = np.amin(np.ravel(res_eq_dict[key]))
                sa_eq_dict[key] = np.mean(np.ravel(sa_eq_dict[key]))
                optim_sa_eq_dict[key] = np.mean(np.ravel(optim_sa_eq_dict[key]))
     
        # # 将 res_eq_dict 按照调整幅度的大小进行排序
        # res_eq_dict = dict(sorted(res_eq_dict.items(), key = lambda item: item[1], reverse = True))
        # # 保留前 nums 个反应和最后 nums 个反应
        # res_eq_dict = dict(list(res_eq_dict.items())[:nums] + list(res_eq_dict.items())[-nums:])
        # # 按照 res_eq_dict 的顺序对 sa_eq_dict 进行排序
        # sa_eq_dict = {key: sa_eq_dict[key] for key in res_eq_dict.keys()}
        # optim_sa_eq_dict = {key: optim_sa_eq_dict[key] for key in res_eq_dict.keys()}
        CompareOPTIM_SA(
            optimal_dict = res_eq_dict,
            SA_dict = sa_eq_dict,
            optim_SA_dict = optim_sa_eq_dict,
            reaction_num = nums,
            save_dirpath = save_dirpath,
            figname = f"RES_Optim_MultiWC",
            figsize = figsize,
            labels = ['Optim', 'reduced_SA', 'optim_SA'],
            color = ['b', 'r', 'g'],
        )
        CompareOPTIM_SA(
            optimal_dict = sa_eq_dict,
            SA_dict = res_eq_dict,
            optim_SA_dict = optim_sa_eq_dict,
            reaction_num = nums,
            save_dirpath = save_dirpath,
            figname = f"SA_Optim_MultiWC",
            figsize = figsize,
            labels = ['reduced_SA', 'Optim', 'optim_SA'],
            color = ['r', 'b',  'g'],
        )
        np.savez(save_dirpath + f'/APARTRes_SA.npz', res_eq_dict = res_eq_dict, sa_eq_dict = sa_eq_dict)


    def Validation_LFS_SA_SingleWC(self, result_eq_dict, T, P, phi, delta, alpha_dict:dict = None,
                                    nums:int = None, reactions = None, minmax_regularization_flag = False, figsize = (10, 10),
                                    save_dirpath:str = "./Adiff_SA_SingleWC", cpu_process = None, **kwargs):
        """
        是 LFS 敏感度版本的 Validation_Adiff_SA_SingleWC
        绘制所有调整的反应的调整幅度与其灵敏度分析的对比图；只绘制调整幅度前 Nums 个反应和灵敏度前 nums 个的图
        绘制方式是金字塔图，左侧为灵敏度右侧为调整幅度

        params:
            mech: 需要展示的机理       
            nums: 展示的反应数量
            save_dirpath
            T, P, phi: 计算机理的条件
            delta: 调整幅度
            result_eq_dict: 机理的 A 值字典
            
        """
        # 通过 cantera 计算机理的A值灵敏度 sa_eq_dict
        ## 确定 core 下的 eq_dict
        mkdirplus(save_dirpath); result_Alist = eq_dict2Alist(result_eq_dict)

        ## 确认 A0 的值
        A0, A0_eq_dict = yaml_eq2A(self.reduced_mech, *list(result_eq_dict.keys()))  
        nums = len(result_Alist) if nums is None else nums     
        
        ## 计算 LFS 的局部敏感度
        sa_json_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_reduced_mech_sensitivity.json'
        if os.path.exists(sa_json_path):
            sa_eq_dict = read_json_data(sa_json_path)
        else:
            sa_eq_dict = yaml2LFS_sensitivity_Multiprocess(
                self.reduced_mech,
                LFS_condition = np.array([[phi, T, P]]),
                fuel = self.fuel, oxidizer = self.oxidizer,
                delta = delta,
                save_path = sa_json_path,
                multiprocess = cpu_process,
            )
        sa_value = eq_dict2Alist(sa_eq_dict, benchmark_eq_dict = A0_eq_dict)

        ## 计算 optim LFS 的局部敏感度
        optim_sa_json_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_optim_mech_sensitivity.json'
        if os.path.exists(optim_sa_json_path):
            optim_sa_eq_dict = read_json_data(optim_sa_json_path)
        else:
            optim_sa_eq_dict = yaml2LFS_sensitivity_Multiprocess(
                self.optim_mech,
                LFS_condition = np.array([[phi, T, P]]),
                fuel = self.fuel, oxidizer = self.oxidizer,
                delta = delta,
                save_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_optim_mech_sensitivity.json',
                multiprocess = cpu_process,
            )
        optim_sa_value = eq_dict2Alist(optim_sa_eq_dict, benchmark_eq_dict = A0_eq_dict)

        # # 计算详细机理在相同情况下的敏感度
        # detail_sa_json_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_detail_mech_sensitivity.json'
        # if os.path.exists(detail_sa_json_path):
        #     detail_sa_eq_dict = read_json_data(detail_sa_json_path)
        # else:
        #     detail_sa_eq_dict = yaml2LFS_sensitivity_Multiprocess(
        #         self.detail_mech,
        #         LFS_condition = np.array([phi, T, P]),
        #         fuel = self.fuel, oxidizer = self.oxidizer,
        #         delta = delta,
        #         specific_reactions = list(result_eq_dict.keys()),
        #         save_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_detail_mech_sensitivity.json',
        #         multiprocess = cpu_process,
        #     )
        # detail_sa_value = eq_dict2Alist(detail_sa_eq_dict, benchmark_eq_dict = A0_eq_dict)
        
        
        # 查询整个APART过程后简化机理的变化值
        if alpha_dict is not None:
            alpha_list = eq_dict2Alist(alpha_dict, benchmark_eq_dict = A0_eq_dict)
            diff_A0 = (result_Alist - A0) / alpha_list   
        else: 
            diff_A0 = result_Alist - A0
        if minmax_regularization_flag:
            diff_A0 = (diff_A0 - np.amin(diff_A0)) / (np.amax(diff_A0) - np.amin(diff_A0))
            diff_A0 = 2 * (diff_A0 - 0.5)
            sa_value = 2 * (sa_value - 0.5)
            optim_sa_value = 2 * (optim_sa_value - 0.5)
            # detail_sa_value = 2 * (detail_sa_value - 0.5)
            
        
        res_eq_dict = Alist2eq_dict(diff_A0, A0_eq_dict)
        sa_eq_dict = Alist2eq_dict(sa_value, A0_eq_dict)
        optim_sa_eq_dict = Alist2eq_dict(optim_sa_value, A0_eq_dict)
        # detail_sa_eq_dict = Alist2eq_dict(detail_sa_value, A0_eq_dict)
        if reactions is not None:
            # 将 reactions 中的反应的灵敏度和调整幅度取出
            sa_eq_dict = {key: sa_eq_dict[key] for key in reactions}
            res_eq_dict = {key: res_eq_dict[key] for key in reactions}
            optim_sa_eq_dict = {key: optim_sa_eq_dict[key] for key in reactions}
            # detail_sa_eq_dict = {key: detail_sa_eq_dict[key] for key in reactions}
        # 将 sa 与 res_eq_dict 里面非单值的项取最大值处理
        for key in res_eq_dict:
            if isinstance(sa_eq_dict[key], Iterable) or isinstance(res_eq_dict[key], Iterable) or isinstance(optim_sa_eq_dict[key], Iterable):
                sa_eq_dict[key] = np.amax(np.ravel(sa_eq_dict[key]))
                res_eq_dict[key] = np.amin(np.ravel(res_eq_dict[key]))
                optim_sa_eq_dict[key] = np.amax(np.ravel(optim_sa_eq_dict[key]))
                # detail_sa_eq_dict[key] = np.amax(np.ravel(detail_sa_eq_dict[key]))
        # # 将 res_eq_dict 按照调整幅度的大小进行排序
        # res_eq_dict = dict(sorted(res_eq_dict.items(), key = lambda item: item[1], reverse = True))
        # # 保留前 nums 个反应和最后 nums 个反应
        # res_eq_dict = dict(list(res_eq_dict.items())[:nums] + list(res_eq_dict.items())[-nums:])
        # # 按照 res_eq_dict 的顺序对 sa_eq_dict 进行排序
        # sa_eq_dict = {key: sa_eq_dict[key] for key in res_eq_dict.keys()}
        # optim_sa_eq_dict = {key: optim_sa_eq_dict[key] for key in res_eq_dict.keys()}
        CompareOPTIM_SA(
            optimal_dict = res_eq_dict,
            SA_dict = sa_eq_dict,
            optim_SA_dict = optim_sa_eq_dict,
            # detail_dict = detail_sa_eq_dict,
            reaction_num = nums,
            save_dirpath = save_dirpath,
            figname = f"RES_Optim_SingleWC",
            figsize = figsize,
            title = f"T = {T:.0f} K, P = {P:.1f} atm, phi = {phi:.1f}",
            labels = ['Optim', 'reduced_SA', 'optim_SA', 'Detail'],
            color = ['b', 'r', 'g', 'purple'],
        )
        CompareOPTIM_SA(
            optimal_dict = sa_eq_dict,
            SA_dict = res_eq_dict,
            optim_SA_dict = optim_sa_eq_dict,
            # detail_dict = detail_sa_eq_dict,
            reaction_num = nums,
            save_dirpath = save_dirpath,
            figname = f"SA_Optim_SingleWC",
            figsize = figsize,
            title = f"T = {T:.0f} K, P = {P:.1f} atm, phi = {phi:.1f}",
            labels = ['reduced_SA', 'Optim', 'optim_SA', 'Detail'],
            color = ['r', 'b',  'g', 'purple'],
        )
        np.savez(save_dirpath + f'/APARTRes_SA_T={T:.0f}_P={P:.1f}_phi={phi:.1f}.npz', res_eq_dict = res_eq_dict, sa_eq_dict = sa_eq_dict)

     
    def Validation_SA_Along_Circ(self, result_eq_dict_list, T, P, phi, delta, alpha_dict_list:list[dict] = None,
                                nums:int = None, reactions = None, species = None, figsize = (10, 4.5), 
                                save_dirpath:str = "./SA_Along_Circ_SingleWC", **kwargs):
        """
        绘制所有调整的反应的调整幅度与其灵敏度分析的对比图；只绘制调整幅度前 Nums 个反应和灵敏度前 nums 个的图
        绘制方式是金字塔图，左侧为灵敏度右侧为调整幅度

        params:
            mech: 需要展示的机理       
            nums: 展示的反应数量
            save_dirpath
            T, P, phi: 计算机理的条件
            delta: 调整幅度
            result_eq_dict: 机理的 A 值字典
            
        """
        mkdirplus(save_dirpath)
        iter_indice = 0
        result_sa_list = []; result_Alist_list = []
        if reactions is None: 
            _, tmp_result_dict = yaml_key2A(result_eq_dict_list[0])
            reactions = list(tmp_result_dict.keys())
            if species is not None:
                reaction = []
                for sp in species:
                    reaction.extend([reac for reac in reactions if sp in reac])
                # reaction 去重
                reactions = list(set(reaction))
        
        ## 确认 A0 的值
        A0, A0_eq_dict = yaml_eq2A(self.reduced_mech, *list(copy.copy(reactions)),)  
        nums = len(result_Alist) if nums is None else nums   
        ## 单独处理 optim_mech
        
        optim_sa_eq_dict = yaml2idt_sensitivity(
            self.optim_mech,
            IDT_phi = [phi], IDT_T = [T], IDT_P = [P],
            fuel = self.fuel, oxidizer = self.oxidizer,
            delta = delta,
            specific_reactions = reactions,
            save_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_optim_mech_sensitivity.json',
        )
        # 将 reactions 中的反应的灵敏度和调整幅度取出
        optim_sa_eq_dict = {key: optim_sa_eq_dict[key] for key in reactions}
        assert len(optim_sa_eq_dict) == len(reactions), f"len of optim_sa_eq_dict is {len(optim_sa_eq_dict)} and len of reactions is {len(reactions)}, key difference is {set(optim_sa_eq_dict.keys()) - set(reactions)}"
        # 将 sa 与 res_eq_dict 里面非单值的项取最大值处理
        for key in optim_sa_eq_dict:
            if isinstance(optim_sa_eq_dict[key], Iterable):
                print("key is ", key, "and it is in the keys()", key in optim_sa_eq_dict.keys())
                optim_sa_eq_dict[key] = np.amax(optim_sa_eq_dict[key])
                
        for result_file in result_eq_dict_list:
            tmp_save_path = save_dirpath + f'/APARTRes_SA_T={T:.0f}_P={P:.1f}_phi={phi:.1f}_iter_{iter_indice}_RES.json'
            # 通过 cantera 计算机理的A值灵敏度 sa_eq_dict
            ## 确定 core 下的 eq_dict
            print('reactions', reactions)
            if not os.path.exists(tmp_save_path):
                result_Alist, _ = yaml_eq2A(result_file, *list(copy.copy(reactions)),) 
                ## 计算 IDT 的局部敏感度
                sa_eq_dict = yaml2idt_sensitivity(
                    result_file,
                    IDT_phi = [phi], IDT_T = [T], IDT_P = [P],
                    fuel = self.fuel, oxidizer = self.oxidizer,
                    delta = delta,
                    specific_reactions = reactions,
                    save_path = save_dirpath + '/Validation_Adiff_SA_SingleWC_reduced_mech_sensitivity.json',
                )
                sa_value = eq_dict2Alist(sa_eq_dict, benchmark_eq_dict = A0_eq_dict)
                
                # 查询整个APART过程后简化机理的变化值
                if alpha_dict_list is not None:
                    alpha_dict = alpha_dict_list[iter_indice]
                    alpha_list = eq_dict2Alist(alpha_dict, benchmark_eq_dict = A0_eq_dict)
                    diff_A0 = (result_Alist - A0) / alpha_list   
                else: 
                    diff_A0 = result_Alist - A0
                
                res_eq_dict = Alist2eq_dict(diff_A0, A0_eq_dict)
                sa_eq_dict = Alist2eq_dict(sa_value, A0_eq_dict)
                assert len(res_eq_dict) == len(reactions), f"len of res_eq_dict is {len(res_eq_dict)} and len of reactions is {len(reactions)}, key difference is {set(res_eq_dict.keys()) - set(reactions)}"
                
                # 将 reactions 中的反应的灵敏度和调整幅度取出
                # sa_eq_dict = {key: sa_eq_dict[key] for key in reactions}
                # res_eq_dict = {key: res_eq_dict[key] for key in reactions}
                # 将 sa 与 res_eq_dict 里面非单值的项取最大值处理
                for key in res_eq_dict:
                    if isinstance(sa_eq_dict[key], Iterable) or isinstance(res_eq_dict[key], Iterable):
                        sa_eq_dict[key] = np.amax(np.ravel(sa_eq_dict[key]))
                        res_eq_dict[key] = np.amin(np.ravel(res_eq_dict[key]))
                # # 将 res_eq_dict 按照调整幅度的大小进行排序
                # res_eq_dict = dict(sorted(res_eq_dict.items(), key = lambda item: item[1], reverse = True))
                # # 保留前 nums 个反应和最后 nums 个反应
                # res_eq_dict = dict(list(res_eq_dict.items())[:nums] + list(res_eq_dict.items())[-nums:])
                # 按照 res_eq_dict 的顺序对 sa_eq_dict 进行排序; 
                # 绝对值处理
                sa_eq_dict = {key: sa_eq_dict[key] for key in reactions}
                res_eq_dict = {key: res_eq_dict[key] for key in reactions}
                write_json_data(save_dirpath + f'/APARTRes_SA_T={T:.0f}_P={P:.1f}_phi={phi:.1f}_iter_{iter_indice}_RES.json', res_eq_dict)
                write_json_data(save_dirpath + f'/APARTRes_SA_T={T:.0f}_P={P:.1f}_phi={phi:.1f}_iter_{iter_indice}_SA.json', sa_eq_dict)
                iter_indice += 1
                result_sa_list.append(sa_eq_dict)
                result_Alist_list.append(res_eq_dict)
            else:
                res_eq_dict = read_json_data(save_dirpath + f'/APARTRes_SA_T={T:.0f}_P={P:.1f}_phi={phi:.1f}_iter_{iter_indice}_RES.json')
                sa_eq_dict = read_json_data(save_dirpath + f'/APARTRes_SA_T={T:.0f}_P={P:.1f}_phi={phi:.1f}_iter_{iter_indice}_SA.json')
                result_sa_list.append(sa_eq_dict)
                result_Alist_list.append(res_eq_dict)  
                iter_indice += 1
        # 绘制不同反应的灵敏度关于 iter 的变化曲线
        for reac in reactions:
            try:
                print("=" * 30)
                fig, ax = plt.subplots(1, 1, figsize = figsize, dpi = 250)
                ax2 = ax.twinx()
                sa_list = []
                for ind, result_sa_dict in enumerate(result_sa_list):
                    print(ind, reac)
                    sa_list.append(result_sa_dict[reac])
                # sa_list = [result_sa_dict[reac] for result_sa_dict in result_sa_list]
                Adiff_list = []
                for ind, result_Alist_dict in enumerate(result_Alist_list):
                    print(ind, reac)
                    Adiff_list.append(result_Alist_dict[reac])
                # Adiff_list = [result_Alist_dict[reac] for result_Alist_dict in result_Alist_list]
                print(sa_list)
                ax.plot(sa_list, label = 'SA', c = '#011627', ls = '--', lw = 3, zorder = 3)
                ax2.plot(Adiff_list,  c = '#54cfc3', ls = '--', lw = 2,  label = 'Adiff', zorder = 2)
                ax.set_xlabel("iter",fontsize = 16,)
                ax.set_ylabel(f"SA",fontsize = 16,)
                ax2.set_ylabel(f"Adiff",fontsize = 16,)
                ax.tick_params(labelsize = 16)
                ax2.tick_params(labelsize = 16)
                ax.set_title(fr"Reaction: {reac} with $T={T:.0f} P={P:.1f} phi={phi:.1f}$", loc='center',fontsize = 16,)
                fig.legend(loc='upper center', ncol = 2,  bbox_to_anchor=(0.5, 1.1), fontsize = 16, frameon = False)
                fig.tight_layout(pad = 1.5)
                figpath = save_dirpath + f"/SA_Adiff_iter_{reac}.png"
                plt.savefig(figpath, bbox_inches='tight', pad_inches=0.5)
                plt.close(fig) 
            except:
                print(reac)
                print(reactions)
                print(result_Alist_list[0].keys())
                print(reac in list(result_Alist_list[0].keys()))
                # 找到 reaction 和 result_Alist_list[0].keys() 中差异的部分
                print(set(result_Alist_list[0].keys()) - set(reactions))
                print(traceback.format_exc())

        
    def ValidationCoreReactionsTraffic(self, time_series:list,
                                        IDT_condition:np.ndarray,
                                        reaction:list = None,
                                        save_path:str = None,):
        """
        绘制在特定 time_series 下的中心反应区反应的速率图片, 使用不同的颜色来指示该通路的流量大小

        最后形成 detail, reduced, mech(optim) 三张一体的流量对比图
        params:
            mech; core: 指示中心反应区的物种
            time_series: 指示在那几个时间节点进行分析
        """
        phi, T, P = IDT_condition
        _, fp,         _, _, time = yaml2FP(self.optim_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer)
        _, fp_detail,  _, _, time_detail = yaml2FP(self.detail_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer)
        _, fp_reduced, _, _,  time_reduced = yaml2FP(self.reduced_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer)
        # 获得 reaction 对应的那些反应的编号
        gas = ct.Solution(self.reduced_mech); detail_gas = ct.Solution(self.detail_mech)
        if reaction is None:
            reaction = gas.reaction_equations()
        indexes = [gas.reaction_equations().index(reaction) for reaction in reaction]
        fp = fp[:, indexes]; fp_reduced = fp_reduced[:, indexes]
        indexes = [detail_gas.reaction_equations().index(reaction) for reaction in reaction]
        fp_detail = fp_detail[:, indexes]

        # 绘制反应流量变化曲线
        fig, ax = plt.subplots(1,3, figsize = (16, 6), dpi = 250)
        for k in range(fp.shape[1]):
            try:
                ax[0].semilogx(time_detail, fp_detail[:,k])
                ax[1].semilogx(time_reduced, fp_reduced[:,k])
                ax[2].semilogx(time, fp[:,k])
            except:
                pass
        # 设置所有子图的 x lim
        ax[0].set_xlim(left = 0.9 * max([time_detail.min(), 1e-5]), right = np.amin([time_detail.max(), 10]))
        ax[1].set_xlim(left = 0.9 * max([time_reduced.min(), 1e-5]), right = np.amin([time_reduced.max(), 10]))
        ax[2].set_xlim(left = 0.9 * max([time.min(), 1e-5]), right = np.amin([time.max(), 10]))
        ax[0].set_ylim(bottom = max([fp_detail.min(), 1e-5]), top = 1.1 * min([fp_detail.max(), 10]))
        ax[1].set_ylim(bottom = max([fp_reduced.min(), 1e-5]), top = 1.1 * min([fp_reduced.max(), 10]))
        ax[2].set_ylim(bottom = max([fp.min(), 1e-5]), top = 1.1 * min([fp.max(), 10]))
        ax[0].set_title("Detail"); ax[1].set_title("Reduced"); ax[2].set_title("Optim")
        ax[0].set_xlabel("time"); ax[1].set_xlabel("time"); ax[2].set_xlabel("time")
        ax[0].set_xlabel("forward progress"); ax[1].set_xlabel("forward progress"); ax[2].set_xlabel("forward progress")
        # 设置图例 under the x axis
        fig.legend(reaction, loc = "upper center", bbox_to_anchor = (0.5, 0.98), ncol = int(np.ceil(len(reaction) / 2)), fontsize = 8, frameon = False)
        fig.suptitle(f"Net Production Rates in IDT Condition: T = {T:.0f} K, P = {P:.1f} atm, phi = {phi:.1f}", y = 0)
        fig.tight_layout(pad = 1.5)
        if not save_path is None:
            fig.savefig(save_path, pad_inches = 0.5)
        np.set_printoptions(suppress = True, precision = 2)
        print(fp)

    
    def ValidationCoreNetProductionRates(self,
                                        time_series:list,
                                        IDT_condition:np.ndarray,
                                        species:list = None,
                                        save_path:str = None,
                                        **kwargs):
        """
        绘制在特定 time_series 下的中心反应区反应的速率图片, 使用不同的颜色来指示该通路的流量大小
        最后形成 detail, reduced, optim_mech 三张一体的流量对比图
        params:
            optim_mech & detail_mech & reduced_mech: the path of the mechanism
            species
            time_series: the time series to plot
            T & P & phi: the temperature, pressure and equivalence ratio
            fuel & oxidizer: the fuel and oxidizer
            save_path: the path to save the figure
            **kwargs: the parameters of plot
        return:
            None
        """
        phi, T, P = IDT_condition
        _, _, _, fp, time = yaml2FP(self.optim_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer)
        _, _, _, fp_detail, time_detail = yaml2FP(self.detail_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer)
        _, _, _, fp_reduced, time_reduced = yaml2FP(self.reduced_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer)
        fp = np.array(fp); fp_detail = np.array(fp_detail); fp_reduced = np.array(fp_reduced)
        # 获得 species 对应的那些反应的编号
        gas = ct.Solution(self.reduced_mech); detail_gas = ct.Solution(self.detail_mech)
        if species is None:
            species = [sp.name for sp in gas.species()]
        indexes = [gas.species_index(species) for species in species]
        fp = fp[:, indexes]; fp_reduced = fp_reduced[:, indexes]
        indexes = [detail_gas.species_index(species) for species in species]
        fp_detail = fp_detail[:, indexes]

        # 绘制反应流量变化曲线
        fig, ax = plt.subplots(1,3, figsize = (16, 6), dpi = 250)
        for k in range(fp.shape[1]):
            try:
                ax[0].plot(time_detail, fp_detail[:,k])
                ax[1].plot(time_reduced, fp_reduced[:,k])
                ax[2].plot(time, fp[:,k])
            except:
                pass
        # 设置所有子图的 x lim
        ax[0].set_xlim(left = np.amin(time_detail), right = np.amax(time_detail))
        ax[1].set_xlim(left = np.amin(time_reduced), right = np.amax(time_reduced))
        ax[2].set_xlim(left = np.amin(time), right = np.amax(time))
        ax[0].set_ylim(bottom = np.amin(fp_detail), top = 1.1 * np.amax(fp_detail))
        ax[1].set_ylim(bottom = np.amin(fp_reduced), top = 1.1 * np.amax(fp_reduced))
        ax[2].set_ylim(bottom = np.amin(fp), top = 1.1 * np.amax(fp))
        print(fp)
        ax[0].set_title("Detail"); ax[1].set_title("Reduced"); ax[2].set_title("Optim")
        ax[0].set_xlabel("time"); ax[1].set_xlabel("time"); ax[2].set_xlabel("time")
        ax[0].set_xlabel("forward progress"); ax[1].set_xlabel("forward progress"); ax[2].set_xlabel("forward progress")
        # 设置图例 under the x axis
        fig.legend(species, loc = "upper center", bbox_to_anchor = (0.5, 1.05), ncol = len(species), fontsize = 8)
        fig.suptitle(f"Net Production Rates in IDT Condition: T = {T:.0f} K, P = {P:.1f} atm, phi = {phi:.1f}")
        if not save_path is None:
            fig.savefig(save_path)
        np.set_printoptions(suppress = True, precision = 2)
        print(fp)


    def ValidationTotalForwardFlux(self, time_series:list,
                                        IDT_condition:np.ndarray,
                                        reactions:list = None,
                                        save_path:str = None,
                                        cut_time = 1, need_percentage = False):
        """
        绘制在特定 time_series 下的中总的正反应通量, 使用不同的颜色来指示该通路的流量大小

        最后形成 detail, reduced, mech(optim) 三张一体的流量对比图
        params:
            mech; core: 指示中心反应区的物种
            time_series: 指示在那几个时间节点进行分析
        """
        mkdirplus(save_path)
        phi, T, P = IDT_condition
        if not os.path.exists(save_path + f'/TotalForwardFlux_optim.json') or not os.path.exists(save_path + f'/TotalForwardFlux_detail.json') or not os.path.exists(save_path + f'/TotalForwardFlux_reduced.json'):
            fp = yaml2total_forward_flux(self.optim_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer, cut_time = cut_time)
            fp_detail = yaml2total_forward_flux(self.detail_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer, cut_time = cut_time)
            fp_reduced = yaml2total_forward_flux(self.reduced_mech, time_series, T, P, phi, fuel = self.fuel, oxidizer = self.oxidizer, cut_time = cut_time)
            print(f'len of fp is {len(fp)}; len of fp_detail is {len(fp_detail)}; len of fp_reduced is {len(fp_reduced)}')
            # 获得 reaction 对应的那些反应的编号
            gas = ct.Solution(self.reduced_mech); detail_gas = ct.Solution(self.detail_mech)
            if reactions is None:
                reactions = gas.reaction_equations()
            indexes = [gas.reaction_equations().index(reaction) for reaction in reactions]
            fp = fp[indexes]; fp_reduced = fp_reduced[indexes]
            # indexes = [detail_gas.reaction_equations().index(reaction) for reaction in reactions]
            # fp_detail = fp_detail[indexes]
            fp_dict = {key: value for key, value in zip(reactions, fp)}
            fp_detail_dict = {key: value for key, value in zip(detail_gas.reaction_equations(), fp_detail)}
            fp_reduced_dict = {key: value for key, value in zip(reactions, fp_reduced)}
            print(f'len of fp_dict is {len(fp_dict)}; len of fp_detail_dict is {len(fp_detail_dict)}; len of fp_reduced_dict is {len(fp_reduced_dict)}')
            write_json_data(save_path + f'/TotalForwardFlux_optim.json', fp_dict)
            write_json_data(save_path + f'/TotalForwardFlux_detail.json', fp_detail_dict)
            write_json_data(save_path + f'/TotalForwardFlux_reduced.json', fp_reduced_dict)
        detailed_flowgraph = read_json_data(save_path +"/TotalForwardFlux_detail.json")
        reduced_flowgraph = read_json_data(save_path +"/TotalForwardFlux_reduced.json")
        optim_flowgraph = read_json_data(save_path +"/TotalForwardFlux_optim.json")
        detail_gas = ct.Solution(self.detail_mech)
        optim_gas = ct.Solution(self.optim_mech)
        # def get_diff_species_Q(gas, dicts, species_nums):
        #     Qmatrix = np.zeros((species_nums, species_nums))
        #     for reac in gas.reactions():
        #         for reactant in reac.reactants:
        #             for product in reac.products:
        #                 Qmatrix[gas.species_names.index(reactant), gas.species_names.index(product)] += dicts[reac.equation]
        #                 # if reactant == 'C3H6' and product == 'C2H3':
        #                 #     print(reac.equation, Qmatrix[gas.species_names.index(reactant), gas.species_names.index(product)])
        #                 # elif reactant == 'C2H3' and product == 'C3H6':
        #                 #     print(reac.equation, Qmatrix[gas.species_names.index(reactant), gas.species_names.index(product)])
        #     # diff 是 Qmatrix 的上半部分 - 上下半部分
        #     diff = Qmatrix - Qmatrix.T
        #     # diff 下半部分mask = 0
        #     diff = np.triu(diff)
        #     # diff 分解成正部分和负部分
        #     diff_plus = np.where(diff < 0, 0, diff); diff_minus = np.where(diff >= 0, 0, diff)
        #     print(diff_minus)
        #     # diff_plus + (-diff_minus).T
        #     diff = diff_plus + (-diff_minus).T
        #     return diff
        # all_species_Qmatrix_detail  = get_diff_species_Q(detail_gas, detailed_flowgraph, detail_gas.n_species)
        # all_species_Qmatrix_reduced = get_diff_species_Q(optim_gas, reduced_flowgraph, optim_gas.n_species)
        # all_species_Qmatrix_optimal = get_diff_species_Q(optim_gas, optim_flowgraph, optim_gas.n_species)
                
        # # 对每一列负值取 0 并除以这一列的和
        # all_species_Qmatrix_optimal_noneg = np.where(all_species_Qmatrix_optimal < 1e-8, 0, all_species_Qmatrix_optimal)
        # all_species_Qmatrix_detail_noneg = np.where(all_species_Qmatrix_detail < 1e-8, 0, all_species_Qmatrix_detail)
        # all_species_Qmatrix_reduced_noneg = np.where(all_species_Qmatrix_reduced < 1e-8, 0, all_species_Qmatrix_reduced)
        # # 检查是否有负值
        # print(np.sum(all_species_Qmatrix_optimal_noneg < 0), f'shape of all_species_Qmatrix_optimal_noneg: {all_species_Qmatrix_optimal_noneg.shape}')
        # print(np.sum(all_species_Qmatrix_detail_noneg < 0), f'shape of all_species_Qmatrix_detail_noneg: {all_species_Qmatrix_detail_noneg.shape}')
        # print(np.sum(all_species_Qmatrix_reduced_noneg < 0), f'shape of all_species_Qmatrix_reduced_noneg: {all_species_Qmatrix_reduced_noneg.shape}')

        # # 矩阵的每一行除以对应行的和；为保证不出现 0 / 0 的情况，除数取 maxmum(1e-8, row_sum)
        # all_species_Qmatrix_optimal_weight = all_species_Qmatrix_optimal_noneg / np.maximum(np.sum(all_species_Qmatrix_optimal_noneg, axis=1), 1e-8)[:, None]
        # all_species_Qmatrix_detail_weight = all_species_Qmatrix_detail_noneg / np.maximum(np.sum(all_species_Qmatrix_detail_noneg, axis=1), 1e-8)[:, None]
        # all_species_Qmatrix_reduced_weight = all_species_Qmatrix_reduced_noneg / np.maximum(np.sum(all_species_Qmatrix_reduced_noneg, axis=1), 1e-8)[:, None]
        # # 验证 weight 的所有行和为 1
        # print(np.sum(all_species_Qmatrix_optimal_weight, axis=1))
        # print(np.sum(all_species_Qmatrix_detail_weight, axis=1))
        # print(np.sum(all_species_Qmatrix_reduced_weight, axis=1))

        # np.savez(save_path + '/APARTRes_TotalForwardFlux_Qmatrix.npz',
        #         all_species_Qmatrix_optimal_weight=all_species_Qmatrix_optimal_weight,
        #         all_species_Qmatrix_detail_weight=all_species_Qmatrix_detail_weight,
        #         all_species_Qmatrix_reduced_weight=all_species_Qmatrix_reduced_weight,
        #         all_species_Qmatrix_detail=all_species_Qmatrix_detail,
        #         all_species_Qmatrix_reduced=all_species_Qmatrix_reduced,
        #         all_species_Qmatrix_optimal=all_species_Qmatrix_optimal,
        #         )
        
        # 计算关于反应的权重向量
        ## 将互相为逆反应的反应抵消
        def examine_inverse_reactions(reac1, reac2):
            reac1_reactants = reac1.reactants; reac1_products = reac1.products
            reac2_reactants = reac2.reactants; reac2_products = reac2.products
            if set(reac1_reactants) == set(reac2_products) and set(reac1_products) == set(reac2_reactants):
                return True
            else:
                return False
        print(detail_gas.reactions()[0], detail_gas.reactions()[1])
        print(examine_inverse_reactions(detail_gas.reactions()[0], detail_gas.reactions()[1]))
        def get_diff_reaction_Q(gas, dicts, need_percentage = False):
            # 将上面的 dicts 定义写成嵌套循环
            origin_dicts = copy.deepcopy(dicts)
            for reac1 in gas.reactions():
                for reac2 in gas.reactions():
                    if reac1.equation in dicts and reac2.equation in dicts and examine_inverse_reactions(reac1, reac2):
                        # 比较 reac1 和 reac2 的正反应通量
                        if origin_dicts[reac1.equation] >= origin_dicts[reac2.equation]:
                            dicts[reac1.equation] = dicts[reac1.equation] - dicts[reac2.equation]
                            dicts.pop(reac2.equation)
                        else:
                            dicts[reac2.equation] = dicts[reac2.equation] - dicts[reac1.equation]
                            dicts.pop(reac1.equation)
            # dicts 只保留正值的 value
            dicts = {key: value for key, value in dicts.items() if value > 0}
            print(len(dicts))
            reaction_Qdict = {
                species: {} for species in gas.species_names
            }
            for reac in gas.reactions():
                if reac.equation in dicts.keys():
                    for reactant in reac.reactants:
                            reaction_Qdict[reactant][reac.equation] = dicts[reac.equation]
            
            # 对于 reaction_Qdict 中的每一个 species 计算与权重和的比例
            if need_percentage:
                for species in reaction_Qdict.keys():
                    total_weight = sum(reaction_Qdict[species].values())
                    for reac in reaction_Qdict[species].keys():
                        reaction_Qdict[species][reac] /= total_weight
            return reaction_Qdict
        detailed_flowgraph = read_json_data(save_path +"/TotalForwardFlux_detail.json")
        reduced_flowgraph = read_json_data(save_path +"/TotalForwardFlux_reduced.json")
        optim_flowgraph = read_json_data(save_path +"/TotalForwardFlux_optim.json")
        all_species_Qmatrix_detail  = get_diff_reaction_Q(detail_gas, detailed_flowgraph, need_percentage = need_percentage)
        all_species_Qmatrix_reduced = get_diff_reaction_Q(optim_gas, reduced_flowgraph, need_percentage = need_percentage)
        all_species_Qmatrix_optimal = get_diff_reaction_Q(optim_gas, optim_flowgraph, need_percentage = need_percentage)
        write_json_data(save_path + f'/APARTRes_TotalForwardFlux_ReactionWeight_optim.json', all_species_Qmatrix_optimal)
        write_json_data(save_path + f'/APARTRes_TotalForwardFlux_ReactionWeight_detail.json', all_species_Qmatrix_detail)
        write_json_data(save_path + f'/APARTRes_TotalForwardFlux_ReactionWeight_reduced.json', all_species_Qmatrix_reduced)
        # return fp_dict, fp_detail_dict, fp_reduced_dict


    def ValidationFatherSampleCompareWithNN(self, father_sample_alist:np.ndarray, reduced_mech:str,
                                            father_sample_idt = None, IDT_condition = None, fuel = None, oxidizer = None,
                                            save_dirpath = '.', **kwargs):
        """
        father sample 版本的 compare_with_nn 函数
        绘制一张 father sample 中 IDT 分布与真实值和简化值的对比图，类似于 compare_with_nn 函数，但是将优化值使用箱线图代替。
        params:
            father_sample_alist: father sample 的 a 值列表
            detail_mech: 详细机理的 mech 文件名
            reduced_mech: 简化机理的 mech 文件名
            father_sample_idt: father sample 的 IDT 值，如果为 None 则自动计算
            IDT_condition: IDT 计算条件
            fuel: 燃料
            oxidizers: 氧化剂
            **kwargs: 其他参数
        return:
            None
        """
        IDT_condition = self.IDT_condition if IDT_condition is None else IDT_condition
        fuel = self.fuel if fuel is None else fuel; oxidizer = self.oxidizer if oxidizer is None else oxidizer
        if father_sample_idt is None:
            if not os.path.exists(save_dirpath + "/FatherSampleIDT.npy"):
                yaml_save_path = mkdirplus(save_dirpath + "/tmp_yaml"); father_sample_idt = []
                for i, alist in enumerate(father_sample_alist):
                    tmp_yaml_save_path = f"{yaml_save_path}/tmp_{i}.yaml"
                    Adict2yaml(reduced_mech, tmp_yaml_save_path, self.eq_dict, alist)
                    idt, _ = yaml2idt(
                        tmp_yaml_save_path,
                        mode = self.IDT_mode,
                        cut_time = kwargs.get("cut_time", 100),
                        IDT_condition = IDT_condition,
                        fuel = fuel,
                        oxidizer = oxidizer,
                    )
                    father_sample_idt.append(idt)
                father_sample_idt = np.array(father_sample_idt)
                np.save(save_dirpath + "/FatherSampleIDT.npy", father_sample_idt)
            else:
                father_sample_idt = np.load(save_dirpath + "/FatherSampleIDT.npy")
            

        FatherSampleCompareWithNN(
            true_idt_data = np.log10(self.true_idt_data),
            reduced_idt_data = np.log10(self.reduced_idt_data),
            father_sample_data = np.log10(father_sample_idt),
            save_path = save_dirpath + "/FatherSampleCompareWithNN.png",
            IDT_condition = IDT_condition,
            **kwargs
        )


    def ValidationIDT_flow_graph(self, optim_mech, IDT_condition, element = 'C', save_dirpath = ".",
                              label_threshold = 0.1, show_details = True, path_threshold = 0.03, scale = -1,
                                save_step = 200, steps_uplimit = 15, sim_step = None, T_upper_bound = 3000, second_diff_thres = 0.1):
        """
        绘制反应流量图，包含 detail_mech 和 reduced_mech 与 optimal_mech 的对比
        params:
            ``label_threshold``: 控制路径强度超过多少才显示标签
            ``show_details``: 控制路径是否显示细节（每个化学反应的贡献）
            ``path_threshold``: 控制路径强度超过多少才在图中显示
            ``scale``: 控制显示数值的尺度，图中数值=路径强度/scale，-1表示使用最大强度进行归一化
            ``save_step``: 每演化save_step步画一次图
        """
        import subprocess
        if sim_step is not None: sim_time = sim_step
        original_save_step = save_step
        for mech in [self.detail_mech, self.reduced_mech, optim_mech]:
            save_step = original_save_step; save_count = 0
            sim_time = sim_step
            save_path = mkdirplus(save_dirpath + f"/IDTflowdiagram_{mech.split('/')[-1].split('.')[0]}")
            gas = ct.Solution(mech)
            phi, T, p = IDT_condition
            gas.TP = T, p * ct.one_atm
            gas.set_equivalence_ratio(phi, fuel = self.fuel, oxidizer = self.oxidizer)
            r = ct.IdealGasReactor(gas)
            net = ct.ReactorNet([r])
            T_ini = r.T; count = 0; Tlist = [T_ini]; time_list = [0]; count_t_list = []; HRRlist = [0]; count_HRR_list = []
            diff_T = [0, 0]
            temperature_plot_title = f"Temperature evolution Temperature: {T:.2f}K, Pressure: {p:.2f}atm, phi: {phi:.2f}"

            while abs(T_ini - r.T) < T_upper_bound and save_count <= steps_uplimit:
                if sim_step is not None:
                    net.advance(sim_time)
                    sim_time += sim_step
                else:
                    sim_time = net.step()
                T = r.T
                if (count % save_step == 0) or count == 1:
                    print(f"Mech {mech.split('/')[-1].split('.')[0]}: Temperature: {T:.2f}K, Time: {sim_time:.2e}s at count {count}; diff_T: {np.diff(diff_T)[-1]}; save_step = {save_step}")
                    print(f"Net rate of progress: {r.kinetics.net_rates_of_progress}; forward rate of progress: {r.kinetics.forward_rates_of_progress}; reverse rate of progress: {r.kinetics.reverse_rates_of_progress}")
                    print(f"count: {count}, T: {T}, save_step: {save_step}; save_count = {save_count}")
                    diagram = ct.ReactionPathDiagram(gas, element)
                    diagram.title = f'Ini:{phi}_{int(T)}K_{p}atm Element: {element} Time: {sim_time:.2e}s delta_T: {T-T_ini:.2f}'
                    
                    # 画图显示设置
                    diagram.label_threshold = label_threshold   
                    diagram.show_details = show_details     
                    diagram.threshold = path_threshold        
                    diagram.scale = scale      
                    dot_file = f'IDTpath{count}_Tem_{int(T)}K.dot'
                    img_file = f"IDTimg{count}_Tem_{int(T)}K.png"
                    img_file = save_path + "/" + img_file
                    dot_file = save_path + "/" + dot_file   
                    diagram.write_dot(dot_file)
                    count_t_list.append([sim_time, T]); count_HRR_list.append([sim_time, r.kinetics.heat_release_rate])
                    save_count += 1
                    try:    
                        subprocess.run('dot {0} -Tpng -o{1} -Gdpi=200'.format(dot_file, img_file).split(" "))
                    except:
                        pass
                count += 1; Tlist.append(T); time_list.append(sim_time); HRRlist.append(r.kinetics.heat_release_rate)
                # 在温度变化非常大的地方，减少 save_step; 在温度变化非常小的地方，增加 save_step
                if 2 * second_diff_thres > np.abs(np.diff(diff_T)[-1]) > second_diff_thres:
                    save_step = 15
                elif np.abs(np.diff(diff_T)[-1]) > 2 * second_diff_thres:
                    save_step = 5
                elif np.abs(np.diff(diff_T)[-1]) < second_diff_thres:
                    save_step = original_save_step
                diff_T.append(Tlist[-1] - Tlist[-2])
            # 绘制 Tlist 关于时间的曲线
            fig, ax = plt.subplots(1,2, figsize = (10, 4), dpi = 250)
            ax[0].plot(time_list, Tlist)
            ax[1].plot(time_list, HRRlist)
            # 绘制 count_t_list 以散点图的形式
            count_t_list = np.array(count_t_list); count_HRR_list = np.array(count_HRR_list)
            ax[0].scatter(count_t_list[:,0], count_t_list[:,1], s = 10, color = 'red')
            ax[1].scatter(count_t_list[:,0], count_HRR_list[:,1], s = 10, color = 'red')
            ax[0].set_xlabel("time"); ax[0].set_ylabel("Temperature")
            ax[1].set_xlabel("time"); ax[1].set_ylabel("HRR")
            ax[1].set_yscale("log")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 1))
            # ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 1))
            fig.suptitle(temperature_plot_title)
            fig.savefig(save_path + "/Temperature_evolution.png")
            

    def ValidationPSR_flow_graph(self, optim_mech, PSR_condition, residence_time_list, element = 'C', save_dirpath = ".",
                              label_threshold = 0.1, show_details = True, path_threshold = 0.02, scale = -1,
                                save_step = 20):
        """
        绘制反应流量图，包含 detail_mech 和 reduced_mech 与 optimal_mech 的对比
        params:
            ``label_threshold``: 控制路径强度超过多少才显示标签
            ``show_details``: 控制路径是否显示细节（每个化学反应的贡献）
            ``path_threshold``: 控制路径强度超过多少才在图中显示
            ``scale``: 控制显示数值的尺度，图中数值=路径强度/scale，-1表示使用最大强度进行归一化
            ``save_step``: 每演化save_step步画一次图
        """
        import subprocess
        def mdot(t):
            return combustor.mass/residence_time
        for mech in [self.detail_mech, self.reduced_mech, optim_mech]:
            save_path = mkdirplus(save_dirpath + f"/PSRflowdiagram/{mech.split('/')[-1].split('.')[0]}")
            gas = ct.Solution(mech)
            phi, T, p = PSR_condition
            gas.TP = T, p * ct.one_atm
            gas.set_equivalence_ratio(phi, fuel = self.fuel, oxidizer = self.oxidizer)
            # r = ct.IdealGasReactor(gas)
            # net = ct.ReactorNet([r])
            # T_ini = r.T; count = 0  

            inlet = ct.Reservoir(gas)               # 进气口预混箱
            gas.equilibrate('HP')
            combustor = ct.IdealGasReactor(gas, volume = 1.0)
            exhaust = ct.Reservoir(gas)

            inlet_mfc = ct.MassFlowController(upstream = inlet, downstream = combustor, mdot = mdot)
            outlet_mfc = ct.PressureController(upstream = combustor, downstream = exhaust, master=inlet_mfc, K=0.01)
            sim = ct.ReactorNet([combustor])

            
            for residence_time in residence_time_list:
                Reduced_T_list = [T]; count = 0
                sim.set_initial_time(0); diff_T = 10000
                while diff_T > 10:
                    time = sim.step()
                    T = combustor.T
                    diff_T = T - Reduced_T_list[-1]
                    if (count % save_step == 0) or count == 1:
                        diagram = ct.ReactionPathDiagram(gas, element)
                        diagram.title = f'Ini:{phi}_{T}K_{p}atm_Restime_{residence_time} Element: {element} Time: {time:.2e}s T: {combustor.T:.2f}'
                        # 画图显示设置
                        diagram.label_threshold = label_threshold   
                        diagram.show_details = show_details     
                        diagram.threshold = path_threshold        
                        diagram.scale = scale      
                        dot_file = f'PSRpath{count}_Tem_{int(T)}K_Restime_{residence_time}.dot'
                        img_file = f"PSRimg{count}_Tem_{int(T)}K_Restime_{residence_time}.png"
                        img_file = save_path + "/" + img_file
                        dot_file = save_path + "/" + dot_file   
                        diagram.write_dot(dot_file)
                    Reduced_T_list.append(combustor.T)
                    count += 1
        return Reduced_T_list


    def ValidationTimeTemperature_WithExperiment(self, experiment_dataset, save_path = None, cut_time = 1,
                               multiprocess:int = False, sharey = True, col_x = 1, col_y = None,LoadIDTExpData_kwargs = {}, **kwargs):
        """
        比较简化机理和真实机理的时间温度变化曲线; 同时指示出实验数据
        params:
            save_path: 保存所有生成文件的文件夹名字
            cut_time: 计算 idt 的时间截断
            multiprocess: 是否使用多进程计算 IDT
            sharey: 是否共享 y 轴
        return:
            None
        """
        experiment_IDT_condition, IDT_fuel, IDT_oxidizer, true_experiment_idt_data = LoadIDTExpDataWithoutMode(experiment_dataset, raw_data = False, **LoadIDTExpData_kwargs)
        mechs = {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech
        }
        IDT = {}
        if save_path is None: save_path = self.vlidpath 
        if multiprocess:
            with ProcessPoolExecutor(max_workers = multiprocess) as exec:
                for mech in mechs.keys():
                    def callback(future):
                        IDT.update({mech: list(future.result())})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    try:
                        tmp_save_path = save_path + f"/{mech}_IDT_ValidationTimeTemperature.npz"
                        future = exec.submit(
                                yaml2idtcurve, 
                                mechs[mech],
                                IDT_condition = experiment_IDT_condition,
                                fuel = IDT_fuel,
                                oxidizer = IDT_oxidizer,
                                save_path = tmp_save_path,
                                cut_time = cut_time,
                                ).add_done_callback(callback)
                    except Exception as r:
                        print(f'Multiprocess error; error reason:{r}')
        else:
            for mech in mechs.keys():
                t0 = time.time()
                tmp_save_path = save_path + f"/{mech}_IDT_ValidationTimeTemperature.npz"
                if os.path.exists(tmp_save_path):
                    tmp_timelist, tmp_tlist = np.load(tmp_save_path, allow_pickle = True)['timelist'], np.load(tmp_save_path, allow_pickle = True)['tlist']
                    IDT.update({mech: [tmp_timelist, tmp_tlist]})
                else:
                    try:
                        tmp_timelist, tmp_tlist = yaml2idtcurve(mechs[mech],
                                        IDT_condition = experiment_IDT_condition,
                                        fuel = IDT_fuel,
                                        oxidizer = IDT_oxidizer,
                                        save_path = tmp_save_path,
                                        cut_time = cut_time,
                                        )
                        IDT.update({mech: [tmp_timelist, tmp_tlist]})
                        print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s")
                    
                    except Exception as r:
                        print(f'error reason:{r}')      
                        print(traceback.format_exc())        

        col_y = int(np.ceil(len(experiment_IDT_condition) / col_x)) if col_y is None else col_y
        fig, axes = plt.subplots(col_x, col_y, figsize = (4.5 *col_y ,4.5 *col_x), dpi = 250, squeeze = False, sharey = sharey, sharex = False)
        index = 0
        for i in range(col_x):
            for j in range(col_y):
                try:
                    phi, T, P = experiment_IDT_condition[index]
                    reduced_IDT_timelist = 1000 * np.array(IDT['reduced'][0][index])
                    reduced_IDT_tlist = IDT['reduced'][1][index]
                    optimal_IDT_timelist = 1000 * np.array(IDT["optimal"][0][index])
                    optimal_IDT_tlist = IDT["optimal"][1][index]
                    axes[i, j].scatter(1000 * true_experiment_idt_data[index], T + 400, label = 'Benchmark', c = '#011627', marker = '*', zorder = 3)
                    axes[i, j].plot(reduced_IDT_timelist, reduced_IDT_tlist, c = '#54cfc3', ls = '--', lw = 2,  label = 'Original', zorder = 1)
                    axes[i, j].plot(optimal_IDT_timelist, optimal_IDT_tlist,  c =  '#E71D36', ls = '-.',  label = 'Optimized', zorder = 2)
                    axes[i, j].set_xlabel(f"$\phi = {phi}$, $T = {T}$ K, $p = {P}$ atm\n" + f"fuel:{self.fuel[index]}, oxidizer:{self.oxidizer[index]}", fontsize = 14)
                    xlim = [0.5 * min(np.min(reduced_IDT_timelist), np.min(optimal_IDT_timelist), 1000 * true_experiment_idt_data[index]), 
                    2 * max(np.max(reduced_IDT_timelist), np.max(optimal_IDT_timelist), 1000 * true_experiment_idt_data[index])] 
                    axes[i,j].set(xlim = xlim, )  # 确定子图的范围
                    index += 1
                    # axes[indT, indp].axhline(y = T + 400, c = 'r', ls = '--')
                except:
                    traceback.print_exc()
        lines, labels = axes[0,0].get_legend_handles_labels()
        # show the legend on the top of x figure, and let it to be flattened
        fig.legend(lines, labels, loc='lower center', ncol = 3, bbox_to_anchor=(0.5, 0.97), fontsize = 16, frameon = False)
        
        fig.suptitle("Time - Temperature Curve")
        fig.tight_layout()
        figpath = save_path + f"/{mech}_time_temperature.png"
        plt.savefig(figpath)
        plt.close(fig)


    def ValidationContourDiagram_Alist_Iteration(self, optim_path, sample_range, true_idt_data, 
                          max_key, second_max_key, dir0, A1_range = (-1, 1), A2_range = (-1, 1), original_A0 = (0, 0),
                          one_dim_sample_size = 200, cpu_nums = 140, block_trace = True, Hull_trace = False, 
                          save_path = None, **kwargs):
        """
        绘制 alist 的迭代轨迹; 轨迹绘制在关于两个坐标下的 Contour Diagram 上，以 max_key 为横坐标、以 second_max_key 为纵坐标，IDT 的误差作为颜色，展示
        在我们的迭代过程中，alist 的变化是逐渐搜索到最优解的，因此我们可以通过绘制 alist 的轨迹来观察搜索过程
        params:
            optim_path: 优化路径
            sample_range: alist 的取值范围
            true_idt_data: 真实的 IDT 数据
            max_key: 作为横坐标的 key
            second_max_key: 作为纵坐标的 key
            dir0: 保存轨迹的文件夹
            A1_range: 第一个坐标的取值范围
            A2_range: 第二个坐标的取值范围
            one_dim_sample_size: 一维采样点的数量
            cpu_nums: cpu 核心数量
        """
        def getRectangleAllVertices(vertices, center_point = [-1, 1]):
            """
            Vertices: 一个二维数组，每一行是一个正方形的四个顶点; 要求顶点顺序为类似于
                [[0,0], [0,1], [1,0], [1,1]]
            """
            intersections = []
            for i, sq1 in enumerate(vertices):
                new_vertices = vertices.copy(); new_vertices.pop(i)
                for vertice in sq1:
                    # 判断 vertice 是否在 new_vertices 构成的正方形中
                    x, y = vertice
                    for vertice2 in new_vertices:
                        x1, y1 = vertice2[0]
                        x4, y4 = vertice2[-1]
                        if x1 <= x <= x4 and y1 <= y <= y4:
                            intersections.extend([
                            [x, y1],
                            [x, y4],
                            [x1, y],
                            [x4, y]
                            ])
            # 删除重复的点
            intersections = np.array(list(set([tuple(t) for t in intersections])))
            intersections = np.r_[intersections, np.array(vertices).reshape((-1, 2))]
            # 删除被覆盖在内部的点
            inner_intersections_index = []
            for ind, vec in  enumerate(intersections):
                diff_intersections = np.delete(intersections, ind, axis = 0) - vec
                # 如果 diff_intersections 中有四个点的坐标符号严格为 [+, +], [-, -], [+, -], [-, +]，则 vec 为内部点
                pp = 0; pm = 0; mp = 0; mm = 0
                for dfvec in diff_intersections:
                    if   (dfvec[0] >= 0 and dfvec[1] >= 0):# or (dfvec[0] > 0 and dfvec[1] >= 0):
                        pp += 1
                    elif (dfvec[0] >= 0 and dfvec[1] <= 0):# or (dfvec[0] > 0 and dfvec[1] <= 0):
                        pm += 1
                    elif (dfvec[0] <= 0 and dfvec[1] >= 0):# or (dfvec[0] < 0 and dfvec[1] >= 0):
                        mp += 1
                    elif (dfvec[0] <= 0 and dfvec[1] <= 0):# or (dfvec[0] < 0 and dfvec[1] <= 0):
                        mm += 1
                if pp > 0 and pm > 0 and mp > 0 and mm > 0:
                    inner_intersections_index.append(ind)
            # 从 intersections 剔除 inner_intersections
            intersections = np.delete(intersections, inner_intersections_index, axis = 0)
            # 计算 intersections 中的质心
            intersections_mean = np.mean(intersections, axis = 0) if center_point is None else center_point
            # 计算 intersections 中每个坐标与质心的连线与 x 轴的夹角; 取值范围为 [0, 2pi]
            intersections_angle = np.arctan2(intersections[:,1] - intersections_mean[1], intersections[:,0] - intersections_mean[0])
            # 按照 intersections_angle 对 intersections 进行排序
            intersections = intersections[np.argsort(intersections_angle)]
            return intersections
        palist, peqdict = yaml_key2A(self.detail_mech)
        mkdirplus(dir0 + "/tmp_yamls"); save_path = dir0 + "/IDT_coutuor_plot.png" if save_path is None else save_path
        if not os.path.exists(dir0 + "/tmp_yamls_idt_data.npz"):
            all_A = []
            for maxkey_value in np.linspace(A1_range[0], A1_range[1], one_dim_sample_size):
                for secondmax_value in np.linspace(A2_range[0], A2_range[1], one_dim_sample_size):
                    all_A.append([maxkey_value, secondmax_value])
            # 使用 pool 进行并行计算, 使用 yaml2idt 进行计算
            pool = Pool(cpu_nums)
            for i, (maxkey_value, secondmax_value) in enumerate(all_A):
                pool.apply_async(ValidationContourDiagram_idt_mps, 
                                kwds = dict(
                                    dir0 = dir0,
                                    peqdict = peqdict,
                                    max_key = max_key,
                                    second_max_key = second_max_key,
                                    maxkey_value = maxkey_value,
                                    secondmax_value = secondmax_value,
                                    index = i,
                                    chem_file = dir0 + f"/tmp_yamls/reduced_chem_{i}.yaml",
                                    setup_file = dir0 + "/settings/setup.yaml",
                                    mode = 0,
                                    cut_time = 10,
                                    save_path = dir0 + f"/tmp_yamls/reduced_chem_{i}.npy",
                                    original_chem_path = self.reduced_mech,
                                )
                                )
            pool.close(); pool.join()
            

            # 读取所有的 reduced_chem_{i}.npz 的 idt 数据
            idt_data = []
            for ii in range(len(all_A)):
                filename = dir0 + f"/tmp_yamls/reduced_chem_{ii}.npy"
                if os.path.exists(filename):
                    idt_data.append(np.load(dir0 + f"/tmp_yamls/reduced_chem_{ii}.npy").tolist())
                else:
                    idt_data.append(np.zeros_like(true_idt_data).tolist())

            idt_data = np.log10(idt_data) - np.log10(true_idt_data)
            np.savez(dir0 + "/tmp_yamls_idt_data.npz", IDT = idt_data, all_A = all_A)

        idt_data = np.load(dir0 + "/tmp_yamls_idt_data.npz")['IDT']
        all_A = np.load(dir0 + "/tmp_yamls_idt_data.npz")['all_A']

        loss_average = np.linalg.norm(idt_data, axis = 1).reshape((one_dim_sample_size, one_dim_sample_size))
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        fig, ax = plt.subplots(1, 1, figsize = (6, 6), dpi = 400, sharey = False, sharex = False)
        ax.set_xlabel(max_key,)
        ax.set_ylabel(second_max_key,)
        ax.set_title("IDT")
        ax.set_xlim(left = A1_range[0], right = A1_range[1])
        ax.set_ylim(bottom = A2_range[0], top = A2_range[1]) 

        ax.contourf(all_A1, all_A2, loss_average, 100, cmap = cm01)
        cbar = fig.colorbar(ax.contourf(all_A1, all_A2, loss_average, 100, cmap = cm01), ax = ax)
        cbar.set_label("IDT")
        
        Rectangle_points = []
        for i in range(len(optim_path) - 1):
            rectangle_point = optim_path[i]
            Rectangle_points.append([
                [rectangle_point[0] - sample_range[i], rectangle_point[1] - sample_range[i]],
                [rectangle_point[0] - sample_range[i], rectangle_point[1] + sample_range[i]],
                [rectangle_point[0] + sample_range[i], rectangle_point[1] - sample_range[i]],
                [rectangle_point[0] + sample_range[i], rectangle_point[1] + sample_range[i]],
                ])
        hull_points = getRectangleAllVertices(Rectangle_points)
        palist, peqdict = yaml_key2A(self.reduced_mech)
        # 在等高线图上根据 optim_path 绘制优化路径，其中 optim_path 是一个二维数组，每一行是一个优化点
        from matplotlib.patches import Rectangle, Polygon
        for i in range(len(optim_path) - 1):
            x = [optim_path[i][0], optim_path[i + 1][0]]
            y = [optim_path[i][1], optim_path[i + 1][1]]
            ax.plot(x, y, color = 'white', linewidth = 2)
            ax.scatter(x, y, edgecolors = 'white', facecolors='none', s = 30)
            if block_trace:
                patch1 = Rectangle((x[0] - sample_range[i], y[0] - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'white', fill = False, linewidth = 1, linestyle='--')
                ax.add_patch(patch1)
            # ax.annotate(f"STEP{i + 1}", xy = (x[1], y[1]), xytext = (x[1] + 0.01, y[1] + 0.01), color = "black", fontsize = 10)
            # ax[1].plot(x, y, color = 'black', linewidth = 0.5)
            # ax[1].annotate("", xy = (x[1], y[1]), xytext = (x[0], y[0]), arrowprops = dict(arrowstyle = "->", linewidth = 0.5))
        if hull_points is not None and Hull_trace:
            ax.scatter(hull_points[:,0], hull_points[:,1], edgecolors = 'yellow', facecolors='none', s = 30)
            patch2 = Polygon(hull_points, color = 'white', fill = False, linewidth = 1, linestyle='--')
            ax.add_patch(patch2)
        fig.tight_layout()
        fig.savefig(save_path)


    def ValidationContourDiagram_Network_Iteration(self, network_path_list, true_idt_data, optim_path, sample_range,  max_key, second_max_key, dir0, 
                                                   A1_range = (-1, 1), A2_range = (-1, 1), peqdict = None, original_A0 = np.array([0, 0]),
                          one_dim_sample_size = 200, cpu_nums = 140, save_dirpath = None, load_best_dnn_args = None, **kwargs):
        """
        绘制 真实情况下IDT的 contour diagram，以及网络预测的 contour diagram，窥视网络预测的效果如何
        params:
            network_path_list: 网络的路径列表
            true_idt_data: 真实的 IDT 数据
            max_key: 作为横坐标的 key
            second_max_key: 作为纵坐标的 key
            dir0: 保存轨迹的文件夹
            A1_range: 第一个坐标的取值范围
            A2_range: 第二个坐标的取值范围
            one_dim_sample_size: 一维采样点的数量
            cpu_nums: cpu 核心数量
            load_best_dnn_args: 加载模型时 load_best_dnn 的参数设置
        """
        if peqdict is None:
            _, peqdict = yaml_key2A(self.detail_mech)
        palist = eq_dict2Alist(peqdict)
        mkdirplus(dir0 + "/tmp_yamls"); mkdirplus(save_dirpath)
        all_A = []
        for maxkey_value in np.linspace(A1_range[0], A1_range[1], one_dim_sample_size):
            for secondmax_value in np.linspace(A2_range[0], A2_range[1], one_dim_sample_size):
                all_A.append([maxkey_value, secondmax_value])
        all_A = np.array(all_A)

        if not os.path.exists(dir0 + "/tmp_yamls_idt_data.npz"):
            # 使用 pool 进行并行计算, 使用 yaml2idt 进行计算
            pool = Pool(cpu_nums)
            for i, (maxkey_value, secondmax_value) in enumerate(all_A):
                pool.apply_async(ValidationContourDiagram_idt_mps, 
                                kwds = dict(
                                    dir0 = dir0,
                                    peqdict = peqdict,
                                    max_key = max_key,
                                    second_max_key = second_max_key,
                                    maxkey_value = maxkey_value,
                                    secondmax_value = secondmax_value,
                                    index = i,
                                    chem_file = dir0 + f"/tmp_yamls/reduced_chem_{i}.yaml",
                                    setup_file = dir0 + "/settings/setup.yaml",
                                    mode = 0,
                                    cut_time = 10,
                                    save_path = dir0 + f"/tmp_yamls/reduced_chem_{i}.npy",
                                    original_chem_path = self.reduced_mech,
                                )
                                )
            pool.close(); pool.join()
            

            # 读取所有的 reduced_chem_{i}.npz 的 idt 数据
            idt_data = []; print("save loss_average")
            for ii in range(len(all_A)):
                filename = dir0 + f"/tmp_yamls/reduced_chem_{ii}.npy"
                if os.path.exists(filename):
                    idt_data.append(np.load(dir0 + f"/tmp_yamls/reduced_chem_{ii}.npy").tolist())
                else:
                    idt_data.append(np.zeros_like(true_idt_data).tolist())

            idt_data = np.log10(idt_data) - np.log10(true_idt_data)
            np.savez(dir0 + "/tmp_yamls_idt_data.npz", IDT = idt_data, all_A = all_A)

        idt_data = np.load(dir0 + "/tmp_yamls_idt_data.npz")['IDT']
        all_A = np.load(dir0 + "/tmp_yamls_idt_data.npz")['all_A']
        # 将 palist 复制成 all_A.shape[0]
        palist = np.tile(palist, (all_A.shape[0], 1)).reshape((all_A.shape[0], -1))
        # 将 eq_dict 里面 max_key 和 second_max_key 的值改为 0 和 0.01
        eq_dict = copy.deepcopy(peqdict); eq_dict[max_key] = 0; eq_dict[second_max_key] = 0.01
        # eq_dict 转化为 Alist, 确定 Alist 中 max_key 和 second_max_key 的位置
        tmp_Alist = eq_dict2Alist(eq_dict)
        tmp_Alist = tmp_Alist.tolist()
        ## 找到 max_key 和 second_max_key 在 Alist 中的位置
        maxkey_index = tmp_Alist.index(0); secondmax_index = tmp_Alist.index(0.01)
        palist[:, maxkey_index] = palist[:, maxkey_index] + all_A[:, 0]; palist[:, secondmax_index] = palist[:, secondmax_index] + all_A[:, 1]
        
        loss_average = np.linalg.norm(idt_data, axis = 1).reshape((one_dim_sample_size, one_dim_sample_size))
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        for i, network_path in enumerate(network_path_list):
            optim_point_x = optim_path[i + 1][0]
            optim_point_y = optim_path[i + 1][1]
            save_path = save_dirpath + f"/IDT_ValidationContourDiagram_circ={i}.png"
            load_best_dnn_args['model_pth_path'] = network_path
            network = load_best_dnn(**load_best_dnn_args)
            fig, ax = plt.subplots(1, 3, figsize = (16, 6), dpi = 400, sharey = False, sharex = False)
            ax[0].set_xlabel(max_key,)
            ax[0].set_ylabel(second_max_key,)
            ax[0].set_title("IDT")
            ax[0].set_xlim(left = A1_range[0], right = A1_range[1])
            ax[0].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
            cbar_baseline = ax[0].contourf(all_A1, all_A2, loss_average, 100, cmap = cm01)
            patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'white', fill = False, linewidth = 1, linestyle='--')
            ax[0].add_patch(patch2)
            cbar = fig.colorbar(cbar_baseline, ax = ax[0])
            cbar.set_label("IDT")
            
            pred_idt = network.forward_IDT(torch.tensor(palist, dtype = torch.float32)).detach().numpy()
            pred_idt = pred_idt - np.log10(true_idt_data)
            loss_average_diff = np.linalg.norm(pred_idt - idt_data, axis = 1).reshape((one_dim_sample_size, one_dim_sample_size))
            pred_idt = np.linalg.norm(pred_idt, axis = 1).reshape((one_dim_sample_size, one_dim_sample_size)) 
            ax[1].set_xlabel(max_key,)
            ax[1].set_ylabel(second_max_key,)
            ax[1].set_title("Pred IDT")
            ax[1].set_xlim(left = A1_range[0], right = A1_range[1])
            ax[1].set_ylim(bottom = A2_range[0], top = A2_range[1])  
            ax[1].contourf(all_A1, all_A2, pred_idt, 100, cmap = cm01)
            patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'white', fill = False, linewidth = 1, linestyle='--')
            ax[1].add_patch(patch2)
            cbar = fig.colorbar(cbar_baseline, ax = ax[1])
            cbar.set_label("Pred IDT")

            ax[2].set_xlabel(max_key,)
            ax[2].set_ylabel(second_max_key,)
            ax[2].set_title("Pred IDT - IDT")
            ax[2].set_xlim(left = A1_range[0], right = A1_range[1])
            ax[2].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
            ax[2].contourf(all_A1, all_A2, loss_average_diff, 100, cmap = cm01)
            patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'white', fill = False, linewidth = 1, linestyle='--')
            ax[2].add_patch(patch2)
            cbar = fig.colorbar(cbar_baseline, ax = ax[2])
            cbar.set_label("Pred IDT")
            fig.tight_layout()
            fig.savefig(save_path)


    def ValidationContourDiagram_Samples_Iteration(self, apart_data_path_list, optim_path, sample_range, 
                                                   true_idt_data, max_key, second_max_key, dir0, 
                                                   A1_range = (-1, 1), A2_range = (-1, 1), peqdict = None,
                                                    one_dim_sample_size = 200, cpu_nums = 140, save_dirpath = None, 
                                                    original_A0 = None, **kwargs):
        """
        绘制 真实情况下IDT的 contour diagram，以及每次通过筛选的 samples 分布，窥视网络筛选的效果如何
        params:
            apart_data_path_list: 筛选出来的 samples 的路径
            true_idt_data: 真实的 IDT 数据
            max_key: 作为横坐标的 key
            second_max_key: 作为纵坐标的 key
            dir0: 保存轨迹的文件夹
            A1_range: 第一个坐标的取值范围
            A2_range: 第二个坐标的取值范围
            one_dim_sample_size: 一维采样点的数量
            cpu_nums: cpu 核心数量
            load_best_dnn_args: 加载模型时 load_best_dnn 的参数设置
        """
        if peqdict is None:
            _, peqdict = yaml_key2A(self.detail_mech)
        mkdirplus(dir0 + "/tmp_yamls"); mkdirplus(save_dirpath)
        if not os.path.exists(dir0 + "/tmp_yamls_idt_data.npz"):
            
            all_A = []
            for maxkey_value in np.linspace(A1_range[0], A1_range[1], one_dim_sample_size):
                for secondmax_value in np.linspace(A2_range[0], A2_range[1], one_dim_sample_size):
                    all_A.append([maxkey_value, secondmax_value])
            # 使用 pool 进行并行计算, 使用 yaml2idt 进行计算
            pool = Pool(cpu_nums)
            for i, (maxkey_value, secondmax_value) in enumerate(all_A):
                
                pool.apply_async(ValidationContourDiagram_idt_mps, 
                                kwds = dict(
                                    dir0 = dir0,
                                    peqdict = peqdict,
                                    max_key = max_key,
                                    second_max_key = second_max_key,
                                    maxkey_value = maxkey_value,
                                    secondmax_value = secondmax_value,
                                    index = i,
                                    chem_file = dir0 + f"/tmp_yamls/reduced_chem_{i}.yaml",
                                    setup_file = dir0 + "/settings/setup.yaml",
                                    mode = 0,
                                    cut_time = 10,
                                    save_path = dir0 + f"/tmp_yamls/reduced_chem_{i}.npy",
                                    original_chem_path = self.reduced_mech,
                                )
                                )
            pool.close(); pool.join()
            

            # 读取所有的 reduced_chem_{i}.npz 的 idt 数据
            idt_data = []
            for ii in range(len(all_A)):
                filename = dir0 + f"/tmp_yamls/reduced_chem_{ii}.npy"
                if os.path.exists(filename):
                    idt_data.append(np.load(dir0 + f"/tmp_yamls/reduced_chem_{ii}.npy").tolist())
                else:
                    idt_data.append(np.zeros_like(true_idt_data).tolist())

            idt_data = np.log10(idt_data) - np.log10(true_idt_data)
            np.savez(dir0 + "/tmp_yamls_idt_data.npz", IDT = idt_data, all_A = all_A)

        idt_data = np.load(dir0 + "/tmp_yamls_idt_data.npz")['IDT']
        all_A = np.load(dir0 + "/tmp_yamls_idt_data.npz")['all_A']

        loss_average = np.linalg.norm(idt_data, axis = 1).reshape((one_dim_sample_size, one_dim_sample_size))
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 将 eq_dict 里面 max_key 和 second_max_key 的值改为 0 和 0.01
        eq_dict = copy.deepcopy(peqdict); eq_dict[max_key] = 0; eq_dict[second_max_key] = 0.01
        # eq_dict 转化为 Alist, 确定 Alist 中 max_key 和 second_max_key 的位置
        tmp_Alist = eq_dict2Alist(eq_dict)
        tmp_Alist = tmp_Alist.tolist()
        ## 找到 max_key 和 second_max_key 在 Alist 中的位置
        maxkey_index = tmp_Alist.index(0); secondmax_index = tmp_Alist.index(0.01)
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        for i, apart_data_path in enumerate(apart_data_path_list):
            apart_data = np.load(apart_data_path)['Alist']
            apart_data_x = original_A0[0] - apart_data[:, maxkey_index]
            apart_data_y = original_A0[1] - apart_data[:, secondmax_index]
            optim_point_x = optim_path[i + 1][0]
            optim_point_y = optim_path[i + 1][1]
            save_path = save_dirpath + f"/IDT_ValidationContourDiagram_circ={i}.png"
            
            fig, ax = plt.subplots(1, 2, figsize = (12, 6), dpi = 400, sharey = False, sharex = False)
            ax[0].set_xlabel(max_key,)
            ax[0].set_ylabel(second_max_key,)
            ax[0].set_title("IDT")
            ax[0].set_xlim(left = A1_range[0], right = A1_range[1])
            ax[0].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
            ax[0].contourf(all_A1, all_A2, loss_average, 100, cmap = cm01)
            cbar = fig.colorbar(ax[0].contourf(all_A1, all_A2, loss_average, 100, cmap = cm01), ax = ax[0])
            cbar.set_label("IDT")
            sample_range_x_delta = np.amax(apart_data_x) - np.amin(apart_data_x)
            sample_range_y_delta = np.amax(apart_data_y) - np.amin(apart_data_y)
            patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1)
            patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
            ax[0].add_patch(patch1)
            ax[0].add_patch(patch2)
            # 在 ax[0] 上标注 optim_path[i]
            ax[0].scatter(optim_path[i + 1][0], optim_path[i + 1][1], c = 'white', marker = '*', zorder = 3)
            # density, x_edge, y_edge = np.histogram2d(apart_data[:,0], apart_data[:,1], bins=one_dim_sample_size) 
            # # 绘制热力图
            ax[1].hist2d(apart_data_x, apart_data_y, bins=one_dim_sample_size, cmap=cm01)
            ax[1].set_xlabel(max_key,)
            ax[1].set_ylabel(second_max_key,)
            ax[1].set_xlim(left = A1_range[0], right = A1_range[1])
            ax[1].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
            patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1, linestyle='--')
            patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
            ax[1].add_patch(patch1)
            ax[1].add_patch(patch2)
            # 在 ax[1] 上标注 optim_path[i]
            ax[1].scatter(optim_path[i + 1][0], optim_path[i + 1][1], c = 'white', marker = '*', zorder = 3)
            # ax[1].imshow(density.T, extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            #             cmap='hot', origin='lower')
            fig.tight_layout()
            fig.savefig(save_path)


    def ValidationContourDiagram_Alist_Iteration_original(self, optim_path, sample_range, true_idt_data, true_hrr_data, weight_array,
                          max_key, second_max_key, dir0, A1_range = (-1, 1), A2_range = (-1, 1), original_A0 = (0, 0),
                          one_dim_sample_size = 200, cpu_nums = 140, block_trace = True, Hull_trace = False, 
                          save_path = None, **kwargs):
        """
        绘制 alist 的迭代轨迹; 轨迹绘制在关于两个坐标下的 Contour Diagram 上，以 max_key 为横坐标、以 second_max_key 为纵坐标，IDT 的误差作为颜色，展示
        在我们的迭代过程中，alist 的变化是逐渐搜索到最优解的，因此我们可以通过绘制 alist 的轨迹来观察搜索过程
        params:
            optim_path: 优化路径
            sample_range: alist 的取值范围
            true_idt_data: 真实的 IDT 数据
            max_key: 作为横坐标的 key
            second_max_key: 作为纵坐标的 key
            dir0: 保存轨迹的文件夹
            A1_range: 第一个坐标的取值范围
            A2_range: 第二个坐标的取值范围
            one_dim_sample_size: 一维采样点的数量
            cpu_nums: cpu 核心数量
        """
        def getRectangleAllVertices(vertices, center_point = [-1, 1]):
            """
            Vertices: 一个二维数组，每一行是一个正方形的四个顶点; 要求顶点顺序为类似于
                [[0,0], [0,1], [1,0], [1,1]]
            """
            intersections = []
            for i, sq1 in enumerate(vertices):
                new_vertices = vertices.copy(); new_vertices.pop(i)
                for vertice in sq1:
                    # 判断 vertice 是否在 new_vertices 构成的正方形中
                    x, y = vertice
                    for vertice2 in new_vertices:
                        x1, y1 = vertice2[0]
                        x4, y4 = vertice2[-1]
                        if x1 <= x <= x4 and y1 <= y <= y4:
                            intersections.extend([
                            [x, y1],
                            [x, y4],
                            [x1, y],
                            [x4, y]
                            ])
            # 删除重复的点
            intersections = np.array(list(set([tuple(t) for t in intersections])))
            intersections = np.r_[intersections, np.array(vertices).reshape((-1, 2))]
            # 删除被覆盖在内部的点
            inner_intersections_index = []
            for ind, vec in  enumerate(intersections):
                diff_intersections = np.delete(intersections, ind, axis = 0) - vec
                # 如果 diff_intersections 中有四个点的坐标符号严格为 [+, +], [-, -], [+, -], [-, +]，则 vec 为内部点
                pp = 0; pm = 0; mp = 0; mm = 0
                for dfvec in diff_intersections:
                    if   (dfvec[0] >= 0 and dfvec[1] >= 0):# or (dfvec[0] > 0 and dfvec[1] >= 0):
                        pp += 1
                    elif (dfvec[0] >= 0 and dfvec[1] <= 0):# or (dfvec[0] > 0 and dfvec[1] <= 0):
                        pm += 1
                    elif (dfvec[0] <= 0 and dfvec[1] >= 0):# or (dfvec[0] < 0 and dfvec[1] >= 0):
                        mp += 1
                    elif (dfvec[0] <= 0 and dfvec[1] <= 0):# or (dfvec[0] < 0 and dfvec[1] <= 0):
                        mm += 1
                if pp > 0 and pm > 0 and mp > 0 and mm > 0:
                    inner_intersections_index.append(ind)
            # 从 intersections 剔除 inner_intersections
            intersections = np.delete(intersections, inner_intersections_index, axis = 0)
            # 计算 intersections 中的质心
            intersections_mean = np.mean(intersections, axis = 0) if center_point is None else center_point
            # 计算 intersections 中每个坐标与质心的连线与 x 轴的夹角; 取值范围为 [0, 2pi]
            intersections_angle = np.arctan2(intersections[:,1] - intersections_mean[1], intersections[:,0] - intersections_mean[0])
            # 按照 intersections_angle 对 intersections 进行排序
            intersections = intersections[np.argsort(intersections_angle)]
            return intersections
        palist, peqdict = yaml_key2A(self.detail_mech)
        mkdirplus(dir0 + "/tmp_yamls"); save_path = dir0 + "/IDT_coutuor_plot.png" if save_path is None else save_path
        if not os.path.exists(dir0 + "/tmp_yamls_loss_average.npz"):
            all_A = []
            for maxkey_value in np.linspace(A1_range[0], A1_range[1], one_dim_sample_size):
                for secondmax_value in np.linspace(A2_range[0], A2_range[1], one_dim_sample_size):
                    all_A.append([maxkey_value, secondmax_value])
            # 使用 pool 进行并行计算, 使用 yaml2idt 进行计算
            pool = Pool(cpu_nums)
            for i, (maxkey_value, secondmax_value) in enumerate(all_A):
                pool.apply_async(ValidationContourDiagram_IDT_HRR_mps, 
                                kwds = dict(
                                    dir0 = dir0,
                                    peqdict = peqdict,
                                    max_key = max_key,
                                    second_max_key = second_max_key,
                                    maxkey_value = maxkey_value,
                                    secondmax_value = secondmax_value,
                                    index = i,
                                    chem_file = dir0 + f"/tmp_yamls/reduced_chem_{i}.yaml",
                                    setup_file = dir0 + "/settings/setup.yaml",
                                    mode = 0,
                                    cut_time = 10,
                                    save_path = dir0 + f"/tmp_yamls/reduced_chem_{i}.npy",
                                    original_chem_path = self.reduced_mech,
                                )
                                )
            pool.close(); pool.join()
            

            # 读取所有的 reduced_chem_{i}.npz 的 idt 数据
            idt_data = []; hrr_data = []
            for ii in range(len(all_A)):
                filename = dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz"
                if os.path.exists(filename):
                    data = np.load(dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz")
                    idt_data.append(data['IDT'].tolist())
                    hrr_data.append(data['HRR'].tolist())
                else:
                    idt_data.append(( 10 *np.ones_like(true_idt_data)).tolist())
                    hrr_data.append((np.zeros_like(true_hrr_data)).tolist())

            idt_data = np.log10(idt_data) - np.log10(true_idt_data)
            hrr_data = np.log10(hrr_data) - np.log10(true_hrr_data)
            loss_average = idt_data * weight_array[0] + hrr_data * weight_array[1]
            np.savez(dir0 + "/tmp_yamls_loss_average.npz", loss_average = loss_average, all_A = all_A)

        loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['loss_average']
        all_A = np.load(dir0 + "/tmp_yamls_loss_average.npz")['all_A']

        loss_average = np.linalg.norm(loss_average, axis = 1).reshape((one_dim_sample_size, one_dim_sample_size))
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        fig, ax = plt.subplots(1, 1, figsize = (6, 6), dpi = 400, sharey = False, sharex = False)
        ax.set_xlabel(max_key,)
        ax.set_ylabel(second_max_key,)
        ax.set_title("IDT")
        ax.set_xlim(left = A1_range[0], right = A1_range[1])
        ax.set_ylim(bottom = A2_range[0], top = A2_range[1]) 

        ax.contourf(all_A1, all_A2, loss_average, 100, cmap = cm01)
        cbar = fig.colorbar(ax.contourf(all_A1, all_A2, loss_average, 100, cmap = cm01), ax = ax)
        cbar.set_label("IDT")
        
        Rectangle_points = []
        for i in range(len(optim_path) - 1):
            rectangle_point = optim_path[i]
            Rectangle_points.append([
                [rectangle_point[0] - sample_range[i], rectangle_point[1] - sample_range[i]],
                [rectangle_point[0] - sample_range[i], rectangle_point[1] + sample_range[i]],
                [rectangle_point[0] + sample_range[i], rectangle_point[1] - sample_range[i]],
                [rectangle_point[0] + sample_range[i], rectangle_point[1] + sample_range[i]],
                ])
        hull_points = getRectangleAllVertices(Rectangle_points)
        palist, peqdict = yaml_key2A(self.reduced_mech)
        # 在等高线图上根据 optim_path 绘制优化路径，其中 optim_path 是一个二维数组，每一行是一个优化点
        from matplotlib.patches import Rectangle, Polygon
        for i in range(len(optim_path) - 1):
            x = [optim_path[i][0], optim_path[i + 1][0]]
            y = [optim_path[i][1], optim_path[i + 1][1]]
            ax.plot(x, y, color = 'white', linewidth = 2)
            ax.scatter(x, y, edgecolors = 'white', facecolors='none', s = 30)
            if block_trace:
                patch1 = Rectangle((x[0] - sample_range[i], y[0] - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'white', fill = False, linewidth = 1, linestyle='--')
                ax.add_patch(patch1)
            # ax.annotate(f"STEP{i + 1}", xy = (x[1], y[1]), xytext = (x[1] + 0.01, y[1] + 0.01), color = "black", fontsize = 10)
            # ax[1].plot(x, y, color = 'black', linewidth = 0.5)
            # ax[1].annotate("", xy = (x[1], y[1]), xytext = (x[0], y[0]), arrowprops = dict(arrowstyle = "->", linewidth = 0.5))
        if hull_points is not None and Hull_trace:
            ax.scatter(hull_points[:,0], hull_points[:,1], edgecolors = 'yellow', facecolors='none', s = 30)
            patch2 = Polygon(hull_points, color = 'white', fill = False, linewidth = 1, linestyle='--')
            ax.add_patch(patch2)
        fig.tight_layout()
        fig.savefig(save_path)


    def ValidationIDT_HRR_LossLandscape_Iteration(self, IDT_condition, apart_data_path_list, optim_path,  true_idt_data, max_key, second_max_key, dir0, 
                                                   true_hrr_data, weight_array, A1_range = (-1, 1), A2_range = (-1, 1), peqdict = None,
                                                    one_dim_sample_size = 200, cpu_nums = 140, save_dirpath = None, original_A0 = (0, 0), 
                                                    vmax = 0.3, **kwargs):
        """
        绘制 真实情况下IDT的 contour diagram，以及每次通过筛选的 samples 分布，窥视网络筛选的效果如何
        params:
            apart_data_path_list: 筛选出来的 samples 的路径
            true_idt_data: 真实的 IDT 数据
            max_key: 作为横坐标的 key
            second_max_key: 作为纵坐标的 key
            dir0: 保存轨迹的文件夹
            A1_range: 第一个坐标的取值范围
            A2_range: 第二个坐标的取值范围
            one_dim_sample_size: 一维采样点的数量
            cpu_nums: cpu 核心数量
            load_best_dnn_args: 加载模型时 load_best_dnn 的参数设置
        """
        if peqdict is None:
            _, peqdict = yaml_key2A(self.detail_mech)
        mkdirplus(dir0 + "/tmp_yamls"); mkdirplus(save_dirpath)
        logger = Log(dir0 + "/ContourDiagram_log.log")
        chem_args = get_yaml_data(self.setup_file)
        fuel, oxidizer = chem_args['fuel'], chem_args['oxidizer']
        if not os.path.exists(dir0 + "/tmp_yamls_loss_average.npz"):
            all_A = []
            for maxkey_value in np.linspace(A1_range[0], A1_range[1], one_dim_sample_size):
                for secondmax_value in np.linspace(A2_range[0], A2_range[1], one_dim_sample_size):
                    all_A.append([maxkey_value, secondmax_value])
            # 使用 pool 进行并行计算, 使用 yaml2idt 进行计算
            pool = Pool(cpu_nums)
            for i, (maxkey_value, secondmax_value) in enumerate(all_A):
                pool.apply_async(ValidationContourDiagram_IDT_HRR_mps, 
                                kwds = dict(
                                    dir0 = dir0,
                                    peqdict = peqdict,
                                    max_key = max_key,
                                    second_max_key = second_max_key,
                                    maxkey_value = maxkey_value,
                                    secondmax_value = secondmax_value,
                                    index = i,
                                    chem_file = dir0 + f"/tmp_yamls/reduced_chem_{i}.yaml",
                                    mode = 0,
                                    cut_time = 10,
                                    save_path = dir0 + f"/tmp_yamls/reduced_chem_{i}.npz",
                                    original_chem_path = self.reduced_mech,
                                    logger = logger,
                                    fuel = fuel,
                                    oxidizer = oxidizer,
                                    IDT_condition = IDT_condition,
                                )
                                )
            pool.close(); pool.join()
            

            # 读取所有的 reduced_chem_{i}.npz 的 idt 数据
            idt_data = []; hrr_data = []
            for ii in range(len(all_A)):
                filename = dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz"
                if os.path.exists(filename):
                    data = np.load(dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz")
                    idt_data.append(data['IDT'].tolist())
                    hrr_data.append(data['HRR'].tolist())
                else:
                    idt_data.append((10 *np.ones_like(true_idt_data)).tolist())
                    hrr_data.append((np.ones_like(true_hrr_data)).tolist())

            idt_data = np.log10(idt_data) - np.log10(true_idt_data)
            hrr_data = np.log10(hrr_data) - np.log10(true_hrr_data)
            # loss_average = idt_data * weight_array[0] + hrr_data * weight_array[1]
            idt_loss_average = np.linalg.norm(idt_data, axis = 1)
            hrr_loss_average = np.linalg.norm(hrr_data, axis = 1)
            loss_average = weight_array[0] * idt_loss_average + weight_array[1] * hrr_loss_average
            np.savez(dir0 + "/tmp_yamls_loss_average.npz", loss_average = loss_average, all_A = all_A, idt_loss_average = idt_loss_average, hrr_loss_average = hrr_loss_average)

        loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['loss_average']
        idt_loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['idt_loss_average']
        hrr_loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['hrr_loss_average']
        all_A = np.load(dir0 + "/tmp_yamls_loss_average.npz")['all_A']

        loss_average = loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        idt_loss_average = idt_loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        hrr_loss_average = hrr_loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 将 eq_dict 里面 max_key 和 second_max_key 的值改为 0 和 0.01
        eq_dict = copy.deepcopy(peqdict); eq_dict[max_key] = 0; eq_dict[second_max_key] = 0.01
        # eq_dict 转化为 Alist, 确定 Alist 中 max_key 和 second_max_key 的位置
        tmp_Alist = eq_dict2Alist(eq_dict)
        tmp_Alist = tmp_Alist.tolist()
        ## 找到 max_key 和 second_max_key 在 Alist 中的位置
        maxkey_index = tmp_Alist.index(0); secondmax_index = tmp_Alist.index(0.01)
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        fig, ax = plt.subplots(1, 3, figsize = (16, 6), dpi = 350, sharey = False, sharex = False)
        for i, apart_data_path in enumerate(apart_data_path_list):
            apart_data = np.load(apart_data_path)['Alist']
            apart_data_x = original_A0[0] - apart_data[:, maxkey_index]
            apart_data_y = original_A0[1] - apart_data[:, secondmax_index]
            # optim_point_x = optim_path[i + 1][0]
            # optim_point_y = optim_path[i + 1][1]
            save_path = save_dirpath + f"/IDT_ValidationContourDiagram_circ={i}.png"
        
            for i, ls in enumerate([idt_loss_average, hrr_loss_average, loss_average]):
                ax[i].set_xlabel(max_key,)
                ax[i].set_ylabel(second_max_key,)
                ax[i].set_title("IDT loss landscape")
                ax[i].set_xlim(left = A1_range[0], right = A1_range[1])
                ax[i].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
                cplt = ax[i].contourf(all_A1, all_A2, ls, 100, cmap='RdYlBu_r', 
                                    vmin = np.amin(ls), vmax = vmax)
                # fig.colorbar(cplt, ax = ax[i])
                sample_range_x_delta = np.amax(apart_data_x) - np.amin(apart_data_x)
                sample_range_y_delta = np.amax(apart_data_y) - np.amin(apart_data_y)
                patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1)
                # patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
                ax[i].add_patch(patch1)
                # ax[0].add_patch(patch2)
                # 在 ax[0] 上标注 optim_path[i]
                ax[i].scatter(optim_path[:, 0], optim_path[:, 1], edgecolors = 'white', facecolors = 'none', marker = 'o', zorder = 3, s = 200)
                ax[i].plot(optim_path[:, 0], optim_path[:, 1], color = 'white', linewidth = 2)
            # density, x_edge, y_edge = np.histogram2d(apart_data[:,0], apart_data[:,1], bins=one_dim_sample_size) 
            # # 绘制热力图
            # ax[1].hist2d(apart_data_x, apart_data_y, bins=one_dim_sample_size, cmap=cm01)
            # sns.kdeplot(
            #     data = loss_average,
            #     x=np.linspace(A1_range[0], A1_range[1], one_dim_sample_size), y=np.linspace(A2_range[0], A2_range[1], one_dim_sample_size),
            #     cmap='RdYlBu', fill=True,
            #     clip=(np.amin(loss_average), vmax),
            #     cut=10,
            #     thresh=0, levels=15,
            #     ax=ax[1],
            # )
            # # ax[1].imshow(loss_average.T, extent=[A1_range[0], A1_range[1], A2_range[0], A2_range[1]], cmap='hot', origin='lower')
            # ax[1].set_xlabel(max_key,)
            # ax[1].set_ylabel(second_max_key,)
            # ax[1].set_xlim(left = A1_range[0], right = A1_range[1])
            # ax[1].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
            # patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1, linestyle='--')
            # # patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
            # ax[1].add_patch(patch1)
            # # ax[1].add_patch(patch2)
            # # 在 ax[1] 上标注 optim_path[i]
            # ax[1].scatter(optim_path[i + 1][0], optim_path[i + 1][1], c = 'white', marker = '*', zorder = 3)
            # # ax[1].imshow(density.T, extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            # #             cmap='hot', origin='lower')
            fig.tight_layout()
            fig.savefig(save_path)
           
           
    def ValidationIDT_LFS_LossLandscape_Iteration(self, IDT_condition, LFS_condition, apart_data_path_list, optim_path,  true_idt_data, max_key, second_max_key, dir0, 
                                                   true_lfs_data, weight_array, A1_range = (-1, 1), A2_range = (-1, 1), peqdict = None,
                                                    one_dim_sample_size = 200, cpu_nums = 140, save_dirpath = None, original_A0 = (0, 0), use_MPI = False, 
                                                    vmax = 0.3, **kwargs):
        """
        绘制 真实情况下IDT的 contour diagram，以及每次通过筛选的 samples 分布，窥视网络筛选的效果如何
        params:
            apart_data_path_list: 筛选出来的 samples 的路径
            true_idt_data: 真实的 IDT 数据
            max_key: 作为横坐标的 key
            second_max_key: 作为纵坐标的 key
            dir0: 保存轨迹的文件夹
            A1_range: 第一个坐标的取值范围
            A2_range: 第二个坐标的取值范围
            one_dim_sample_size: 一维采样点的数量
            cpu_nums: cpu 核心数量
            load_best_dnn_args: 加载模型时 load_best_dnn 的参数设置
        """
        if peqdict is None:
            _, peqdict = yaml_key2A(self.detail_mech)
        mkdirplus(dir0 + "/tmp_yamls"); mkdirplus(save_dirpath)
        logger = Log(dir0 + "/ContourDiagram_log.log")
        chem_args = get_yaml_data(self.setup_file)
        fuel, oxidizer = chem_args['fuel'], chem_args['oxidizer']
        if not os.path.exists(dir0 + "/tmp_yamls_loss_average.npz"):
            all_A = []
            for maxkey_value in np.linspace(A1_range[0], A1_range[1], one_dim_sample_size):
                for secondmax_value in np.linspace(A2_range[0], A2_range[1], one_dim_sample_size):
                    all_A.append([maxkey_value, secondmax_value])
            pool = Pool(cpu_nums)
            for i, (maxkey_value, secondmax_value) in enumerate(all_A):
                pool.apply_async(ValidationContourDiagram_IDT_HRR_LFS_mps, 
                kwds = dict(
                        dir0 = dir0,
                        IDT_condition = IDT_condition,
                        LFS_condition = LFS_condition,
                        peqdict = peqdict,
                        max_key = max_key,
                        second_max_key = second_max_key,
                        maxkey_value = maxkey_value,
                        secondmax_value = secondmax_value,
                        setup_file = self.setup_file,
                        mode = 0,
                        cut_time = 10,
                        original_chem_path = self.reduced_mech,
                        logger = logger,
                        fuel = fuel,
                        oxidizer = oxidizer,
                        index = i, 
                        chem_file = dir0 + f'/tmp_yamls/reduced_chem_{i}.yaml',
                        save_path = dir0 + f"/tmp_yamls/reduced_chem_{i}.npz",  
                        ))
            pool.close(); pool.join()
            

            # 读取所有的 reduced_chem_{i}.npz 的 idt 数据
            idt_data = []; hrr_data = []
            for ii in range(len(all_A)):
                filename = dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz"
                if os.path.exists(filename):
                    data = np.load(dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz")
                    idt_data.append(data['IDT'].tolist())
                    hrr_data.append(data['HRR'].tolist())
                else:
                    idt_data.append((10 *np.ones_like(true_idt_data)).tolist())
                    hrr_data.append((100 * np.ones_like(true_lfs_data)).tolist())

            idt_data = np.log10(idt_data) - np.log10(true_idt_data)
            lfs_data = lfs_data - true_lfs_data
            idt_loss_average = np.linalg.norm(idt_data, axis = 1)
            lfs_loss_average = np.linalg.norm(lfs_data, axis = 1)
            loss_average = weight_array[0] * idt_loss_average + weight_array[1] * lfs_loss_average
            np.savez(dir0 + "/tmp_yamls_loss_average.npz", loss_average = loss_average, all_A = all_A, idt_loss_average = idt_loss_average, lfs_loss_average = lfs_loss_average)

        loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['loss_average']
        idt_loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['idt_loss_average']
        lfs_loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['lfs_loss_average']
        all_A = np.load(dir0 + "/tmp_yamls_loss_average.npz")['all_A']

        loss_average = loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        idt_loss_average = idt_loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        lfs_loss_average = lfs_loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 将 eq_dict 里面 max_key 和 second_max_key 的值改为 0 和 0.01
        eq_dict = copy.deepcopy(peqdict); eq_dict[max_key] = 0; eq_dict[second_max_key] = 0.01
        # eq_dict 转化为 Alist, 确定 Alist 中 max_key 和 second_max_key 的位置
        tmp_Alist = eq_dict2Alist(eq_dict)
        tmp_Alist = tmp_Alist.tolist()
        ## 找到 max_key 和 second_max_key 在 Alist 中的位置
        maxkey_index = tmp_Alist.index(0); secondmax_index = tmp_Alist.index(0.01)
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        fig, ax = plt.subplots(1, 3, figsize = (16, 6), dpi = 350, sharey = False, sharex = False)
        for i, apart_data_path in enumerate(apart_data_path_list):
            apart_data = np.load(apart_data_path)['Alist']
            apart_data_x = original_A0[0] - apart_data[:, maxkey_index]
            apart_data_y = original_A0[1] - apart_data[:, secondmax_index]
            # optim_point_x = optim_path[i + 1][0]
            # optim_point_y = optim_path[i + 1][1]
            save_path = save_dirpath + f"/IDT_ValidationContourDiagram_circ={i}.png"
        
            for i, ls in enumerate([idt_loss_average, lfs_loss_average, loss_average]):
                ax[i].set_xlabel(max_key,)
                ax[i].set_ylabel(second_max_key,)
                ax[i].set_title("IDT loss landscape")
                ax[i].set_xlim(left = A1_range[0], right = A1_range[1])
                ax[i].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
                cplt = ax[i].contourf(all_A1, all_A2, ls, 100, cmap='RdYlBu_r', 
                                    vmin = np.amin(ls), vmax = vmax)
                # fig.colorbar(cplt, ax = ax[i])
                sample_range_x_delta = np.amax(apart_data_x) - np.amin(apart_data_x)
                sample_range_y_delta = np.amax(apart_data_y) - np.amin(apart_data_y)
                patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1)
                # patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
                ax[i].add_patch(patch1)
                # ax[0].add_patch(patch2)
                # 在 ax[0] 上标注 optim_path[i]
                ax[i].scatter(optim_path[:, 0], optim_path[:, 1], edgecolors = 'white', facecolors = 'none', marker = 'o', zorder = 3, s = 200)
                ax[i].plot(optim_path[:, 0], optim_path[:, 1], color = 'white', linewidth = 2)
            # density, x_edge, y_edge = np.histogram2d(apart_data[:,0], apart_data[:,1], bins=one_dim_sample_size) 
            # # 绘制热力图
            # ax[1].hist2d(apart_data_x, apart_data_y, bins=one_dim_sample_size, cmap=cm01)
            # sns.kdeplot(
            #     data = loss_average,
            #     x=np.linspace(A1_range[0], A1_range[1], one_dim_sample_size), y=np.linspace(A2_range[0], A2_range[1], one_dim_sample_size),
            #     cmap='RdYlBu', fill=True,
            #     clip=(np.amin(loss_average), vmax),
            #     cut=10,
            #     thresh=0, levels=15,
            #     ax=ax[1],
            # )
            # # ax[1].imshow(loss_average.T, extent=[A1_range[0], A1_range[1], A2_range[0], A2_range[1]], cmap='hot', origin='lower')
            # ax[1].set_xlabel(max_key,)
            # ax[1].set_ylabel(second_max_key,)
            # ax[1].set_xlim(left = A1_range[0], right = A1_range[1])
            # ax[1].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
            # patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1, linestyle='--')
            # # patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
            # ax[1].add_patch(patch1)
            # # ax[1].add_patch(patch2)
            # # 在 ax[1] 上标注 optim_path[i]
            # ax[1].scatter(optim_path[i + 1][0], optim_path[i + 1][1], c = 'white', marker = '*', zorder = 3)
            # # ax[1].imshow(density.T, extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            # #             cmap='hot', origin='lower')
            fig.tight_layout()
            fig.savefig(save_path)
        
                
    def ValidationIDT_HRR_LFS_LossLandscape_Iteration(self, apart_data_path_list, optim_path, true_idt_data, max_key, second_max_key, dir0, 
                                                   true_hrr_data, true_lfs_data, weight_array, IDT_condition, LFS_condition, A1_range = (-1, 1), A2_range = (-1, 1), peqdict = None,
                                                    one_dim_sample_size = 200, cpu_nums = 140, save_dirpath = None, original_A0 = None, sample_range = None, 
                                                    use_MPI = False, vmax = 0.3, **kwargs):
        """
        绘制 真实情况下IDT的 contour diagram，以及每次通过筛选的 samples 分布，窥视网络筛选的效果如何
        params:
            apart_data_path_list: 筛选出来的 samples 的路径
            true_idt_data: 真实的 IDT 数据
            max_key: 作为横坐标的 key
            second_max_key: 作为纵坐标的 key
            dir0: 保存轨迹的文件夹
            A1_range: 第一个坐标的取值范围
            A2_range: 第二个坐标的取值范围
            one_dim_sample_size: 一维采样点的数量
            cpu_nums: cpu 核心数量
            load_best_dnn_args: 加载模型时 load_best_dnn 的参数设置
        """
        if peqdict is None:
            _, peqdict = yaml_key2A(self.detail_mech)
            write_json_data(dir0 + "/peqdict.json", peqdict)
        mkdirplus(dir0 + "/tmp_yamls"); mkdirplus(save_dirpath)
        logger = Log(dir0 + "/ContourDiagram_log.log")
        chem_args = get_yaml_data(self.setup_file)
        fuel, oxidizer = chem_args['fuel'], chem_args['oxidizer']
        if not os.path.exists(dir0 + "/tmp_yamls_loss_average.npz"):
            logger.info(f"len of conditions are {len(IDT_condition)} and {len(LFS_condition)}; cpu nums are {cpu_nums}")
            all_A = []
            for maxkey_value in np.linspace(A1_range[0], A1_range[1], one_dim_sample_size):
                for secondmax_value in np.linspace(A2_range[0], A2_range[1], one_dim_sample_size):
                    all_A.append([maxkey_value, secondmax_value])
            # 使用 pool 进行并行计算, 使用 yaml2idt 进行计算
            pool = Pool(cpu_nums)
            for i, (maxkey_value, secondmax_value) in enumerate(all_A):
                pool.apply_async(ValidationContourDiagram_IDT_HRR_LFS_mps, 
                kwds = dict(
                        dir0 = dir0,
                        IDT_condition = IDT_condition,
                        LFS_condition = LFS_condition,
                        peqdict = peqdict,
                        max_key = max_key,
                        second_max_key = second_max_key,
                        maxkey_value = maxkey_value,
                        secondmax_value = secondmax_value,
                        setup_file = self.setup_file,
                        mode = 0,
                        cut_time = 10,
                        original_chem_path = self.reduced_mech,
                        logger = logger,
                        fuel = fuel,
                        oxidizer = oxidizer,
                        index = i, 
                        chem_file = dir0 + f'/tmp_yamls/reduced_chem_{i}.yaml',
                        save_path = dir0 + f"/tmp_yamls/reduced_chem_{i}.npz",  
                        ))
            pool.close(); pool.join()
            

            # 读取所有的 reduced_chem_{i}.npz 的 idt 数据
            idt_data = []; hrr_data = []; lfs_data = []
            for ii in range(len(all_A)):
                filename = dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz"
                if os.path.exists(filename):
                    data = np.load(dir0 + f"/tmp_yamls/reduced_chem_{ii}.npz")
                    idt_data.append(data['IDT'].tolist())
                    hrr_data.append(data['HRR'].tolist())
                    lfs_data.append(data['LFS'].tolist())
                else:
                    idt_data.append((10 *np.ones_like(true_idt_data)).tolist())
                    hrr_data.append((np.ones_like(true_hrr_data)).tolist())
                    lfs_data.append((100 * np.ones_like(true_lfs_data)).tolist())
            idt_data = np.log10(idt_data) - np.log10(true_idt_data)
            hrr_data = np.log10(hrr_data) - np.log10(true_hrr_data)
            lfs_data = lfs_data - true_lfs_data
            idt_loss_average = np.linalg.norm(idt_data, axis = 1)
            hrr_loss_average = np.linalg.norm(hrr_data, axis = 1)
            lfs_loss_average = np.linalg.norm(lfs_data, axis = 1)
            loss_average = idt_loss_average * weight_array[0] + hrr_loss_average * weight_array[1] + lfs_loss_average * weight_array[2]
            np.savez(dir0 + "/tmp_yamls_loss_average.npz", loss_average = loss_average, all_A = all_A, idt_loss_average = idt_loss_average, 
                     hrr_loss_average = hrr_loss_average, lfs_loss_average = lfs_loss_average)

        loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['loss_average']
        idt_loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['idt_loss_average']
        hrr_loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['hrr_loss_average']
        lfs_loss_average = np.load(dir0 + "/tmp_yamls_loss_average.npz")['lfs_loss_average']
        all_A = np.load(dir0 + "/tmp_yamls_loss_average.npz")['all_A']
        
        loss_average = loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        idt_loss_average = idt_loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        hrr_loss_average = hrr_loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        lfs_loss_average = lfs_loss_average.reshape((one_dim_sample_size, one_dim_sample_size))
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 将 eq_dict 里面 max_key 和 second_max_key 的值改为 0 和 0.01
        eq_dict = copy.deepcopy(peqdict); eq_dict[max_key] = 0; eq_dict[second_max_key] = 0.01
        # eq_dict 转化为 Alist, 确定 Alist 中 max_key 和 second_max_key 的位置
        tmp_Alist = eq_dict2Alist(eq_dict)
        tmp_Alist = tmp_Alist.tolist()
        ## 找到 max_key 和 second_max_key 在 Alist 中的位置
        maxkey_index = tmp_Alist.index(0); secondmax_index = tmp_Alist.index(0.01)
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        all_A1 = all_A[:,0].reshape((one_dim_sample_size, one_dim_sample_size))
        all_A2 = all_A[:,1].reshape((one_dim_sample_size, one_dim_sample_size))
        # 将 eq_dict 里面 max_key 和 second_max_key 的值改为 0 和 0.01
        eq_dict = copy.deepcopy(peqdict); eq_dict[max_key] = 0; eq_dict[second_max_key] = 0.01
        # eq_dict 转化为 Alist, 确定 Alist 中 max_key 和 second_max_key 的位置
        tmp_Alist = eq_dict2Alist(eq_dict)
        tmp_Alist = tmp_Alist.tolist()
        ## 找到 max_key 和 second_max_key 在 Alist 中的位置
        maxkey_index = tmp_Alist.index(0); secondmax_index = tmp_Alist.index(0.01)
        # 以 all_A 的两个分量为横坐标和纵坐标，绘制 IDT 的等高线图
        fig, ax = plt.subplots(1, 4, figsize = (20, 6), dpi = 300, sharey = False, sharex = False)
        for i, apart_data_path in enumerate(apart_data_path_list):
            fig2, ax2 = plt.subplots(figsize=(6, 1))
            fig2.subplots_adjust(bottom=0.5)
            apart_data = np.load(apart_data_path)['Alist']
            apart_data_x = original_A0[0] - apart_data[:, maxkey_index]
            apart_data_y = original_A0[1] - apart_data[:, secondmax_index]
            # optim_point_x = optim_path[i + 1][0]
            # optim_point_y = optim_path[i + 1][1]
            save_path = save_dirpath + f"/IDT_ValidationContourDiagram_circ={i}.png"
            title_name = ['IDT', 'HRR', 'LFS', 'Total']
            for i, ls in enumerate([idt_loss_average, hrr_loss_average, lfs_loss_average, loss_average]):
                ax[i].set_xlabel(max_key,)
                ax[i].set_ylabel(second_max_key,)
                ax[i].set_title(f"{title_name[i]} loss landscape")
                ax[i].set_xlim(left = A1_range[0], right = A1_range[1])
                ax[i].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
                cplt = ax[i].contourf(all_A1, all_A2, ls, 100, cmap='RdYlBu_r', 
                                    vmin = np.amin(ls), vmax = vmax)
                # fig.colorbar(cplt, ax = ax[i])
                sample_range_x_delta = np.amax(apart_data_x) - np.amin(apart_data_x)
                sample_range_y_delta = np.amax(apart_data_y) - np.amin(apart_data_y)
                patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1)
                # patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
                ax[i].add_patch(patch1)
                # ax[0].add_patch(patch2)
                # 在 ax[0] 上标注 optim_path[i]
                ax[i].scatter(optim_path[:, 0], optim_path[:, 1], edgecolors = 'white', facecolors = 'none', marker = 'o', zorder = 3, s = 200)
                ax[i].plot(optim_path[:, 0], optim_path[:, 1], color = 'white', linewidth = 2)
                cbar = fig2.colorbar(cplt, cax=ax2, orientation='horizontal')
            # density, x_edge, y_edge    = np.histogram2d(apart_data[:,0], apart_data[:,1], bins=one_dim_sample_size) 
            # # 绘制热力图
            # ax[1].hist2d(apart_data_x, apart_data_y, bins=one_dim_sample_size, cmap=cm01)
            # ax[1].set_xlabel(max_key,)
            # ax[1].set_ylabel(second_max_key,)
            # ax[1].set_xlim(left = A1_range[0], right = A1_range[1])
            # ax[1].set_ylim(bottom = A2_range[0], top = A2_range[1]) 
            # patch1 = Rectangle((np.amin(apart_data_x), np.amin(apart_data_y)), sample_range_x_delta , sample_range_y_delta , color = 'white', fill = False, linewidth = 1, linestyle='--')
            # patch2 = Rectangle((optim_point_x - sample_range[i], optim_point_y - sample_range[i]), sample_range[i] * 2, sample_range[i] * 2, color = 'black', fill = False, linewidth = 1, linestyle='--')
            # ax[1].add_patch(patch1)
            # ax[1].add_patch(patch2)
            # 在 ax[1] 上标注 optim_path[i]
            # ax[1].scatter(optim_path[i + 1][0], optim_path[i + 1][1], c = 'white', marker = '*', zorder = 3)
            # ax[1].imshow(density.T, extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            #             cmap='hot', origin='lower')
            fig.tight_layout()
            fig.savefig(save_path)
            fig2.savefig(f'save_dirpath + f"/IDT_ValidationContourDiagram_colorbar.png')
            break
               
               
    """==============================================================================================================="""
    """                                  Validation FOR Experiment Data                                               """
    """==============================================================================================================="""
         
    
    def ExpValidationIDT_ON_TimeTemperatureCurve(self, IDT_condition, experiment_data:float, save_dirpath = ".", 
                                                 fuel = None, oxidizer = None, cut_time = 1, yaml2idt_kwargs = {}, right_x_lim = 0.01, 
                                                 logscale = False, **kwargs):
        """
        结合实验数据针对 IDT 进行验证，绘制 time - T 曲线后在其上标注 Experiment IDT 和 Optimized IDT
        同时绘制 OH 浓度分数变化曲线和压强变化曲线
        暂时只支持单点工况
        """
        phi, T, P = IDT_condition
        fuel = self.fuel if fuel is None else fuel
        oxidizer = self.oxidizer if oxidizer is None else oxidizer
        # 生成 Time-Temperature 曲线
        timelist, tlist, idt, T = yaml2idtcurve(self.optim_mech,
                                        IDT_condition = IDT_condition,
                                        fuel = fuel,
                                        oxidizer = oxidizer,
                                        cut_time = cut_time,
                                        yaml2idtcurve_needIDT = True,
                                        
                                        **yaml2idt_kwargs
                                        )
        # 生成 Time-Pressure 曲线
        # right_x_lim = timelist[-1] + 0.01
        gas = ct.Solution(self.optim_mech); gas.TP = T, P * ct.one_atm; gas.set_equivalence_ratio(phi, fuel, oxidizer)
        _,_, Ptimelist, Plist = idt_definition_pressure_slope_max(
            gas = gas,
            cut_time = cut_time * 10,
            need_P = True,
            idt_defined_T_diff = 1000,
        )
        # 生成 Time-OH 曲线
        OHtimelist, OHlist = yaml2mole_curve(
            chem_file = self.optim_mech,
            species = 'OH',
            mole_condition = IDT_condition,
            fuel = fuel,
            oxidizer = oxidizer,
            cut_time = cut_time,
        )
        # 获取 Plist 的量级
        Plist_order = int(np.log10(np.amax(Plist)))
        OHlist_order = int(np.log10(np.amax(OHlist)))
        Plist = Plist / 10 ** Plist_order
        OHlist = OHlist / 10 ** OHlist_order
        # 绘制三条曲线
        fig, ax = plt.subplots(1, 1, figsize = (8, 6), dpi = 400,)
        ax1 = ax.twinx(); ax2 = ax.twinx()
        ax2.spines['right'].set_position(('axes', 1.15))
        ax.plot(timelist, tlist, color = 'black', linewidth = 1, label = "Temperature")
        ax1.plot(Ptimelist, Plist, color = 'red', linewidth = 1, label = "Pressure")
        ax2.plot(OHtimelist, OHlist, color = 'blue', linewidth = 1, label = "OH")
        
        ax.set_xlim(left = 0, right = right_x_lim)
        ax1.set_xlim(left = 0, right = right_x_lim)
        ax2.set_xlim(left = 0, right = right_x_lim)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (K)")
        ax1.set_ylabel(fr"Pressure (atm) $\times 10^{Plist_order}$")
        ax2.set_ylabel(fr"OH Mole Concentration $\times 10^{Plist_order}$")
        if logscale:
            ax.set_xscale('log')
        # LEGEND 放在图片的顶端
        lines, labels = ax.get_legend_handles_labels()
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines1 + lines2, labels + labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, shadow=True, ncol=3)
        fig.tight_layout()
        fig.savefig(save_dirpath + "/Time_Temperature_Pressure_OH.png")
        plt.close(fig)
        
        
    def ValidationIDT_WithExperiment(self, IDT_condition:np.ndarray = None, experiment_IDT = None, simulation_IDTdata_index = None, mechs:dict = {}, fuel: list = None, 
                                     oxidizers:list = None, cut_time = 1, save_dirpath = None, LoadIDTExpData_kwargs = {},
                                     train_probepoint_index = None, detail_mechanism_index = None,
                                     **kwargs):
        """
        将得到的优化结果与真实的实验数据进行对比; 我们默认结果展示都是以 1000 / T 作为横坐标，以 IDT 作为纵坐标
        除了 reduced_mech 之外，会考虑所有的 mechs

        20230325: 删除了对于 detail_mech 的绘制

        params:
            IDT_condition: 输入的每条工况必须和 experiment_IDT 的每条数据对应
            experiment_IDT: 实验数据
            oxidizers: 注意到在很多实验中一些惰性气体的比例是不同的，因此在这种情况下请逐条输入当前 experiment_IDT 下的燃料情况
                一般的输入为 "O2: xxx, N2: xxx, AR: xxx"

            head: 最后图上显示 head 个元素。将根据真值和试验值的差距从大到小绘制

        """
        format_settings()
        np.set_printoptions(suppress = True)
        if save_dirpath is None: save_dirpath = self.vlidpath
        # IDT_condition, fuel, oxidizers, true_experiment_idt_data = LoadIDTExpDataWithoutMode(experiment_dataset, raw_data = False, **LoadIDTExpData_kwargs)
        mkdirplus(save_dirpath + "/ValidationIDT_WithExperiment_data")
        print(f'len of IDT, fuel and oxidizers are {len(IDT_condition)}, {len(fuel)} and {len(oxidizers)}')
        mechs = mechs | {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech
        }
        IDT = {}
        for mech in mechs.keys():
            tmp_IDT = []; 
            for condition, tmp_fuel, tmp_oxidizer in zip(IDT_condition, fuel, oxidizers):
                phi, T, P = condition
                tmp_save_path = save_dirpath + f"/ValidationIDT_WithExperiment_data/{mech}_IDT_T={T}_P={P}_phi={phi}.npz"
                t0 = time.time(); 
                tmp_idt, _ = yaml2idt(
                        mechs[mech],
                        mode = self.IDT_mode,
                        IDT_condition = np.array(condition),
                        fuel = tmp_fuel,
                        oxidizer = tmp_oxidizer,
                        save_path = tmp_save_path,
                        cut_time = cut_time,
                        )
                tmp_IDT.append(tmp_idt)
                print(f"Finished the {mech} IDT calculating! cost {time.time() - t0} s; idt is {tmp_idt}")     
            IDT.update({mech: tmp_IDT})
            print('tmp_idt:', len(tmp_IDT))
                     
        
        # 将所有其他的值减去真实值
        IDT = IDT | {"experiment": experiment_IDT}
        # [print(f'{key}:', len(IDT[key])) for key in IDT.keys()]

        df = pd.DataFrame(IDT, columns = list(IDT.keys()), dtype = np.float64)
        # df = df.reset_index().rename(columns={"index":"working_conditions"})
        df.to_csv(save_dirpath + "/ValidationIDT_WithExperiment_dataset.csv")

        num_datum = len(IDT_condition)
        fig, ax = plt.subplots(1,1, figsize=(7, 4), dpi = 300)
        fig.subplots_adjust(left=0.15, bottom=0.1, right=0.8, top=0.8, wspace=None, hspace=None)

        # 计算相对误差
        original_reduced_data = np.log10(IDT['reduced'])
        original_optimal_data = np.log10(IDT['optimal'])
        original_experiment_data = np.log10(IDT['experiment'])
        reduced_relative_error = np.mean(np.abs((original_reduced_data - original_experiment_data) / original_experiment_data)) * 100
        optimal_relative_error = np.mean(np.abs((original_optimal_data - original_experiment_data) / original_experiment_data)) * 100
        # 按照从大到小的顺序排序 experiment_data; 并且取出对应的 index 来排序 IDT['optimal']; IDT['reduced']
        sort_index = np.argsort(IDT['experiment'])[::-1]
        IDT['experiment'] = np.array(IDT['experiment'])[sort_index]
        IDT['optimal'] = np.array(IDT['optimal'])[sort_index]
        IDT['reduced'] = np.array(IDT['reduced'])[sort_index]
        IDT_condition = np.array(IDT_condition)[sort_index]
        
        # 在 sort_index 中提取出来 simulation_IDTdata_index
        if simulation_IDTdata_index is not None:
            print(f'simulation_IDTdata_index={simulation_IDTdata_index}')
            print(f'sort_index={sort_index}')
            # 按照 sort_index 的顺序重排 simulation_IDTdata_index
            simulation_IDTdata_index = np.array([np.where(sort_index == i)[0][0] for i in simulation_IDTdata_index])
            print(f'simulation_IDTdata_index={simulation_IDTdata_index}')
        
        print(f'The sixth element of IDT[\'experiment\'] is {np.log10(IDT["experiment"][6])}; the sixth element of IDT[\'optimal\'] is {np.log10(IDT["optimal"][6])}; the sixth element of IDT[\'reduced\'] is {np.log10(IDT["reduced"][6])}')
        print(f'the IDT condition is {IDT_condition[6]}')
        line0 = ax.plot(np.arange(num_datum),IDT['reduced'],  label = 'Original',  c = '#54cfc3',ls = '--', lw = 3, zorder = 2)
        
        # 在绘制 IDT['experiment'] 时，simulation_IDTdata_index 的位置 plot 的线颜色为 #FF9F1C 而非 simulation_IDTdata_index 的位置 plot 的线颜色为 #011627
        ## 对  IDT['experiment'] 插值，计算每个点的中点对应的值
        IDT_experiment = np.array(IDT['experiment'])
        IDT_experiment_xaxis = np.arange(num_datum, dtype=np.float32)
        # xaixs 中每两个点之间的中点增加
        IDT_experiment_xaxis_interpolate = (IDT_experiment_xaxis[0:-1] + IDT_experiment_xaxis[1:]) / 2
        IDT_experiment_xaxis = np.insert(IDT_experiment_xaxis, np.arange(1, len(IDT_experiment_xaxis)), IDT_experiment_xaxis_interpolate)
        ## IDT_experiment_xaxis 头尾增加一个复制第一个元素和最后一个元素
        IDT_experiment_xaxis = np.array([-0.5] + IDT_experiment_xaxis.tolist() + [IDT_experiment_xaxis[-1]+0.5])
        
        IDT_experiment_interpolate = (IDT_experiment[0:-1] + IDT_experiment[1:]) / 2
        IDT_experiment_interpolate = np.insert(IDT_experiment, np.arange(1, len(IDT_experiment)), IDT_experiment_interpolate)
        IDT_experiment_interpolate = np.array([IDT_experiment_interpolate[0]] + IDT_experiment_interpolate.tolist() + [IDT_experiment_interpolate[-1]])
        ## 绘制 simulation_IDTdata_index 的位置 plot 的线
        
        print(f'IDT_experiment_xaxis={IDT_experiment_xaxis}, IDT_experiment_interpolate={IDT_experiment_interpolate}')
        for i in np.arange(1, num_datum):
            if i in simulation_IDTdata_index:
                line1 = ax.plot([IDT_experiment_xaxis[2*i-1], IDT_experiment_xaxis[2*i+1]], [IDT_experiment_interpolate[2*i-1], IDT_experiment_interpolate[2*i+1]], c = '#AAB083', lw = 4, zorder = 2, label = 'Experiment Benchmark')
            else:
                line2 = ax.plot([IDT_experiment_xaxis[2*i-1], IDT_experiment_xaxis[2*i+1]], [IDT_experiment_interpolate[2*i-1], IDT_experiment_interpolate[2*i+1]], c = 'black', lw = 4, zorder = 2, label = 'Simulation Benchmark')
        
        # ax.plot(np.arange(num_datum), IDT['experiment'],  c = '#011627', ls = '--', lw = 2.5,  label = 'Benchmark', zorder = 1)
        # 提取 train_probepoint_index 对应的数据: train_probepoint_index 是一个列表
        train_probepoint_index = np.where(np.in1d(sort_index, train_probepoint_index))[0]
        not_train_probepoint_index = np.setdiff1d(sort_index, train_probepoint_index)
        detail_mechanism_index = np.where(np.in1d(sort_index, detail_mechanism_index))[0]
        experiment_mechanism_index = np.setdiff1d(sort_index, detail_mechanism_index)
        
        train_probepoint_detail_mechanism_index = np.setdiff1d(train_probepoint_index, experiment_mechanism_index)
        train_probepoint_experiment_index = np.setdiff1d(train_probepoint_index, detail_mechanism_index)
        
        not_train_probepoint_detail_mechanism_index = np.setdiff1d(not_train_probepoint_index, experiment_mechanism_index)
        not_train_probepoint_experiment_index =  np.setdiff1d(not_train_probepoint_index, detail_mechanism_index)
        
        sca1 = ax.scatter(train_probepoint_detail_mechanism_index[::2], np.array(IDT['optimal'])[train_probepoint_detail_mechanism_index][::2], 
                   marker = '^', edgecolors = '#0700e6', facecolors = 'none', s = 80,  label = 'Detailed Alignment point', zorder = 3, linewidths=2.5)
        # sca2 = ax.scatter(train_probepoint_experiment_index, np.array(IDT['optimal'])[train_probepoint_experiment_index],
        #              marker = '^', facecolors = 'none', facecolors = '#E71D36', s = 60,  label = 'Experiment Alignment point', zorder = 3)
        sca2 = ax.scatter(train_probepoint_experiment_index[::2], np.array(IDT['optimal'])[train_probepoint_experiment_index][::2],
                     marker = '^', edgecolors = '#0700e6', facecolors = 'none', s = 80,  linewidth = 2.5, label = 'Alignment point', zorder = 3,)
        sca3 = ax.scatter(not_train_probepoint_detail_mechanism_index, np.array(IDT['optimal'])[not_train_probepoint_detail_mechanism_index],
                        marker = 'o', facecolors =  '#E71D36', edgecolors = 'none', s = 80,  label = 'Detailed Optimized', zorder = 3, linewidths=3)
        # ax.scatter(not_train_probepoint_experiment_index, np.array(IDT['optimal'])[not_train_probepoint_experiment_index],
        #                 marker = 'o', edgecolors = 'none', facecolors = '#E71D36', s = 60,  label = 'Experiment Optimized', zorder = 3)
        
        # ax.scatter(np.arange(num_datum), IDT['optimal'][0:num_datum], marker = 'o', edgecolors = 'w', facecolors = '#E71D36', s = 110,  label = 'Optimized', zorder = 1)
        ylim1 = [0.5*np.min(IDT['optimal']), 0.5*np.min(IDT['reduced']), 0.5 * np.min(IDT['experiment'])]
        ylim2 = [2*np.max(IDT['optimal']), 2*np.max(IDT['reduced']), 2 * np.max(IDT['experiment'])]
        ylim = [np.min(ylim1), np.max(ylim2)] 
        ax.set(ylim = ylim, yscale = 'log')  # 确定子图的范围
        print(f'max of np.max(IDT[experiment]), np.max(IDT[optimal]), np.max(IDT[reduced]) is {np.max(IDT["experiment"])}, {np.max(IDT["optimal"])}, {np.max(IDT["reduced"])}')
        # 设置 xtick 倾斜 45 度
        ax.set_xticks(np.arange(num_datum))
        ax.set_xticklabels([], rotation = 35, fontsize = 16, ha='right')
        ax.tick_params(axis="x", width=0.5, length=0.5)
        ax.tick_params(axis="y", width=0.5, length=0.5)
        # yticklabel 设置为 $10^k$ 表示
        ax.set_yticklabels([f'$10^{{{int(np.log10(i))}}}$' for i in ax.get_yticks()], fontsize = 14)
        # 右上角表明 phi T, P 的范围
        ax.text(0.98, 0.98, rf"$\phi \in [{np.min(IDT_condition[:,0]):.1f}, {np.max(IDT_condition[:,0]):.1f}]$" + "\n" + 
                rf"$T \in [{np.floor(np.min(IDT_condition[:,1]))}, {np.ceil(np.max(IDT_condition[:,1]))}]$ " + r'$\bf{K}$' 
                +'\n'+ rf"$P \in [{np.floor(np.min(IDT_condition[:,2]))}, {np.ceil(np.max(IDT_condition[:,2]))}]$ " + r'$\bf{atm}$',
               fontsize = 14, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')

        ax.set_ylabel(f'Ignition Delay Time (s)'  
                    #   \nRelative Error {reduced_relative_error:.1f}% → {optimal_relative_error:.1f}%
                      , fontsize = 16, labelpad = 2)
        # if train_probepoint_index is not None:
        #     print(f'train_probepoint_index: {train_probepoint_index}')
        #     [ax.xaxis.get_ticklabels()[ind].set_bbox(dict(alpha=0.5, edgecolor='blue')) for ind in train_probepoint_index]
        # 关闭 grid
        ax.grid(False)
        # 在t图像边框外面的右侧定义竖排的 legend; 单列
        ax.legend([line0[0], line1[0], line2[0], sca1, sca3], ['Original', 'Experiment Benchmark', 'Simulation Benchmark', 'Alignment point', 'Optimized'], loc = 'lower left', bbox_to_anchor = (0, 0), 
                  fontsize = 16, frameon = False, ncol = 1, handletextpad = 0.5,
                  borderaxespad=0., labelspacing=0.1, columnspacing=0.2)
        fig.tight_layout()
        ax.set_xlabel('Experiment & Simulation Benchmarks', fontsize = 16, labelpad = 15)
        fig.savefig(save_dirpath + f"/ValidationIDT_WithExperiment_reduced_error={reduced_relative_error:.1f}_optim_error={optimal_relative_error:.1f}.png", dpi = 300,
                    bbox_inches='tight', pad_inches=0.15)


    def ValidationIDT_WithExperiment2(self, experiment_datasets:dict,  save_path = ".", cut_time = 1, probe_point = None, yaml2idt_kwargs = {}, LoadIDTExpData_kwargs = {}, probe_in_dataset = False, 
                                      sources_dict = None,xH2 = None, base_mech = None, smooth_window= None, pressure_group_threshold = None, **kwargs):
        """
        20241124 增加
        绘制 IDT 关于初值温度倒数的图像; 并将不同组的 P 画在同一张图上
        experiment_datasets: 实验数据集，应该以 dict 的形式输入
        """
        for key, experiment_dataset in experiment_datasets.items():
            experiment_IDT_condition, IDT_fuel, IDT_oxidizer, IDT_mode, true_experiment_idt_data, uncertainty_values = LoadIDTExpData(experiment_dataset, raw_data = True, uncertainty=True, **LoadIDTExpData_kwargs)
            
            if probe_in_dataset is True:
                probe_point = return_probe_point_index(experiment_dataset)
            elif probe_point is not None or isinstance(probe_in_dataset, Iterable):
                if probe_point is None:
                    probe_point = probe_in_dataset
                probe_point_index = []
                for tmp_condition in probe_point:
                    tmp_condition = np.array(tmp_condition)
                    for ind, cond in enumerate(experiment_IDT_condition):
                        if np.equal(tmp_condition, cond).all():
                            probe_point_index.append(ind)
                            print(f"Found probe point {tmp_condition} at index {ind} in experiment_IDT_condition")
                probe_point_index = np.array(probe_point_index)
                probe_point = probe_point_index
            else:
                probe_point = None
            
            if pressure_group_threshold is not None:
                # 按照 experiment_IDT_condition 的 P 列分成大于等于和小于 pressure_group_threshold 的两组
                tmp_pressure_group = [
                    np.where(experiment_IDT_condition[:, 2] < pressure_group_threshold)[0],
                    np.where(experiment_IDT_condition[:, 2] >= pressure_group_threshold)[0]
                ]
            else:
                tmp_pressure_group = None
            
            if base_mech == None:
                mechs = {
                "optimal": self.optim_mech,
                "reduced": self.reduced_mech,
            }
            else:
                mechs = {
                    "optimal": self.optim_mech,
                    "reduced": self.reduced_mech,
                    "base": base_mech,
                }
            IDT = {}
            multiprocess = os.cpu_count() - 1
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_lineplot_IDT_{key}.npz"
                if os.path.exists(tmp_save_path):
                    idt = np.load(tmp_save_path, allow_pickle=True)['IDT']
                else:
                    idt = yaml2idt_Mcondition(
                        chem_file = mechs[mech],
                        mode = IDT_mode,
                        IDT_condition= experiment_IDT_condition,
                        fuel = IDT_fuel,
                        oxidizer = IDT_oxidizer,
                        cut_time = cut_time,
                        cpu_process = multiprocess,
                        # save_dirpath = save_path,
                        **yaml2idt_kwargs
                    )
                    np.savez(tmp_save_path, IDT = idt)
                IDT.update({mech: idt})
              
            # 根据 T 重新排序所有数据
            
            sort_index = np.argsort(experiment_IDT_condition[:,1])
            experiment_IDT_condition = experiment_IDT_condition[sort_index]
            true_experiment_idt_data = true_experiment_idt_data[sort_index]
            if np.all(uncertainty_values == 1):
                uncertainty_values = None
            else:
                uncertainty_values = uncertainty_values[sort_index] * true_experiment_idt_data
            IDT = {key: value[sort_index] for key, value in IDT.items()}
            # 计算 IDT 的相对误差
            IDT['detail'] = np.array(true_experiment_idt_data)
            optim_relerror = np.mean(np.abs(IDT["detail"] - IDT["optimal"]) / IDT["detail"]) * 100
            reduced_relerror = np.mean(np.abs(IDT["detail"] - IDT["reduced"]) / IDT["detail"]) * 100
            if 'base' in IDT.keys():
                base_relerror = np.mean(np.abs(IDT["detail"] - IDT.get("base", None)) / IDT["detail"]) * 100
            else:
                base_relerror = None
            plt.rcParams['axes.spines.top'] = True
            plt.rcParams['axes.spines.right'] = True
            plt.rcParams['axes.spines.left'] = True
            plt.rcParams['axes.spines.bottom'] = True
            CompareDRO_IDT_lineplot2(
                detail_data =  true_experiment_idt_data,
                reduced_data = IDT["reduced"],
                optimal_data = IDT["optimal"],
                base_data = IDT.get("base", None),
                xH2 = xH2,
                smooth_window=smooth_window,
                IDT_condition=experiment_IDT_condition,
                probe_point=probe_point,
                save_path = save_path + f"/ValidationIDT_{key}_optimrelerror={optim_relerror:.1e}_reducedrelerror={reduced_relerror:.1e}_baserelerror={base_relerror}.png",
                sources_dict=sources_dict,
                benchmark_color="#ff474a",
                # benchmark_color="#000000",
                uncertainty= uncertainty_values,
                pressure_group = tmp_pressure_group,
                **kwargs
            )


    def ValidationHRR_WithExperiment2(self, experiment_datasets:dict, save_path = ".", cut_time = 1, probe_point = None, yaml2idt_kwargs = {}, LoadIDTExpData_kwargs = {}, probe_in_dataset = False, 
                                      sources_dict = None, **kwargs):
        """
        """
        for key, experiment_dataset in experiment_datasets.items():
            if probe_in_dataset:
                probe_point = return_probe_point_index(experiment_dataset)
            experiment_IDT_condition, IDT_fuel, IDT_oxidizer, IDT_mode, true_experiment_idt_data = LoadIDTExpData(experiment_dataset, raw_data = True, **LoadIDTExpData_kwargs)
            
            mechs = {
                "optimal": self.optim_mech,
                "reduced": self.reduced_mech
            }
            IDT = {}
            multiprocess = os.cpu_count() - 1
            for mech in mechs.keys():
                tmp_save_path = save_path + f"/{mech}_ValidationIDT_lineplot_IDT_{key}.npz"
                if os.path.exists(tmp_save_path):
                    idt = np.load(tmp_save_path, allow_pickle=True)['IDT']
                else:
                    idt = yaml2idt_Mcondition(
                        chem_file = mechs[mech],
                        mode = IDT_mode,
                        IDT_condition= experiment_IDT_condition,
                        fuel = IDT_fuel,
                        oxidizer = IDT_oxidizer,
                        cut_time = cut_time,
                        cpu_process = multiprocess,
                        # save_dirpath = save_path,
                        **yaml2idt_kwargs
                    )
                    np.savez(tmp_save_path, IDT = idt)
                IDT.update({mech: idt})
              
            # 根据 T 重新排序所有数据
            
            sort_index = np.argsort(experiment_IDT_condition[:,1])
            experiment_IDT_condition = experiment_IDT_condition[sort_index]
            true_experiment_idt_data = true_experiment_idt_data[sort_index]
            IDT = {key: value[sort_index] for key, value in IDT.items()}
            # 计算 IDT 的相对误差
            IDT['detail'] = np.array(true_experiment_idt_data)
            optim_relerror = np.mean(np.abs(IDT["detail"] - IDT["optimal"]) / IDT["detail"]) * 100
            reduced_relerror = np.mean(np.abs(IDT["detail"] - IDT["reduced"]) / IDT["detail"]) * 100
            plt.rcParams['axes.spines.top'] = True
            plt.rcParams['axes.spines.right'] = True
            plt.rcParams['axes.spines.left'] = True
            plt.rcParams['axes.spines.bottom'] = True
            CompareDRO_IDT_lineplot2(
                detail_data =  true_experiment_idt_data,
                reduced_data = IDT["reduced"],
                optimal_data = IDT["optimal"],
                IDT_condition= experiment_IDT_condition,
                probe_point=probe_point,
                save_path = save_path + f"/ValidationIDT_{key}_optimrelerror={optim_relerror}_reducedrelerror={reduced_relerror}.png",
                sources_dict=sources_dict,
                benchmark_color='#AAB083',
                **kwargs
            )

    
    def ValidationFS_WithExperiment(self, experiment_dataset, LoadIDTExpData_kwargs = {}, save_path = ".", probe_point = None,xH2 = None, xN2 = None, base_mech = None,smooth_window = None, selected_T =None, pressures_to_plot = None, **kwargs):
        """
        20241125 增加
        """
        mkdirplus(save_path)
        FS_condition, fuel, oxidizers, experiment_data, uncertainty_values = LoadIDTExpDataWithoutMode(
            experiment_dataset,
            condition_prefix = 'LFS', uncertainty=True,
            **LoadIDTExpData_kwargs
        )
        mechs = {
            "optimal": self.optim_mech,
            "reduced": self.reduced_mech,
        }
        if base_mech is not None: mechs.update({"base": base_mech})
        FS = {}
        for mech in mechs.keys():
            tmp_save_path = save_path + f"/{mech}_ValidationFS_lineplot_FS.npz"
            if os.path.exists(tmp_save_path):
                tmp_fs = np.load(tmp_save_path, allow_pickle = True)['FS']
                FS.update({mech: tmp_fs})
            else:
                t0 = time.time()
                print(f"Start the {mech} yaml2FS_Mcondition calculating! cpu process: {os.cpu_count() - 1}")
                tmp_fs = yaml2FS_Mcondition(
                                mechs[mech],
                                FS_condition= FS_condition,
                                fuel = fuel,
                                oxidizer = oxidizers,
                                cpu_process = os.cpu_count() - 1,
                                )
                np.savez(tmp_save_path, FS = tmp_fs)
                FS.update({mech: tmp_fs})
                print(f"Finished the {mech} FS calculating! cost {time.time() - t0} s")

        if np.all(uncertainty_values == 1):
            uncertainty_values = None
        else:
            uncertainty_values = uncertainty_values * experiment_data * 100
        print(f'uncertainty_values={uncertainty_values}')
        # 计算 FS 的相对误差
        # print(f'shape of FS["optimal"]={FS["optimal"].shape}, shape of FS["reduced"]={FS["reduced"].shape}, shape of experiment_data={experiment_data.shape}')
        FS['detail'] = np.array(experiment_data)
        optim_relerror = np.mean(np.abs(FS["detail"] - FS["optimal"]) / FS["detail"]) * 100
        reduced_relerror = np.mean(np.abs(FS["detail"] - FS["reduced"]) / FS["detail"]) * 100
        if 'base' in FS.keys():
            base_relerror = np.mean(np.abs(FS["detail"] - FS.get("base", None)) / FS["detail"]) * 100

        classified_index = classify_array_T_P(FS_condition)
        print(f'optimal FS: {FS["optimal"]}; reduced FS: {FS["reduced"]}; experiment FS: {FS["detail"]}')
        CompareDRO_LFS_lineplot2(
            detail_lfs =  FS["detail"],
            reduced_lfs = FS["reduced"],
            optimal_lfs = FS["optimal"],
            base_lfs = FS.get("base", None),
            index_group= classified_index,
            smooth_window= smooth_window,
            save_path = save_path + f"/Validation_FS_lineplot_optimrelerror={optim_relerror:.1e}_reducedrelerror={reduced_relerror:.1e}.png",
            FS_condition = FS_condition,
            probe_point = probe_point,
            concat_phi=True,
            benchmark_color='red',
            xH2=xH2,
            xN2=xN2,
            selected_T=selected_T,
            pressures_to_plot=pressures_to_plot,
            uncertainty=uncertainty_values,
            **kwargs
        )

    
    """==============================================================================================================="""
    """                                  Validation FOR Neural Network                                                """
    """==============================================================================================================="""
    
    @staticmethod
    def NeuralNetwork_SampleFilter_Acc(neuralnetwork_list, nn_json_list, load_best_dnn_args, apart_data_list, 
                                       true_idt_data, save_dirpath = None, **kwargs):
        """
        验证我们训练的神经网络在筛选样本方面的预测值与 Cantera 计算值的差距，以证明 DNN 可以纯化样本。
        params:
            neuralnetwork_list: 神经网络的列表
            nn_json_list: 神经网络的 json 文件的列表
            load_best_dnn_args: 加载模型时 load_best_dnn 的参数设置
            save_dirpath: 保存路径
        *注意 apart_data_list 是从 circ=1 开始的而 neuralnetwork_list 是从 circ=0 开始的
        """
        Average_Acc = []; Network_Acc, Cantera_Acc = [], []
        true_idt_data = np.log10(true_idt_data)
        for i, (apart_data_path, neuralnetwork_path, nn_json_path) in enumerate(zip(apart_data_list, neuralnetwork_list, nn_json_list)):
            print(f"Start to calculate the {i}th neural network's accuracy")
            save_path = save_dirpath + f"/NeuralNetwork_SampleFilter_Acc_{i}.npz"
            load_best_dnn_args['model_pth_path'] = neuralnetwork_path
            load_best_dnn_args['model_json_path'] = nn_json_path
            network = load_best_dnn(**load_best_dnn_args)
            apart_data = np.load(apart_data_path)['Alist']; apart_data_idt = np.log10(np.load(apart_data_path)['all_idt_data'])
            apart_data = torch.tensor(apart_data, dtype = torch.float32)
            pred_idt = network.forward_Net1(apart_data).detach().numpy()
            network_acc = np.linalg.norm(pred_idt - true_idt_data, axis = 1)
            cantera_acc = np.linalg.norm(apart_data_idt - true_idt_data, axis = 1)
            Average_Acc.append(np.mean(network_acc))
            Network_Acc.append(network_acc); Cantera_Acc.append(cantera_acc)
            print(f"Finished the {i}th neural network's accuracy; Previous Network averge accuracy is {Average_Acc[-1]}; Cantera accuracy is {np.mean(cantera_acc)}")
            print(f'shape of pred_idt: {network_acc.shape}, {cantera_acc.shape}')
        Circs = np.arange(len(Average_Acc))
        min_samplesize = min([
            len(Network_Acc[i]) for i in range(len(Network_Acc))
        ])
        Network_Acc = np.array(
            [Network_Acc[i][0:min_samplesize] for i in range(len(Network_Acc))]
        )
        Cantera_Acc = np.array(
            [Cantera_Acc[i][0:min_samplesize] for i in range(len(Cantera_Acc))]
        )
        print(f'shape of Network_Acc: {Network_Acc.shape}')
        pd_network_pred_loss = pd.DataFrame(Network_Acc.T, columns = Circs)
        pd_cantera_loss = pd.DataFrame(Cantera_Acc.T, columns = Circs)
        pd_network_pred_loss['resource'] = 'pd_network_pred_loss'
        pd_cantera_loss['resource'] = 'pd_cantera_loss'
        data = pd.concat([pd_network_pred_loss, pd_cantera_loss], axis = 0)
        data = pd.melt(data, id_vars = ['resource'], var_name = 'Circ', value_name = 'IDTloss')
        fig, ax = plt.subplots(1, 1, figsize = (10, 10 * len(Circs) / 27) ,dpi = 400)
        sns.violinplot(data = data, ax = ax, y = 'Circ', x = 'IDTloss', hue = 'resource',
                        split = True , bw = 0.1, orient = 'h')
        fig.legend(loc = 'upper center', ncol = 2, bbox_to_anchor = (0.5, 0.95), fontsize = 12, frameon = False)
        fig.savefig(save_dirpath + "/NeuralNetwork_SampleFilter_Acc.png", bbox_inches='tight', pad_inches=0.05)
        return Average_Acc, Network_Acc, Cantera_Acc


def ValidationContourDiagram_idt_mps(dir0, peqdict, max_key, second_max_key, maxkey_value, secondmax_value, index, chem_file:str, mode:int = 0, cut_time:float | list = 1, setup_file:str = None, 
             save_path:str = None, original_chem_path = "./settings/reduced_chem.yaml", **kwargs):
    t0 = time.time()
    copy_peqdict = copy.deepcopy(peqdict)
    copy_peqdict[max_key] = copy_peqdict[max_key] + maxkey_value; copy_peqdict[second_max_key] = copy_peqdict[second_max_key] + secondmax_value
    Adict2yaml(original_chem_path = original_chem_path,
               chem_path = chem_file,
               eq_dict = copy_peqdict)
    idt, _ = yaml2idt(
           chem_file = chem_file, mode = mode, cut_time = cut_time, setup_file = setup_file, **kwargs
       )
    np.save(save_path, idt)
    print(f"mech {index} generate cost IDT: {time.time() - t0:.2f} s; IDT:{idt[0]:.2f}")
    os.remove(chem_file)


def ValidationContourDiagram_IDT_HRR_mps(IDT_condition, peqdict, max_key, second_max_key, maxkey_value, secondmax_value, index, chem_file:str, logger,
                                        fuel, oxidizer,  mode:int = 0, cut_time:float | list = 1, save_path:str = None, original_chem_path = "./settings/reduced_chem.yaml", **kwargs):
    
    try:
        t0 = time.time()
        copy_peqdict = copy.deepcopy(peqdict)
        copy_peqdict[max_key] = copy_peqdict[max_key] + maxkey_value; copy_peqdict[second_max_key] = copy_peqdict[second_max_key] + secondmax_value
        Adict2yaml(original_chem_path = original_chem_path,
                chem_path = chem_file,
                eq_dict = copy_peqdict)
        try:
            idt, hrr, _ = _GenOneIDT_HRR(IDT_condition, fuel, oxidizer, chem_file, 0, logger, mode, cut_time = cut_time, **kwargs)
        except:
            idt, hrr = 1000, 1
            logger.info(f'The mechanism {index} is error at IDT generation')
        np.savez(save_path, IDT = idt, HRR = hrr)
        logger.info(f"mech {index} generate cost IDT: {time.time() - t0:.2f} s; IDT:{idt[0]:.2f}, HRR:{hrr[0]:.2f}")
        os.remove(chem_file)
    except:
        logger.info(traceback.format_exc())


def ValidationContourDiagram_IDT_HRR_LFS_mps(IDT_condition, LFS_condition, peqdict, max_key, second_max_key, maxkey_value, secondmax_value, index, 
                                             chem_file:str, fuel, oxidizer, mode:int = 0, cut_time:float | list = 1, 
                                            save_path:str = None, original_chem_path = "./settings/reduced_chem.yaml", logger = None, **kwargs):
    t0 = time.time()
    copy_peqdict = copy.deepcopy(peqdict)
    copy_peqdict[max_key] = copy_peqdict[max_key] + maxkey_value; copy_peqdict[second_max_key] = copy_peqdict[second_max_key] + secondmax_value
    Adict2yaml(original_chem_path = original_chem_path, chem_path = chem_file, eq_dict = copy_peqdict)
    try:
        idt, hrr, _ = _GenOneIDT_HRR(IDT_condition, fuel, oxidizer, chem_file, 0, logger, mode, cut_time = cut_time, **kwargs)
    except:
        idt, hrr = 1000, 1
        logger.info(f'The mechanism {index} is error at IDT generation')
    t1 = time.time()
    try:
        lfs = _GenOneLFS(LFS_condition, fuel, oxidizer, chem_file, 0, logger, **kwargs)
    except:
        lfs = 1000
        logger.info(f'The mechanism {index} is error at LFS generation')
    # lfs = np.zeros(LFS_condition.shape[0])
    np.savez(save_path, IDT = idt, HRR = hrr, LFS = lfs)
    if logger is not None:
        logger.info(f"mech {index} generate cost LFS: {time.time() - t1:.2f} + IDT: {t1 - t0:.2f} s; IDT:{idt[0]:.2f}, HRR:{hrr[0]:.2f}, LFS:{lfs[0]:.2f}")
    os.remove(chem_file)