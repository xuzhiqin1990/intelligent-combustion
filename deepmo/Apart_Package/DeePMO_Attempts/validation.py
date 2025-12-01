from ..APART_plot.APART_plot import Ahis_plot, compare_mech, compare_species, SA, sns_data_prepare
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from seaborn.objects import Area
import pandas as pd
import sys
from APART_base import APART_base
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.cantera_utils import *
from utils.setting_utils import *
from utils.yamlfiles_utils import *
from utils.DeePMR_base_network import *

"""下面两个函数是计算father sample离散程度的函数"""

def father_sample_discrete_degree(apart_data_A, apart_data_IDT, true_idt_data, core_size:int):
    """
    计算每个循环中 father sample 的离散程度
    离散程度的定义是来自于 PCA，对协方差矩阵 X^T @ X 计算特征值，考查特征值的分布。

    如果多个特征值的大小差不多，则可以说明在线性上这些数据的分布比较离散；如果大部分特征值都很低，说明在
    特征方向上的分布是最离散的，其他方向都比较聚集

    params:
        apart_data_A: 输入 apart_data_circ=x.npz 中的 Alist
        apart_data_IDT: 输入 apart_data_circ=x.npz 中的 IDT 用于判断哪些是 father sample
        true_idt_data: 真实 IDT data
        core_size: 取前多少位为 father sample
    return:
        lambdas: 协方差矩阵的特征值
    """
    diff_idt = np.amax(np.abs(np.log10(apart_data_IDT) - np.log10(true_idt_data)), axis = 1)
    index = np.argsort(diff_idt); apart_data_A = apart_data_A[index,:]
    coreAlist = apart_data_A[0:core_size, :]

    covariance = coreAlist.T @ coreAlist
    eigenvalue, _ = np.linalg.eig(covariance)

    return -np.sort(-eigenvalue)

def FSDD_plot(*father_sample:np.ndarray, **kwargs):
    """
    根据已经给定的fathersample 绘制叠嶂图
    """
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    eigenvaluelist = []
    for sam in father_sample:
        tmp_covariance = sam.T @ sam
        tmp_eigenvalue, _ = np.linalg.eig(tmp_covariance)
        tmp_eigenvalue = -np.sort(-tmp_eigenvalue)
        eigenvaluelist.append(tmp_eigenvalue)
    eigenvaluelist = np.log10(eigenvaluelist)
    
    data_index = np.tile(np.arange(eigenvaluelist.shape[1]), len(father_sample))
    circ_index = np.repeat(np.arange(len(father_sample)), eigenvaluelist.shape[1])

    df = pd.DataFrame(dict(index = data_index, CIRC = circ_index, PCA = eigenvaluelist.flatten()))
    
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row = "CIRC", hue = "CIRC", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map_dataframe(sns.lineplot, x = 'index', y = "PCA",alpha=1, linewidth=1.5)
    # g.map_dataframe(sns.lineplot, x = 'index', y = "PCA", color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "PCA")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.savefig("./analysis/FSDD_plot.png")


"""=================================================================================================="""
"""                                            验证网络的外延预测性                                    """
"""=================================================================================================="""

class ANET_extension():
    """
    1. 通过 gen_extension_sample 生成数据

    2. Gendata_multiprocess 构造数据集

    3. ANET_pred 预测数据

    4. compare 进行比较

    Warning: 由于每次都会偏移采样中心，所以这个东西存在错误！只有在 CIRC 0 是对的
    """
    def __init__(self, reactors: list[str] = None, producers: list[str] = None, circ: int = 0, setup_path: str = './settings/setup.yaml') -> None:
        super().__init__(reactors, producers, circ, basic_set = False, setup_path = setup_path)
        
        # extension 特殊部分
        self.model_current_json = f'./model/settings_circ={circ}.json'
        self.json_data = read_json_data(self.model_current_json)
        self.idt_net = load_best_dnn(DNN2, self.model_current_json, target = 'idt', device = 'cpu')
        self.APART_args['psr_mode'] = False

        # APART_ce 的 basic set
        self.set_FO(self.chem_args['fuel'], self.chem_args['oxidizer'])     # 设置燃料和氧化剂
        self.set_TPphi(self.chem_args['ini_T'], self.chem_args['ini_P'], self.chem_args['ini_phi'])  # 设置初始温度压强当量比采样点

        # 计算真实机理的IDT，并加载APART参数
        tmp_data = np.load('./data/true_data/true_ref_psr_data.npz')
        psr_condition = tmp_data['condition']
        RES_TIME_LIST = tmp_data['RES_TIME_LIST']
        self.set_psr_condition(psr_condition, RES_TIME_LIST) # 设置PSR参数
        self.true_data_path = './data/true_data'; self.reduced_data_path = "./data/APART_data/reduced_data"
        self.load_ref_data()

        # 增加到指定反应物的反应路径的反应
        _, tmp_eq_dict1 = yaml_key2A(self.APART_args['reduced_chem_path'], rea_keywords = self.reactors, pro_keywords = self.producers)
        _, tmp_eq_dict2 = yaml_key2A(self.APART_args['reduced_chem_path'], pro_keywords = self.reactors)
        
        self.eq_dict = dict(tmp_eq_dict1, **tmp_eq_dict2)
        Alist = np.array([])
        for key in self.eq_dict.keys():
            Alist = np.r_[Alist, np.squeeze(self.eq_dict[key])]
        
        self.A0 = Alist
        self.APART_args['eq_dict'] = self.eq_dict
        
        # 特殊部分: 用于单个CIRC 循环采样的判定
        self.sample_length = 0

    def gen_extension_sample(self, tiles = 50, sample_nums = 1e4):
        """
        生成向外延拓的数据，生成的方式为（以 r_alpha 为例子）在 [0,1] 的范围内，均匀划分成
        tiles 个数，随后再每个tiles里面采样sample_nums 个点
        将所有的点送到 gendata_multiprocess 里面计算
        随后再每个tiles里面计算平均预测误差绘制热力图
        其延拓体现在全局预测，看采样界周围那些误差就行了

        return: None
            self.origin_samples: (tiles, sample_nums, len(A)) 的一个 ndarray
            self.samples: (tiles * sample_nums, len(A)) 的一个 ndarray
        """
        self.GenAPARTDataLogger = Log("./log/extension_sample.log")
        alphas = np.linspace(-1, 1, tiles)
        self.samples = []
        for k in range(len(alphas) - 1):
            self.samples.append(
                sample_constant_A(
                    int(sample_nums),
                    self.A0,
                    alphas[k],
                    alphas[k+1],
                ).tolist()
            )
        self.origin_samples = np.array(self.samples)
        self.samples = self.origin_samples.reshape((-1, len(self.A0)))
    
    def ANET_pred(self, sample_true_idt = None,**kwargs):
        """
        预测 sample 的 IDT
        sample_true_idt: 样本点的 cantera 生成值，需要先从 Gendatamultiprocess 生成; 
                        若为None 则需要 extension data 补充
        kwargs:
            extension_data: 输入 gather data 之后生成的 extension data 文件，若为 None 直接从 self.sample 读取
            sample_size: (tiles, sample_nums)如果不经过 gen_extension_sample，则改变数据形状需要输入此参数
                         默认为 tiles = 50, sample_nums = 1e4
        """
        extension_data = kwargs.get("extension_data", None)
        if not extension_data is None:
            extension_data = np.load(extension_data)
            self.samples = extension_data['Alist']
            sample_true_idt = extension_data['all_idt_data']
            sample_size = kwargs.get("sample_size", (50, int(1e4)))
        else:
            sample_size = self.origin_samples.shape[0:2]

        pred = self.idt_net(torch.tensor(self.samples, dtype = torch.float32)).detach().numpy()
        pred_error = np.linalg.norm(pred - np.log10(sample_true_idt), ord = 2, axis = 1)
        pred_error = pred_error.reshape(sample_size)
        pred_error = np.mean(pred_error, axis = 1)
        return pred_error

    def extension_compare(self, pred_error, **kwargs):
        """
        输入所有 tile 的预测误差绘制热力图
        """
        pixel_per_bar = 4
        dpi = 200        
        l_alpha = int(len(pred_error) * (1 + self.APART_args['l_alpha'][self.circ]) / 2)
        r_alpha = int(len(pred_error) * (1 + self.APART_args['r_alpha'][self.circ]) / 2)
        fig, ax = plt.subplots(1,1,figsize=(8 * len(pred_error) * pixel_per_bar / dpi, 2), dpi = dpi)
        ax.spines.right.set_color('none')
        ax.spines.top.set_color('none')
        ax.spines.left.set_color('none')
        p = ax.imshow(pred_error.reshape(1, -1), cmap='RdBu', aspect='auto', interpolation='nearest')
        ax.set_xlabel(r"Sample Range" + "\n" + "\u25EE: left bound; \u25ED: right bound", {'fontsize':10},)
        ax.set_title(f"CRIC {self.circ} ANET average predict loss", {'fontsize':10}, 'left')
        ax.set_xticks([-1, l_alpha, int(len(pred_error)/2), r_alpha,len(pred_error)])
        ax.set_yticks([])
        ax.set_xticklabels(['-1', u'\u25EE', '0', u'\u25ED','1'], fontsize = 8) # 设置刻度标签
        
        fig.colorbar(p, ax = ax, pad = 1/dpi)
        fig.tight_layout(rect = (0,0,1.1,1))
        fig.savefig(kwargs.get("save_path", f"./analysis/CIRC{self.circ}_extension_compare.png"))

