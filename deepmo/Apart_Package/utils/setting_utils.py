# -*- coding:utf-8 -*-
import os, yaml, json, re, logging
import numpy as np, numpy.ma as ma
import torch


def mkdirplus(dir_path):
    if not dir_path is None:
        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except:
                os.makedirs(dir_path)
        return dir_path



def get_yaml_data(yaml_file, key_name = None):
    file = open(yaml_file, 'r', encoding="utf-8")         # 获取yaml文件数据
    data = yaml.load(file.read(), Loader=yaml.FullLoader) # 将yaml数据转化为字典
    file.close()
    if key_name is None:
        tmp_dict = {}
        [tmp_dict.update(data[key]) for key in data.keys()]
        return tmp_dict
    return data[key_name]


def read_json_data(json_file_name):
    with open(json_file_name, 'r', encoding='utf8') as fp:
        args = json.load(fp)
    fp.close()
    return args


def write_json_data(json_file_name, args, cover = False):
    """
    if cover = True, We will ignore the original json file and create a new one
    """
    if os.path.exists(json_file_name) and not cover:
        with open(json_file_name, "r", encoding="utf-8") as f:
            try:
                old_data = json.load(f)
            except:
                old_data = {}
            old_data.update(args); args = old_data
    # 调整ndarray 为 list
    for key in args.keys():
        if isinstance(args[key], np.ndarray):
            args[key] = args[key].tolist()
        if isinstance(args[key], (np.integer, np.floating, np.bool_)):
            args[key] = args[key].item()
    with open(json_file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(args, ensure_ascii=False, indent=4, separators=(',', ':')))
    f.close()

def get_best_pth(model_path):
    Data = read_json_data(f'{model_path}/settings.json')
    best_epoch = int(Data['stop_epoch'])
    return f'{model_path}/model_pth/model{best_epoch}.pth'


def load_best_dnn(model, model_json_path, device = 'cuda', model_pth_path = None, target:str = None,
                  prefix:str = None, suffix:str = None, DNN_instantiation_keys = None,) -> torch.nn.Module:
    """
    加载 early stopping 中最好的 DNN 
    params:
        model_json_path: json 文件位置
        model_pth_path: 模型的 pth 文件位置, 默认为None, 会自动读取 json 文件中的 best_ppth
        device: cpu or cuda
        target: 如果是多任务学习，需要指定是哪个任务的模型
        prefix: 如果是多输入模型，需要指定是哪个输入的模型
        suffix: 如果是多输出模型，需要指定是哪个输出的模型
        DNN_instantiation_keys: 不使用上述设置, 如果需要重新实例化 DNN, 则需要传入 DNN 的参数keys:
            请按照以下顺序输入: 
            [model_best_ppth, *(model 实例化需要的参数排列)]
    return:
        DNN
    """
    json_data = read_json_data(model_json_path)
    if DNN_instantiation_keys is not None:
        model_pth_path = json_data[DNN_instantiation_keys[0]] if model_pth_path is None else model_pth_path
        model_instantiation_args = [json_data[key] for key in DNN_instantiation_keys[1:]]
        my_model = model(*model_instantiation_args).to(device)
    else:
        if (target is None) and (prefix is None) and (suffix is None):
            model_pth_path = json_data[f'best_ppth'] if model_pth_path is None else model_pth_path
            my_model = model(json_data['input_dim'], json_data['hidden_units'], json_data['output_dim']).to(device)
        elif target is not None:
            model_pth_path = json_data[f'best_{target}_ppth'] if model_pth_path is None else model_pth_path
            my_model = model(json_data['input_dim'], json_data['hidden_units'], json_data[f'{target}_output_dim']).to(device)
        elif prefix is not None:
            model_pth_path = json_data[f'{prefix}_best_ppth'] if model_pth_path is None else model_pth_path
            my_model = model(json_data[f'{prefix}_input_dim'], json_data['hidden_units'], json_data[f'{prefix}_output_dim']).to(device)
        elif suffix is not None:
            model_pth_path = json_data[f'best_ppth_{suffix}'] if model_pth_path is None else model_pth_path
            my_model = model(json_data[f'input_dim_{suffix}'], json_data['hidden_units'], json_data[f'output_dim_{suffix}']).to(device)
    checkpoint = torch.load(model_pth_path, map_location = device)
    my_model.load_state_dict(checkpoint['model']) # 实例化网络
    return my_model


def load_best_dnn_optim(optim, model_json_path, target = 'idt', device = 'cuda'):
    json_data = read_json_data(model_json_path)
    model_pth_path = json_data[f'best_{target}_ppth']
    checkpoint = torch.load(model_pth_path, map_location = device)
    optim.load_state_dict(checkpoint['optimizer']) # 实例化网络
    return optim

# 加载 root_file_name 下所有含有 matchstr 的文件，返回的是 generator
def read_all_files(root_file_name, matchstr):
    for filewalk in os.walk(root_file_name):
        for filename in filewalk[2]:
            if matchstr in filename:
                yield os.path.join(filewalk[0], filename)


# 在某文件夹内匹配文件开头后按照文件名最后一个数字的顺序读取文件
def get_file_list(dir, file_head):
    file_list = []
    for file in os.listdir(dir):
        if file.startswith(file_head):
            file_list.append(file)
    file_list = sorted(file_list, key = lambda x: int(x.split('=')[-1].split('.')[0]))
    file_list = [dir + '/' + tmp_file_list for tmp_file_list in file_list ]
    return file_list


# find the files in the dir "./model/model_pth/" which are start with "model_"
# and end with ".pth"
# and sort them according to the number after "circ="
# and return the file name
def find_pth_files(dir, file_start = "model_best_stopat", file_type = ".pth"):
    files = os.listdir(dir)
    files = [file for file in files if re.match(file_start, file) and file.endswith(file_type)]
    files.sort(key = lambda x: int(re.findall(r"circ=(\d+)", x)[0]))
    files = [dir + '/' + file for file in files]
    return files


def flatten_data_permutation(*data:np.ndarray, **kwargs):
    """
    交换 flatten data 的顺序
    将输入的 data 逐个按照 input_size 来 reshape 后按照 swap axis 来交换维度，最后再 flatten
    """
    input_size = kwargs.get('input_size', None)
    swap_axis = kwargs.get('swap_index', (0,1,2))
    return_data = []
    for datum in data:
        datum = datum.reshape(input_size)
        datum = datum.transpose(swap_axis)
        return_data.append(datum.flatten())
    return return_data


def zscore(data:np.ndarray, mask_on_threshold = None):
    """
    zscore 标准化
    params:
        data: 需要标准化的数据
        mask_on_threshold: 如果不为 None, 则会将 data 中小于 mask_on_threshold 的数据标记为 numpy.ma.masked
    return:
        data, mean, std
    """
    if mask_on_threshold is not None:
        data:ma.MaskedArray = ma.masked_less(data, mask_on_threshold)
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        if np.any(std == 0):
            std = np.ones_like(std)
        data = (data - mean) / std
        # 将 mask 的部分还原为 -10
        data = data.filled(fill_value = -10)
    else:
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        if np.any(std == 0):
            std = np.ones_like(std)
        data = (data - mean) / std

    return data, mean, std


def reverse_zscore(data:np.ndarray, mean:np.ndarray, std:np.ndarray, replace_maskon_items = None):
    """
    反标准化
    params:
        data: 需要反标准化的数据
        mean: 均值
        std: 标准差
        replace_maskon_items: 针对 data 中的 -1 元素，会使用 replace_maskon_items 来替换；替换遵循 broadcast 规则
            e.g.
                data = np.array([[1,2,3], [4,5,6], [-1,8,-1]])
                replace_maskon_items = np.array([0,1,2])
                reverse_zscore(data, mean, std, replace_maskon_items)
                >>> array([[ 1.        ,  2.        ,  3.        ],
                            [ 4.        ,  5.        ,  6.        ],
                            [0           ,  8.        , 2         ]], dtype=float32)
    return:
        data
    """
    # 替换 mean 中的 None/Null 值为 0; std 中的 None/Null 值为 1
    mean = np.where(np.array(mean) == None, 0, mean).astype(np.float32)
    std = np.where(np.array(std) == None, 1, std).astype(np.float32)
    # 检测 data 中是否有 -1 的元素
    if np.any(data <= -0.1):
        assert replace_maskon_items is not None, "replace_maskon_items can not be None when data contains -1"
        # 将 data 中的 -1 mask 掉
        data = ma.masked_less(data, -0.1)
        # zscore 还原
        data = data * std + mean
        # 还原为 numpy.ndarray
        data = data.filled(fill_value = -10)
        # 将 data 中的 -1 替换为 replace_maskon_items; 遵循 broadcast 规则
        data = np.where(data == -10, replace_maskon_items, data)
    else:
        data = data * std + mean
    return data


def DataNormalizationProcess(data:np.ndarray, method:str = 'None', mask_on_threshold = None, BCT_lambda = 0.15, **kwargs):
    """
    数据标准化流程，继承了不标准化、log、BCT、 zscore 和 minmax 5种方法
    params:
        data: 需要标准化的数据
        method: 标准化方法; 可选项为 'zscore', 'minmax', 'log', 'BCT', 'None'
        mask_on_threshold: 如果不为 None, 则会将 data 中小于 mask_on_threshold 的数据标记为 numpy.ma.masked
        kwargs: 用于传递给 zscore 和 minmax 的参数
    return:
        zscore: data, mean, std
        minmax: data, min_value, max_value
        log: log10(data), 0, 1
        BCT: BCT(data), 0, 1
        None: data, 0, 1
    """
    if method == 'zscore':
        return zscore(data, mask_on_threshold = mask_on_threshold)
    elif method == 'minmax':
        # 计算 data 按列的最大最小值
        max_value = np.max(data, axis = 0); min_value = np.min(data, axis = 0)
        # 将 data 中小于 mask_on_threshold 的数据标记为 numpy.ma.masked
        if mask_on_threshold is not None:
            data:ma.MaskedArray = ma.masked_less(data, mask_on_threshold)
            # minmax 标准化
            data = (data - min_value) / (max_value - min_value)
            # 将 mask 的部分还原为 -10
            data = data.filled(fill_value = -10)
        else:
            data = (data - min_value) / (max_value - min_value)
        return data, min_value, max_value
    elif method == 'log':
        return np.log10(data), 0, 1
    elif method == 'BCT':
        # 进行 BCT 变换
        def BoxCoxTransform(data:np.ndarray, lmbda:float = 0.15):
            if lmbda == 0:
                return np.log(data)
            return (data ** lmbda - 1) / lmbda
        # 计算 data 的BCT变化
        data = BoxCoxTransform(data, lmbda = BCT_lambda)
        return data, 0, 1
    elif method == 'None':
        return data, 0, 1


def ReverseDataNormalizationProcess(data:np.ndarray, method:str = 'None', mean = None, std = None, min_value = None, max_value = None, replace_maskon_items = None, BCT_lambda = 0.15):
    """
    数据反标准化流程，继承了不标准化、log、BCT、 zscore 和 minmax 5种方法
    params:
        data: 需要标准化的数据
        method: 标准化方法; 可选项为 'zscore', 'minmax', 'log', 'BCT', 'None'
        mean: 均值
        std: 标准差
        min_value: 最小值
        max_value: 最大值
        replace_maskon_items: 针对 data 中的 -1 元素，会使用 replace_maskon_items 来替换；替换遵循 broadcast 规则
            e.g.
                data = np.array([[1,2,3], [4,5,6], [-1,8,-1]])
                replace_maskon_items = np.array([0,1,2])
                reverse_zscore(data, mean, std, replace_maskon_items)
                >>> array([[ 1.        ,  2.        ,  3.        ],
                            [ 4.        ,  5.        ,  6.        ],
                            [0           ,  8.        , 2         ]], dtype=float32)
        BCT_lambda: BCT 变换的参数
    return:
        zscore: data
        minmax: data
        log: exp(data)
        BCT: BCT(data)
        None: data
    """
    if method == 'zscore':
        return reverse_zscore(data, mean, std, replace_maskon_items = replace_maskon_items)
    elif method == 'minmax':
        # 替换 mean 中的 None/Null 值为 0; std 中的 None/Null 值为 1
        min_value = np.where(np.array(min_value) == None, 0, min_value).astype(np.float32)
        max_value = np.where(np.array(max_value) == None, 1, max_value).astype(np.float32)
        # 检测 data 中是否有 -1 的元素
        if np.any(data <= -0.1):
            assert replace_maskon_items is not None, "replace_maskon_items can not be None when data contains -1"
            # 将 data 中的 -1 mask 掉
            data = ma.masked_less(data, -0.1)
            # minmax 还原
            data = data * (max_value - min_value) + min_value
            # 还原为 numpy.ndarray
            data = data.filled(fill_value = -10)
            # 将 data 中的 -1 替换为 replace_maskon_items; 遵循 broadcast 规则
            data = np.where(data == -10, replace_maskon_items, data)
        else:
            data = data * (max_value - min_value) + min_value
        return data
    elif method == 'log':
        return 10 ** np.array(data)
    elif method == 'BCT':
        # 进行 BCT 变换
        def InverseBoxCoxTransform(data:np.ndarray, lmbda:float = 0.15):
            if lmbda == 0:
                return np.exp(data)
            return (data * lmbda + 1) ** (1 / lmbda)
        # 计算 data 的BCT变化
        data = InverseBoxCoxTransform(data, lmbda = BCT_lambda)
        return data
    elif method == 'None':
        return data

"""======================================================================================================"""
"""                                                        日志类                                         """
"""======================================================================================================"""

# 可以追加输入的日志
# 调用方式为 my_logger = Log('./my_logger.log')
# 若不存在则创建，若存在则追加写入
class Log:
    def __init__(self, file_name, mode = 'a'):
        # 第一步，创建一个logger
        self.logger = logging.getLogger(file_name)  # file_name为多个logger的区分唯一性
        self.logger.setLevel(logging.DEBUG)  # Log等级总开关
        # 如果已经有handler，则用追加模式，否则直接覆盖
        # mode = 'a' if self.logger.handlers else 'w'
        # 第二步，创建handler，用于写入日志文件和屏幕输出
        fmt = "%(asctime)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt)
        # 文件输出
        fh = logging.FileHandler(file_name, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        fh.setFormatter(formatter)
        # # # 往屏幕上输出
        # sh = logging.StreamHandler()
        # sh.setFormatter(formatter)  # 设置屏幕上显示的格式
        # sh.setLevel(logging.DEBUG)
        # 先清空handler, 再添加
        self.logger.handlers = []
        self.logger.addHandler(fh)
        # self.logger.addHandler(sh)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
