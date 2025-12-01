from sympy import Q
from Apart_Package.APART_base import *
from Apart_Package.utils.cantera_utils import *
from Apart_Package.utils.setting_utils import *
from Apart_Package.utils.yamlfiles_utils import * 


class _DeePMO(APART_base):
    """DeePMO 函数公用核心部分"""

    def __init__(self, 
                 circ:int,
                 basic_set = True, 
                 setup_file: str = './settings/setup.yaml', 
                 cond_file: str = None,
                 SetAdjustableReactions_mode:int = None, 
                 IDT_csv_path:str = None,
                 LFS_csv_path:str = None,
                 **kwargs) -> None:
        
        ct.suppress_thermo_warnings()
        self.APART_args = get_yaml_data(setup_file, 'APART_args') | get_yaml_data(setup_file, 'APART_base') | kwargs

        # 设置整个体系中需要使用的 IDT 准则; 使用 PSR 或者 Mole 与否
        True_IDT_Cut_Time = kwargs.pop("idt_cut_time", 1); self.idt_cut_time = self.APART_args.get("True_IDT_Cut_Time", True_IDT_Cut_Time)
        cond_file = "./data/true_data/true_psr_data.npz" if cond_file is None else cond_file
        PSR_reduced_base = False if "PSR_reduced_base" not in self.APART_args else self.APART_args["PSR_reduced_base"]
        SetAdjustableReactions_mode = SetAdjustableReactions_mode if not SetAdjustableReactions_mode is None else self.APART_args.get("SetAdjustableReactions_mode", 0)
        IDT_csv_path = self.APART_args.get("IDT_csv_path", IDT_csv_path)
        LFS_csv_path = self.APART_args.get("LFS_csv_path", LFS_csv_path)
        
        super().__init__(circ = circ, setup_file = setup_file, cond_file = cond_file, PSR_reduced_base = PSR_reduced_base, **kwargs)
        self.circ = circ
        self.load_path()
        self.GenAPARTDataLogger = Log(f'./log/APART_GenData_circ={self.circ}.log')
        self.TrainAnetLogger = Log(f"./log/ANET_train_circ={self.circ}.log")
        self.InverseLogger = Log(f"./log/ANET_inverse_circ={self.circ}.log")
        
        
        # 特殊部分: 加载 A0
        # 增加到指定反应物的反应路径的反应并保存为 JSON 文件
        if not SetAdjustableReactions_mode is None:
            self.SetAdjustableReactions(
                mode = SetAdjustableReactions_mode,
                select_ratio = self.APART_args.get("SetAdjustableReactions_select_ratio", 0.3),
                **kwargs
            )
        # write_json_data(self.reduced_data_path + f"/eq_dict_circ={self.circ}.json", self.eq_dict)
        # 加载 IDT 相关真实与简化数据
        if not self.IDT_mode is False or 'IDT_csv_path' in self.APART_args:
            self.APART_args.update(idt_cut_time = self.idt_cut_time)
            self.LoadTrueData(self.true_data_path + '/true_idt.npz', mode = 'idt', idt_cut_time = self.idt_cut_time, csv_path = IDT_csv_path)
            self.APART_args['IDT_condition'] = self.IDT_condition
            self.APART_args['IDT_fuel'] = self.IDT_fuel
            self.APART_args['IDT_oxidizer'] = self.IDT_oxidizer
            self.APART_args['IDT_mode'] = self.IDT_mode
            self.GenAPARTDataLogger.info(f"LoadTrueData: IDT_condition_length = {len(self.IDT_condition)}; IDT_csv_path = {IDT_csv_path}")
            # 计算 reduced_idt_data 和 true_idt_data 的相对误差
            self.APART_args['IDT_relative_error'] = np.mean(np.abs((self.reduced_idt_data - self.true_idt_data) / self.true_idt_data)) * 100
            self.GenAPARTDataLogger.info(f"IDT_relative_error = {self.APART_args['IDT_relative_error']}")
        if hasattr(self, 'PSR_condition'):
            self.LoadTrueData(self.true_data_path + '/true_psr.npz', mode = 'psr')
            self.APART_args['PSR_condition'] = self.PSR_condition
            self.APART_args['RES_TIME_LIST'] = self.RES_TIME_LIST
            self.APART_args['PSR_fuel'] = self.PSR_fuel
            self.APART_args['PSR_oxidizer'] = self.PSR_oxidizer
            self.GenAPARTDataLogger.info(f"LoadTrueData: PSR_condition_length = {len(self.PSR_condition)}")
        if hasattr(self, 'LFS_condition') or 'LFS_csv_path' in self.APART_args:
            true_lfs_data_path = './data/true_data/true_lfs_data.npz'
            self.LoadTrueData(self.true_data_path + '/true_lfs.npz', mode = 'lfs', csv_path = LFS_csv_path, true_lfs_data_path = true_lfs_data_path)
            self.APART_args['LFS_condition'] = self.LFS_condition
            self.APART_args['LFS_fuel'] = self.LFS_fuel
            self.APART_args['LFS_oxidizer'] = self.LFS_oxidizer
            self.APART_args['LFS_relative_error'] = np.mean(np.abs((self.reduced_lfs_data - self.true_lfs_data) / self.true_lfs_data)) * 100
            self.GenAPARTDataLogger.info(f"LoadTrueData: LFS_condition_length = {len(self.LFS_condition)}; Relative_error = {self.APART_args['LFS_relative_error']}")
        if hasattr(self, 'PSR_concentration_condition') or 'PSR_concentration_csv_path' in self.APART_args:
            self.LoadTrueData(self.true_data_path + '/true_psr_concentration.npz', mode = 'PSR_concentration', csv_path = self.APART_args.get('PSR_concentration_csv_path', None))
            self.GenAPARTDataLogger.info(f"LoadTrueData: PSR_concentration_length = {len(self.PSR_concentration_condition)}")
       
        # 特殊部分: 加载 IDT 等的 weight
        IDT_weight = self.APART_args.get("IDT_weight", 1)
        PSR_weight = self.APART_args.get("PSR_weight", None)
        HRR_weight = self.APART_args.get("HRR_weight", None)
        LFS_weight = self.APART_args.get("LFS_weight", None)
        PSR_concentration_weight = self.APART_args.get("PSR_concentration_weight", None)
        self.init_weights(
            IDT_weight = IDT_weight, 
            PSR_weight = PSR_weight, 
            LFS_weight = LFS_weight, 
            HRR_weight = HRR_weight, 
            PSR_concentration_weight = PSR_concentration_weight
        )

    def init_weights(self, IDT_weight = None, PSR_weight = None, LFS_weight = None, HRR_weight = None, PSR_concentration_weight = None):
        """
        初始化 IDT, PSR, LFS, HRR, PSR_concentration 的权重
        """
        write_args = {}
        if IDT_weight is not None: 
            print(f"Using true_idt_data and true_idt_uncertainty to initialize IDT_weight")
            self.IDT_weight = IDT_weight * np.ones_like(self.true_idt_data) / self.true_idt_uncertainty
            self.GenAPARTDataLogger.info(f"IDT_weight initialized: {self.IDT_weight}")
            write_args['IDT_weight'] = self.IDT_weight.tolist()
        if PSR_weight is not None:
            if hasattr(self, 'true_psr_data'):
                self.PSR_weight = PSR_weight * np.ones_like(self.true_psr_data)
            elif hasattr(self, 'PSR_condition'):
                self.PSR_weight = PSR_weight * np.ones(self.PSR_condition.shape[0])
            else:
                raise ValueError("PSR_weight is None, but true_psr_data or true_psr_extinction_data is not set.")
            self.GenAPARTDataLogger.info(f"PSR_weight initialized: {self.PSR_weight}")
            write_args['PSR_weight'] = self.PSR_weight.tolist()
        if LFS_weight is not None:
            self.LFS_weight = LFS_weight * np.ones_like(self.true_lfs_data) / self.true_lfs_uncertainty
            self.GenAPARTDataLogger.info(f"LFS_weight initialized: {self.LFS_weight}")
            write_args['LFS_weight'] = self.LFS_weight.tolist()
        if HRR_weight is not None:
            self.HRR_weight = HRR_weight * np.ones_like(self.true_hrr_data)
            self.GenAPARTDataLogger.info(f"HRR_weight initialized: {self.HRR_weight}")
            write_args['HRR_weight'] = self.HRR_weight.tolist()
        if PSR_concentration_weight is not None:
            self.PSR_concentration_weight = PSR_concentration_weight * np.ones_like(self.true_psr_concentration_data)
            self.GenAPARTDataLogger.info(f"PSR_concentration_weight initialized: {self.PSR_concentration_weight}")
            write_args['PSR_concentration_weight'] = self.PSR_concentration_weight.tolist()
        self.WriteCurrentAPART_args(
            **write_args,
        )
        
    def LoadPreviousSetupIDT_condition(self, circ, json_name = None):
        """
        加载之前 CIRC 中的设置到 APART_args，因为设置可能发生了变化使得 setup.yaml 文件中的一些设定变的过时
        加载后将直接覆盖 __init__ 中设置的 self.APART_args 中的 IDT_condition
        直接覆盖 self.IDT_condition
        """
        if json_name is None: 
            if circ == 0:
                json_name = f"./model/model_pth/settings_circ={circ}.json"
            else:
                json_name = f"./model/model_pth/settings_circ={circ - 1}.json"
        json_data = read_json_data(json_name)
        self.APART_args.update(IDT_condition = json_data['IDT_condition'])
        self.IDT_condition = np.array(self.APART_args['IDT_condition'])
        self.GenAPARTDataLogger.info(f"LoadPreviousSetupIDT_condition FINISHED! IDT_condition_length = {len(self.IDT_condition)}")


    def LoadCurrentSetupIDT_condition(self, circ, json_name = None):
        """
        加载目前 CIRC 中的设置到 APART_args
        加载后将直接覆盖 __init__ 中设置的 self.APART_args
        直接覆盖 self.IDT_condition
        """
        if json_name is None: 
            json_name = f"./model/model_pth/settings_circ={circ}.json"
        json_data = read_json_data(json_name)
        self.APART_args.update(json_data)
        self.IDT_condition = np.array(self.APART_args['IDT_condition'])


    def SetAdjustableReactions(self, mode = 0, reserved_equations = None, 
                                save_jsonpath = None, logger = None, **kwargs):
        """
        设置允许调整的化学反应; 在 basic set 中调用
        存在以下设置方法
            0. 根据 kwargs 中参数键，自动化设置 mode
            1. 通过给定反应物列表，设置所有列表中物质作为反应物的反应为可调整反应
            2. 在 1 的基础上，增加列表中物质作为生成物的反应(可逆反应)
            3. 通过给定反应列表，设置所有列表中反应为可调整反应
            4. 通过灵敏度分析的结果，对不同 QoI 灵敏度归一化后求和，
                按照灵敏度和大小排序后设置前一定比例的反应为可调整反应
            
        params:
            mode: 设置模式，1-4
            reserved_equations: 需要单独保存的反应列表，mode = 1, 2, 4
            save_jsonpath: 保存的 json 文件路径，mode = 1, 2, 4
            kwargs: 
                reactors: mode = 1 需要指定的参数; 或者在 APART_args 中以 rea_keywords 指定
                reactors: mode = 2 需要指定的参数
                reactions: mode = 3 需要指定的参数
                mode = 4 需要指定的参数:
                    target_chem: 计算灵敏度的目标化学反应
                    select_ratio: 选择多少比例的反应用于调整
                    IDT_sensitivity, PSR_sensitivity, Mole_sensitivity, LFS_sensitivity: default: None
                    weight_IDT, weight_PSR, weight_Mole, weight_LFS: default: 1 筛选调整反应的权重
        
        requirement for APART_args:
            rea_keywords: mode = 1, 2
            reactions: mode = 3
            SetAdjustableReactions_select_ratio: mode = 4
        add:
            self.eq_dict, self.reduced_mech_A0, self.gen_yaml, self.APART_args['eq_dict']
        """
        if save_jsonpath is None: save_jsonpath = './data/APART_data/reduced_data/SetAdjustableReactions_Sensitivity.json'
        mkdirplus(os.path.dirname(save_jsonpath))
        # 如果 mode 为 0 需要根据 kwargs 中相应的参数是否存在设置 mode
        if mode == 0:
            if 'reactors' in kwargs or 'rea_keywords' in self.APART_args:
                mode = 1
            elif 'reactions' in kwargs or 'reactions' in self.APART_args:
                mode = 3
            elif 'select_ratio' in kwargs or 'SetAdjustableReactions_select_ratio' in self.APART_args:
                mode = 4
            else:
                mode = 2
                warnings.warn('mode is given as 0, but no reactors, reactions or select_ratio in kwargs, use mode = 2 and set reactors as None')
        match mode:
            case 1: # 通过给定反应物列表，设置所有列表中物质作为反应物的反应为可调整反应
                # 从 kwargs pop 出 reactors 或者从 self.APART_args 中读取 rea_keywords
                reactors = self.APART_args.get('rea_keywords', False)
                if not reactors: reactors = kwargs.pop('reactors', None)
                self.GenAPARTDataLogger.info(f"SetAdjustableReactions: mode = 1, reactors = {reactors}")
                _, eq_dict = yaml_key2A(self.reduced_mech, rea_keywords = reactors,)
            case 2: # 在 1 的基础上，增加列表中物质作为生成物的反应(可逆反应)
                reactors = self.APART_args.get('rea_keywords', False)
                if not reactors: reactors = kwargs.pop('reactors', None)
                self.GenAPARTDataLogger.info(f"SetAdjustableReactions: mode = 2, reactors = {reactors}")
                _, tmp_eq_dict1 = yaml_key2A(self.reduced_mech, rea_keywords = reactors,)
                _, tmp_eq_dict2 = yaml_key2A(self.reduced_mech, pro_keywords = reactors)
                eq_dict = dict(tmp_eq_dict1, **tmp_eq_dict2)
            case 3: # 通过给定反应列表，设置所有列表中反应为可调整反应
                ## 从字典 kwargs 中读取 reactions 若不存在报错 KeyError: When mode = 3, reactions must be given!
                reactions = self.APART_args.get('reactions', False)
                if not reactions: reactions = kwargs.pop('reactions', None)
                if reactions is None: raise KeyError("When mode = 3, reactions must be given!")
                _, eq_dict = yaml_eq2A(self.reduced_mech, *reactions)
            case 4:
                ## 从字典 kwargs 中读取 IDT_sensitivity, PSR_sensitivity, Mole_sensitivity, LFS_sensitivity
                target_chem = kwargs.pop('target_chem', self.reduced_mech)
                select_ratio = kwargs.pop('select_ratio', 0.3)
                IDT_sensitivity = kwargs.pop('IDT_sensitivity', None); weight_IDT = kwargs.pop('weight_IDT', 1)
                PSR_sensitivity = kwargs.pop('PSR_sensitivity', None); weight_PSR = kwargs.pop('weight_PSR', 1)
                if not self.IDT_mode is False: 
                    if IDT_sensitivity is None: 
                        ## 计算所有反应关于 IDT 的灵敏度
                        IDT_sensitivity = yaml2idt_sensitivity(
                            target_chem,
                            IDT_condition = self.IDT_condition,
                            fuel = self.IDT_fuel, oxidizer = self.IDT_oxidizer,
                            mode = self.IDT_mode, save_path = save_jsonpath
                        )
                        # IDT_sensitivity 内所有 value 取绝对值后求平均值，替换原来的位置
                        IDT_sensitivity = {k: np.mean(np.abs(v)) * weight_IDT for k, v in IDT_sensitivity.items()}
                        # 所有的 value 标准化: value - min(value) / (max(value) - min(value))
                        IDT_sensitivity = {k: (v - min(IDT_sensitivity.values())) / (max(IDT_sensitivity.values()) - min(IDT_sensitivity.values())) for k, v in IDT_sensitivity.items()}
                        # # 根据 value 降序排序
                        # IDT_sensitivity = dict(sorted(IDT_sensitivity.items(), key = lambda item: item[1], reverse = True))
                        write_json_data(save_jsonpath, {"IDT_sensitivity": IDT_sensitivity})
                    Sensitivity = copy.deepcopy(IDT_sensitivity)
                if not self.PSR_mode is False:
                    if PSR_sensitivity is None:
                        ## 计算所有反应关于 PSR 的灵敏度
                        PSR_sensitivity = yaml2psr_sensitivity(
                            target_chem,
                            PSR_condition = self.PSR_condition,
                            RES_TIME_LIST = self.RES_TIME_LIST,
                            fuel = self.PSR_fuel, oxidizer = self.PSR_oxidizer,
                            mode = self.PSR_mode, save_path = save_jsonpath
                        )
                        # PSR_sensitivity 内所有 value 取绝对值后求平均值，替换原来的位置
                        PSR_sensitivity = {k: np.mean(np.abs(v)) * weight_PSR for k, v in PSR_sensitivity.items()}
                        # 所有的 value 标准化: value - min(value) / (max(value) - min(value))
                        PSR_sensitivity = {k: (v - min(PSR_sensitivity.values())) / (max(PSR_sensitivity.values()) - min(PSR_sensitivity.values())) for k, v in PSR_sensitivity.items()}
                        # # 根据 value 降序排序
                        # PSR_sensitivity = dict(sorted(PSR_sensitivity.items(), key = lambda item: item[1], reverse = True))
                        write_json_data(save_jsonpath, {
                                    "PSR_sensitivity": PSR_sensitivity, 
                                    "IDT_sensitivity": IDT_sensitivity
                                    })
                # Sensitivity 内所有的 key 对应的 value 相加 PSR_sensitivity 的 value
                Sensitivity = {k: Sensitivity[k] + v for k, v in PSR_sensitivity.items()}
                # 根据 value 降序排序 Sensitivity
                Sensitivity = dict(sorted(Sensitivity.items(), key = lambda item: item[1], reverse = True))
                # 选取 Sensitivity 中 value 最大的 select_ratio * len(Sensitivity) 个 key 作为 target_reactions
                target_reactions = list(Sensitivity.keys())[: int(select_ratio * len(Sensitivity))]
                _, eq_dict = yaml_eq2A(self.reduced_mech, *target_reactions)
        if not reserved_equations is None:
            # 单独获得 reserved_equations 的 eq_dict
            _, reserved_eq_dict = yaml_eq2A(self.reduced_mech, *reserved_equations)
            # 将 reserved_eq_dict 与 eq_dict 合并
            eq_dict.update(reserved_eq_dict)

        Alist = eq_dict2Alist(eq_dict); self.eq_dict = eq_dict    
        self.reduced_mech_A0 = Alist
        self.APART_args['eq_dict'] = self.eq_dict


    def WriteCurrentAPART_args(self, save_json = None, cover = False, **kwargs):
        """
        保存当前下的 APART_args 中的参数
            kwargs: 存放参数用于 update APART_args 中的 key-value
        rewrite: 
            self.APART_args
        """
        if save_json is None: save_json = self.model_current_json
        self.APART_args.update(kwargs)
        write_json_data(save_json, self.APART_args, cover = cover)


    def load_path(self):

        self.true_data_path = mkdirplus('./data/true_data')
        mkdirplus('./data/APART_data')
        mkdirplus('./data/APART_data/tmp')
        mkdirplus('./data/APART_data/ANET_data')
        mkdirplus('./data/APART_data/reduced_data')

        # 生成存放网络数据的文件夹
        mkdirplus('./model')
        mkdirplus('./log')   
        self.model_path = mkdirplus('./model/model_pth')
        self.model_loss_path = mkdirplus('./model/loss_his')
        self.model_current_json = f'{self.model_path}/settings_circ={self.circ}.json'
        if self.circ > 0:
            self.model_previous_json = f'{self.model_path}/settings_circ={self.circ - 1}.json'

        self.apart_data_path = f'./data/APART_data/apart_data_circ={self.circ}.npz'
        self.reduced_data_path = './data/APART_data/reduced_data'


    def cal_samples_quality(self, Alist_L2_benchmark, Alist_L2penalty = 0.0001):
        """
        计算采样点的质量，包括 IDT, PSR, Mole, LFS. 计算方法为将所有的 QoI 误差加权求和并截取最优的 20 个样本作为本轮的样本质量，同时增加自变量和初始值的 penalty
        通过和最初的样本质量对比，修正下一步的采样范围
        请在第一次循环之后使用，在求得采样范围时计算
        return previous_error / circ0_error
        """
        if self.circ == 0:
            warnings.warn("cal_samples_quality can only be used after the first and second circulation")
            return 1
        previous_samples = np.load(f'./data/APART_data/apart_data_circ={self.circ - 1}.npz')
        previous_Alist = previous_samples['Alist']
        # 加载 circ=0 的数据
        circ0_samples = np.load(f'./data/APART_data/apart_data_circ=0.npz')
        circ0_Alist = circ0_samples['Alist']
        previous_error, circ0_error = 0, 0
        
        if self.IDT_mode is not None:
            previous_IDT_data = previous_samples['all_idt_data']
            previous_IDT_data = np.log10(previous_IDT_data)
            benchmark_IDT_data = np.log10(self.true_idt_data)
            circ0_IDT_data = circ0_samples['all_idt_data']
            circ0_IDT_data = np.log10(circ0_IDT_data)
            previous_error += np.linalg.norm(self.IDT_weight * (previous_IDT_data - benchmark_IDT_data), ord = np.inf, axis = -1)
            circ0_error += np.linalg.norm(self.IDT_weight * (circ0_IDT_data - benchmark_IDT_data), ord = np.inf, axis = -1)
        if hasattr(self, 'PSR_condition') and hasattr(self, 'true_psr_data'):
            previous_PSR_data = previous_samples['all_psr_data']
            benchmark_PSR_data = self.true_psr_data
            circ0_PSR_data = circ0_samples['all_psr_data']
            previous_error += np.linalg.norm(self.PSR_weight * (previous_PSR_data - benchmark_PSR_data), ord = np.inf, axis = -1)
            circ0_error += np.linalg.norm(self.PSR_weight * (circ0_PSR_data - benchmark_PSR_data), ord = np.inf, axis = -1)
        if hasattr(self, 'LFS_condition'):
            previous_LFS_data = previous_samples['all_lfs_data']
            benchmark_LFS_data = self.true_lfs_data
            circ0_LFS_data = circ0_samples['all_lfs_data']
            previous_error += np.linalg.norm(self.LFS_weight * (previous_LFS_data - benchmark_LFS_data), ord = np.inf, axis = -1)
            circ0_error += np.linalg.norm(self.LFS_weight * (circ0_LFS_data - benchmark_LFS_data), ord = np.inf, axis = -1)

        # 计算 previous 和 circ0 的 Alist 误差
        previous_Alist_error = np.linalg.norm(previous_Alist - Alist_L2_benchmark, ord = np.inf)
        circ0_Alist_error = np.linalg.norm(circ0_Alist - Alist_L2_benchmark, ord = np.inf)
        # 误差求和
        previous_error += previous_Alist_error * Alist_L2penalty * np.linalg.norm(previous_IDT_data - benchmark_IDT_data, ord = np.inf)
        circ0_error += circ0_Alist_error * Alist_L2penalty * np.linalg.norm(circ0_IDT_data - benchmark_IDT_data, ord = np.inf)
        
        # 排序并求出 error 最小的前 20 个 error 的平均值
        previous_error = np.mean(np.sort(previous_error)[:20])
    
        circ0_error = np.mean(np.sort(circ0_error)[:20])
        # 返回 previous_error 和 circ0_error
        return previous_error / circ0_error


def PSRex_shrink_error(all_psr_extinction_data:np.ndarray, true_psrex_data:np.ndarray = None, ord = np.inf, return_Rlos = False) -> np.ndarray:
    """
    在计算 PSRex 的误差时，有时我们会容许好的采样点 PSRex 的值小于详细机理的 PSRex 值。这是由于两方面原因：
    1. 如此做可以大大简化优化过程步骤，使得优化更加快速
    2. 一般我们设置的 true_psrex_data 小于真实的情况，这样做可以使得我们的模型更加保守，更加安全
    但是我们同样不希望采样点的值小于 true_psrex_data 的值太多，因此我们需要一个函数来判断这种情况
    我们采取以下策略：
        a. all_psr_extinction_data - true_psrex_data in [-0.25, 0] 时，认定 loss = 0
        b. all_psr_extinction_data - true_psrex_data in [-0.5, -0.25] 时，认定 loss 加权系数为 1/5
        b. all_psr_extinction_data - true_psrex_data in (-inf, -0.5) 时，认定 loss 加权系数为 2/5
        c. all_psr_extinction_data - true_psrex_data in (0, inf) 时，认定 loss 加权系数为 1
    params:
        all_psr_extinction_data: 采样点的log2 PSRex 值
        true_psrex_data: 真实的log2 PSRex 值
    return:
        psrex_shrink_error: PSRex 的误差
    """
    psrex_shrink_error = np.zeros_like(all_psr_extinction_data)
    psrex_shrink_error[(all_psr_extinction_data - true_psrex_data <= 0) & (all_psr_extinction_data - true_psrex_data > -0.125)] = 0
    psrex_shrink_error[(all_psr_extinction_data - true_psrex_data > -0.25) & (all_psr_extinction_data - true_psrex_data <= -0.125)] = 1 / 5
    psrex_shrink_error[(all_psr_extinction_data - true_psrex_data > -0.5) & (all_psr_extinction_data - true_psrex_data <= -0.25)] = 2 / 5
    psrex_shrink_error[all_psr_extinction_data - true_psrex_data <= -0.5] = 3 / 5
    psrex_shrink_error[all_psr_extinction_data - true_psrex_data > 0] = 6 / 5

    if return_Rlos:
        return psrex_shrink_error * np.abs(all_psr_extinction_data - true_psrex_data)
    elif len(all_psr_extinction_data.shape) == 1:
        return np.linalg.norm(psrex_shrink_error * np.abs(all_psr_extinction_data - true_psrex_data), ord = ord)
    else:
        return np.linalg.norm(psrex_shrink_error * np.abs(all_psr_extinction_data - true_psrex_data), ord = ord, axis = 1)
    

