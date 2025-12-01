from Apart_Package.APART_base import *
from Apart_Package.DeePMO_V1.DeePMO_IDT import *


class DeePMO_without_DNNfilter(DeePMO_IDT):
    """
    DeePMO without DNN filter: DeePMO_IDT 的基础上去掉 DNN 筛选部分，以确定 DNN 筛选在整个体系中的重要性。
    DNN filter 的消融实验。
    实验中使用优秀样本分布情况来对比 DNN filter 的效果。
    """
    def __init__(self, circ=0, setup_file: str = './settings/setup.yaml',
                 SetAdjustableReactions_mode: int = None, previous_best_chem: str = None, GenASampleRange=None, GenASampleRange_mode=None, **kwargs) -> None:
        super().__init__(circ, setup_file, SetAdjustableReactions_mode, previous_best_chem, GenASampleRange, GenASampleRange_mode, **kwargs)
        
        
    def ASample(self, sample_size = None, coreAlist = None, passing_rate_upper_limit = None, 
                IDT_reduced_threshold = None, cluster_ratio = False, father_sample_save_path = None, 
                start_circ = 0, **kwargs):
        """
        重写了 APART.ASample_IDT 函数，主要是为了增加针对不同反应调节采样范围的函数
        202230730: 增加了通过率上限 passing_rate_upper_limit,
        """
        np.set_printoptions(precision = 2, suppress = True); t0 = time.time()

        # 预设置
        self.GenAPARTDataLogger.info(f"Start The ASample Process; Here we apply three aspect into consideration: IDT: True")
        # 提取采样的左右界限 + 采样阈值

        # 检测类中是否存在 self.idt_threshold
        if not hasattr(self, 'idt_threshold'):
            idt_threshold = self.APART_args['idt_threshold']
            self.idt_threshold = np.array(idt_threshold)[self.circ - 1] if isinstance(idt_threshold, Iterable) else idt_threshold

        sample_size = self.APART_args['sample_size'] if sample_size is None else sample_size
        core_size = self.APART_args.get('father_sample_size', 1)
        passing_rate_upper_limit = self.APART_args.get('passing_rate_upper_limit', 0.5) if passing_rate_upper_limit is None else passing_rate_upper_limit

        sample_size = np.array(sample_size)[self.circ] if isinstance(sample_size, Iterable) else sample_size    
        core_size = int(np.array(core_size)[self.circ]) if isinstance(core_size, Iterable) else int(core_size)
        
        # 将YAML文件的A值提取并构建均匀采样点
        if self.circ == 0 or self.circ == start_circ:
            self.samples = sample_constant_A(sample_size, self.reduced_mech_A0, self.l_alpha, self.r_alpha)

        else:
            self.GenAPARTDataLogger.info(f"idt_threshold: log10({self.idt_threshold}) = {np.log10(self.idt_threshold)}; sample_size: {sample_size}; father sample size: {core_size}")
            self.GenAPARTDataLogger.info("="*100)

            # 读取之前的 apart_data.npz 从中选择前 1% 的最优采样点作为核心; 依然选择 IDT 作为指标，不涉及 

            if coreAlist is None:
                cluster_weight = kwargs.get('cluster_weight', 0.1)
                previous_coreAlist = self.SortALIST(
                        apart_data_path = os.path.dirname(self.apart_data_path) + f'/apart_data_circ={self.circ - 1}.npz',
                        experiment_time = core_size, IDT_reduced_threshold = IDT_reduced_threshold, 
                        father_sample_save_path = father_sample_save_path, 
                        logger = self.GenAPARTDataLogger, cluster_ratio = cluster_ratio, cluster_weight = cluster_weight, **kwargs)
                previous_eq_dict = read_json_data(os.path.dirname(self.apart_data_path) + f'/reduced_data/eq_dict_circ={self.circ - 1}.json')
                coreAlist = []
                for A0 in previous_coreAlist:
                    previous_eq_dict = Alist2eq_dict(A0, previous_eq_dict)
                    # 将 self.eq_dict 中与 previous_eq_dict 相同的项替换为 previous_eq_dict 中的值
                    tmp_eq_dict = {
                        key: previous_eq_dict[key] if key in previous_eq_dict else self.eq_dict[key] for key in self.eq_dict.keys()
                    }
                    coreAlist.append(eq_dict2Alist(tmp_eq_dict))
            else:
                core_size = len(coreAlist)
            
            coreAlist = np.array(coreAlist)
            self.best_sample = coreAlist[0,:]; np.save(os.path.dirname(self.apart_data_path) + 
                                            f"/best_sample_circ={self.circ}.npy", self.best_sample)
                
            t0 = time.time()
            self.samples = []; tmp_sample_size = int(2 * (sample_size) // core_size)  
            while len(self.samples) < sample_size:
                # 每次采样的样本点不能太少，也不能太多；因此创建自适应调节机制
                if tmp_sample_size >= sample_size * 0.002:
                    tmp_sample_size = int(2 * (sample_size - len(self.samples)) // core_size)
                else:
                    tmp_sample_size = int(2 * sample_size // core_size)
                for A0 in coreAlist:
                    tmp_sample = sample_constant_A(sample_size, A0, self.l_alpha, self.r_alpha)
                    self.samples.extend(tmp_sample.tolist())

                    # 样本太多立即退出
                    if len(self.samples) >= sample_size * 1.2:
                        self.GenAPARTDataLogger.warning(f"IDT: Stop the Sampling Process, Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                        self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
                        break

                self.GenAPARTDataLogger.info(f"In this Iteration, IDT: Total Size has come up to {len(self.samples)} data after this iter! cost {time.time() - t0:.2f}s")  
                self.GenAPARTDataLogger.info(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024} GB")
        
        self.samples = np.array(self.samples)
        self.father_samples = coreAlist
        np.save(f'./data/APART_data/Asamples_{self.circ}.npy', self.samples)  
        # 更新 idt_threshold
        self.APART_args['idt_threshold'] = self.idt_threshold
        self.APART_args['passing_rate_upper_limit'] = passing_rate_upper_limit
        self.APART_args['sample_size'] = sample_size
        # 保存 APART_args 到 self.current_json_path
        self.WriteCurrentAPART_args(cover = True)
        
        self.GenAPARTDataLogger.info(f"End The ASample_IDT Progress! The size of samples is {len(self.samples)}, cost {time.time() - t0:.2f}s")  
        return self.samples
    
    
    def SkipSolveInverse(self, father_sample:str = None, save_dirpath = f'./inverse_skip', 
                             csv_path = None, device = 'cpu', IDT_reduced_threshold = None, raw_data = False, 
                             experiment_time = 15, **kwargs):
        """
        自动跳过 Inverse 部分直接使用最优样本就可以拿到结果，可以直接放在训练步骤之后使用
        """
        np.set_printoptions(suppress=True, precision=3)

        save_folder = mkdirplus(save_dirpath)
        # 加载最优的样本
        if not father_sample is None and os.path.exists(father_sample):
            tmp_father_sample = np.load(father_sample)
            inverse_alist = tmp_father_sample['Alist']
        else:
            inverse_alist = self.SortALIST(self.apart_data_path, experiment_time = experiment_time,
                                    IDT_reduced_threshold = IDT_reduced_threshold,)
        # 加载网络
        optim_net = load_best_dnn(Network_PlainSingleHead, self.model_current_json, device = device)
        for index in range(experiment_time):
            self.InverseLogger.info(f'experiment_index: {index}')
            inverse_path = mkdirplus(save_folder + f'/{index}')
            try:  
                # IDT图像部分
                t1 = time.time()
                # 生成初值机理
                A_init = np.array(inverse_alist[index], dtype = np.float64)
                Adict2yaml(eq_dict = self.eq_dict, original_chem_path = self.reduced_mech, chem_path = inverse_path +'/optim_chem.yaml', Alist = A_init)
                # 根据 A_init 查找 self.apart_data_path 中的数据
                apart_data = np.load(self.apart_data_path)
                Alist_data = apart_data['Alist']; idt_data = apart_data['all_idt_data']
                # 查找 Ainit 的 index
                index = np.where(np.all(Alist_data == A_init, axis = 1))[0][0] 
                cantera_idt_data = idt_data[index]
                
                # 简化机理指标 vs 真实机理指标的绘图
                # IDT part
                relative_error = np.mean(np.abs((cantera_idt_data - self.true_idt_data) / self.true_idt_data)) * 100
                self.InverseLogger.info(f"Relative Error is {relative_error} %")  
                # log scale
                true_idt_data = np.log10(self.true_idt_data); cantera_idt_data = np.log10(cantera_idt_data); 
                reduced_idt_data = np.log10(self.reduced_idt_data)
                
                self.InverseLogger.info(f"Average Diff between final_A and A0 is {np.mean(np.abs(A_init - self.reduced_mech_A0))}; While the min and max is {np.min(np.abs(A_init - self.A0))} and {np.max(np.abs(A_init - self.A0))}")
                self.InverseLogger.info("Compare First IDT:" + "\n" + f"True:{true_idt_data}; " + "\n" + f"Reduced:{reduced_idt_data};\n" + 
                                        f"Cantera:{cantera_idt_data};")
                self.InverseLogger.info("-" * 90)
                              

                compare_nn_train3(
                        true_idt_data,
                        cantera_idt_data,
                        reduced_idt_data,
                        labels = [r'$Optimal$', r'$Reduced$'],
                        markers = ['+', '+', ],
                        colors = ['blue', 'red',],
                        title = f'IDT  Relative Error: {relative_error:.2f} %',
                        save_path = inverse_path + '/compare_nn_IDT.png',
                        wc = self.IDT_condition
                    )
        
                # 保存 IDT 的相关数据
                np.savez(
                        inverse_path + "/IDT_data.npz",
                        true_idt_data = true_idt_data,
                        reduced_idt_data = reduced_idt_data,
                        cantera_idt_data = cantera_idt_data,
                        Alist = A_init
                        )

                self.InverseLogger.info(f'plot compare idt picture done! time cost:{time.time() - t1} seconds')
            except Exception:
                exstr = traceback.format_exc()
                self.InverseLogger.info(f'!!ERROR:{exstr}')