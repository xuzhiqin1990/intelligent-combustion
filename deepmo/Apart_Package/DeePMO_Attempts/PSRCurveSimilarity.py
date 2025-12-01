# -*- coding:utf-8 -*-
import os, time, traceback, subprocess
import numpy as np, pandas as pd, seaborn as sns, cantera as ct
from Apart_Package.utils.cantera_PSR_definations import *
from Apart_Package.utils.cantera_utils import *
from Apart_Package.utils.setting_utils import *
from Apart_Package.utils.yamlfiles_utils import *
from concurrent.futures import ProcessPoolExecutor

class PSRCurveSimilarityMeasure(object):
    def __init__(self, setup_file:str, PSR_condition_file: str = None) -> None:
        """
        在 init 中进行以下事项：
        1. 读取 yaml 文件，获取 PSR 曲线的计算参数: PSR_condition 和 RES_TIME_LIST
        2. 计算详细机理的 PSR 曲线数据
        """
        self.setup_file = setup_file
        self.APART_args = get_yaml_data(setup_file)
        self.PSR_condition = np.array([
            [phi, T, P] for phi in self.APART_args['PSR_phi'] for T in self.APART_args['PSR_T'] for P in self.APART_args['PSR_P']
        ])
        self.PSR_decay_exp = self.APART_args['PSR_decay_exp']
        self.detailed_mechanism = self.APART_args['detailed_mechanism']
        self.reduced_mechanism = self.APART_args['reduced_mechanism']
        self.logger = Log("./log/PSRCurveSimilarity.log")

        if PSR_condition_file is not None and os.path.exists(PSR_condition_file):
            data = np.load(PSR_condition_file)
            self.PSR_condition = data['PSR_condition']
            self.RES_TIME_LIST = data['RES_TIME_LIST']
            self.true_psr_data = data['true_psr_data']
            self.reduced_psr_data = data['reduced_psr_data']
            self.true_psr_extinction_time = data['true_psr_extinction_time']
        else:
            mkdirplus(os.path.dirname(PSR_condition_file))
            # 计算 detailed mechanism 的 PSR 熄火时间
            self.true_psr_extinction_time = yaml2psr_extinction_time(
                self.detailed_mechanism, setup_file = setup_file,
                exp_factor = 2 ** self.PSR_decay_exp,
            )
            # 生成 RES_TIME_LIST
            RES_TIME_LIST_nums = self.APART_args['RES_TIME_LIST_nums']
            # RES_TIME_LIST_space = self.APART_args['RES_TIME_LIST_space']
            self.RES_TIME_LIST = np.array([
                2 ** np.linspace(1, np.ceil(np.log2(tmp_psrex)), RES_TIME_LIST_nums) for tmp_psrex in self.true_psr_extinction_time
                ]
            )
            self.true_psr_data = yaml2psr(
                self.detailed_mechanism, setup_file = setup_file,
                PSR_condition = self.PSR_condition,
                RES_TIME_LIST = self.RES_TIME_LIST,
            )
            self.reduced_psr_data = yaml2psr(
                self.reduced_mechanism, setup_file = setup_file,
                PSR_condition = self.PSR_condition,
                RES_TIME_LIST = self.RES_TIME_LIST,
            )
            np.savez(
                PSR_condition_file,
                PSR_condition = self.PSR_condition,
                RES_TIME_LIST = self.RES_TIME_LIST,
                true_psr_data = self.true_psr_data,
                reduced_psr_data = self.reduced_psr_data,
                true_psr_extinction_time = self.true_psr_extinction_time,
            )
        self.A0, self.eq_dict = yaml_key2A(self.reduced_mechanism)
        self.samples = []
    
    def ASample(self, l_alpha, r_alpha, circ = 0):
        """
        基本的针对 Alist 的采样; 采用均匀采样的方法
        """
        self.samples = sample_constant_A(
            size = self.APART_args['sample_size'],
            A0 = self.A0,
            l_alpha = l_alpha,
            r_alpha = r_alpha,
            n_latin = True, 
            save_path = f"./data/Asample_{circ}.npy"
        )
    
    def GenDataFuture(self, samples:np.ndarray = None, start_sample_index = 0, cpu_process = None, ignore_error_path = None, **kwargs):
        """
        使用 future 模块的 ProcessPoolExecutor 生成数据; 
        params:
            samples: np.ndarray, 用于生成数据的初始点集
            idt_cut_time_alpha: float, 计算 IDT 的 cut time 阈值系数
            start_sample_index: int, 从 samples 的第几个点开始生成数据
            cpu_process: int, 使用的 cpu 核心数
            ignore_error_path: str, 保存 ignore_error 的路径
            save_path: str, 保存数据的路径
        """
        cpu_process = os.cpu_count() - 1 if cpu_process is None else cpu_process
        expected_sample_size = self.APART_args['sample_size'] if isinstance(self.APART_args['sample_size'], int) else self.APART_args['sample_size'][self.circ]
        samples = self.samples if samples is None else samples
        true_psr_data = self.true_psr_data.reshape(self.RES_TIME_LIST.shape)
        RES = []
        def callback(status):
            status = status.result()
            if not status is None:
                RES.append(status)
        mkdirplus("./data/tmp")
        sample_length = np.size(samples, 0)
        self.logger.info(f"GenDataFuture: sample_length: {sample_length}; sample_shape {np.shape(samples)}")
        with ProcessPoolExecutor(max_workers = cpu_process) as exec:
            for index in range(sample_length):
                try:
                    future = exec.submit(
                            GenOnePSRCurveData, 
                            index = index + start_sample_index,
                            reduced_chem = self.reduced_mechanism,
                            Alist = samples[index],
                            eq_dict = self.eq_dict,
                            true_psr_data = true_psr_data,
                            PSR_condition = self.PSR_condition,
                            RES_TIME_LIST = self.RES_TIME_LIST,
                            fuel = self.APART_args['fuel'],
                            oxidizer = self.APART_args['oxidizer'],
                            logger = self.logger,
                            **kwargs
                            ).add_done_callback(callback)
                except Exception as r:
                    self.logger.info(f'Multiprocess error; error reason:{r}')
        if not ignore_error_path is None:
            np.save(ignore_error_path, np.array(RES))  
        return len(RES)

    

def GenOnePSRCurveData(index, reduced_chem, Alist, eq_dict, true_psr_data, PSR_condition, RES_TIME_LIST, fuel, oxidizer, logger):
    """
    """
    save_path = f"./data/tmp/{index}th.npz"
    chem_file = Adict2yaml(reduced_chem, f"./data/tmp/{index}th.yaml", eq_dict, Alist)
    try:
        t0 = time.time()
        psr_curve, PSR_T = yaml2PSRCurveSimilarityMeasure(
            chem_file = chem_file,
            measurebased_data = true_psr_data,
            PSR_condition = PSR_condition,
            RES_TIME_LIST = RES_TIME_LIST,
            fuel = fuel,
            oxidizer = oxidizer,
            psr_tol = 10,
        )
        np.savez(save_path, psr_curve = psr_curve, Alist = Alist, psr_T = PSR_T)
        logger.info(f"GenOnePSRCurveData: {index}th finished, time cost: {time.time() - t0:.2f}s")
        return 1
    except Exception as e:
        logger.info(f"Error in GenOnePSRCurveData: {e}")
        return None
    
    
            