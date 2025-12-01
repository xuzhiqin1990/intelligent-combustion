# -*- coding:utf-8 -*-
import os, yaml, json, re, logging, time
import cantera as ct
import numpy as np
import torch.nn as nn

from .cantera_utils import solve_flame_speed, solve_idt_fast, solve_psr

'''============================================================================================================='''
'''                                         DeePMR 0d base class                                                '''
'''============================================================================================================='''

class DeePMR_base_class():
    def __init__(self, mechanism = './settings/chem.yaml'):
        self.mechanism = mechanism
        ct.suppress_thermo_warnings()


    # 获取机理信息
    def mechanism_info(self):
        # self.gas = ct.Solution(self.mechanism)
        all_species = ct.Species.listFromFile(self.mechanism)
        ref_phase = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species)
        all_reactions = ct.Reaction.listFromFile(self.mechanism, ref_phase)
        self.gas = ct.Solution(thermo='ideal-gas', kinetics='gas', species=all_species, reactions = all_reactions)
        self.all_species   = ct.Species.listFromFile(self.mechanism)
        self.all_reactions = ct.Reaction.listFromFile(self.mechanism, self.gas)
        self.species_num   = len(self.all_species)      # 组分个数
        self.reactions_num = len(self.all_reactions)    # 反应个数

    
    # 指定燃料和氧化剂
    def set_FO(self, fuel, oxidizer):
        self.fuel, self.oxidizer = fuel, oxidizer
        # self.fo_index = [self.gas.species_index(self.fuel), self.gas.species_index('O2'), self.gas.species_index('N2')]

    # 设置初始温度压强当量比采样点
    def set_TPphi(self, ini_T, ini_P, ini_phi):
        self.ini_T = list(ini_T); self.ini_P = list(ini_P); self.ini_phi = list(ini_phi)
        self.num_T = len(ini_T);  self.num_P = len(ini_P);  self.num_phi = len(ini_phi)
    
    # 设置PSR条件与Restime
    def set_psr_condition(self, condition, res_time):
        self.psr_condition = condition; self.RES_TIME_LIST = res_time
    

    # ================================================
    #            数据生成模块，生成0d火焰数据
    # ================================================
    # 生成真实或者简化点火数据 + psr 数据
    def GenTrueData(self, mech:str, mode:str = 'true', **kwargs):
        """
        生成所有的真实或者简化机理数据，包括以下2种：IDT， PSR
        params:
            mode: {'true','reduced','zdy'} 接受这两种的输入，分别表示真实点火和简化点火 和自定义状态
            mech: 机理文件路径; 没有默认值，一定要指定机理
            kwargs:
                kwargs 中包含所有 solve idt fast 的参数；详见 solve idt fast
                kwargs 中包含所有 solve flame 的参数，详见 solve flame
                
                save_path: 如果不为 None, 则会将数据保存到这个路径
        return:
            IDT, T, PSR_T
        """
        IDT, Temperature, PSR_T = [], [], []
        t0 = time.time()
        gas = ct.Solution(mech)
        # 计算 IDT
        for k in self.ini_phi:
            for i in self.ini_T:
                for j in self.ini_P:
                    gas.TP = i, j * ct.one_atm
                    gas.set_equivalence_ratio(k, self.fuel, self.oxidizer)
                    idt, T = solve_idt_fast(gas, **kwargs) # 求点火延迟时间和最终火焰温度
                    IDT.append(idt); Temperature.append(T)
        # 计算 PSR         
        for i in range(len(self.psr_condition)):
            phi, T, P = self.psr_condition[i][0], self.psr_condition[i][1], self.psr_condition[i][2]
            psr_T = solve_psr(gas, self.RES_TIME_LIST[i],  T, P, phi, self.fuel, self.oxidizer)
            PSR_T.extend(psr_T)
        if mode == 'true':
            np.savez(kwargs.get('save_path', './data/true_data.npz'), IDT = IDT, T = Temperature, PSR_T = PSR_T)
            print('生成真实数据耗时:{} s'.format(time.time() - t0))
            return IDT, T, PSR_T
        if mode == 'reduced':
            np.savez(kwargs.get('save_path', './data/reduced_data.npz'), IDT = IDT, T = Temperature, PSR_T = PSR_T)
            print('生成简化机理数据耗时:{} s'.format(time.time() - t0))
            return IDT, T, PSR_T
        if mode == 'zdy':
            np.savez(kwargs.get('save_path', './true_data.npz'), IDT = IDT, T = Temperature, PSR_T = PSR_T)
            return IDT, PSR_T

        
    # 加载真实点火数据
    def load_true_data(self):
        if not os.path.exists(self.true_data_path + '/true_idt_data.npz'):
            self.GenTrueData(mode = 'true', mech = self.detailed_mech)
        Data = np.load('%s/true_idt_data.npz' % (self.true_data_path))
        self.true_idt_data, self.true_T_data = Data['IDT'], Data['T']

        Data = np.load('%s/true_ref_psr_data.npz' % (self.true_data_path), allow_pickle = True)
        self.psr_condition = Data['condition']
        self.RES_TIME_LIST = Data['RES_TIME_LIST']
        self.true_psr_data = Data['T_LIST']

    # ================================================
    #                    make file
    # ================================================
    def generate_dir(self):
        # 公共参数文件夹
        if not os.path.exists('./data'):
            os.mkdir('./data')
            os.mkdir('./data/dnn_data')
            os.mkdir('./data/ignition_data')
            os.mkdir('./data/true_data')
            os.mkdir('./data/gathered_dnn_data')
            os.mkdir('./data/vector_data')
            
        # 生成存放网络数据的文件夹
        if not os.path.exists('./model'):
            os.mkdir('./model')
        if not os.path.exists('./model/model_idt_iteration'):
            os.mkdir('./model/model_idt_iteration')
            os.mkdir('./model/model_idt_iteration/model_pth')
            os.mkdir('./model/model_idt_iteration/loss_his')
        if not os.path.exists('./model/model_psr_iteration'):
            os.mkdir('./model/model_psr_iteration')
            os.mkdir('./model/model_psr_iteration/model_pth')
            os.mkdir('./model/model_psr_iteration/loss_his')

        if not os.path.exists('./log'):
            os.mkdir('./log')

    def load_path(self):
        self.data_path              = './data'
        self.dnn_data_path          = './data/dnn_data'
        self.ignition_data_path     = './data/ignition_data'
        self.true_data_path         = './data/true_data'
        self.gathered_dnn_data_path = './data/gathered_dnn_data'
        self.vector_data_path       = './data/vector_data'

        self.model_idt_path      = './model/model_idt_iteration'
        self.model_idt_pth_path  = './model/model_idt_iteration/model_pth'
        self.model_idt_loss_path = './model/model_idt_iteration/loss_his'

        self.model_psr_path      = './model/model_psr_iteration'
        self.model_psr_pth_path  = './model/model_psr_iteration/model_pth'
        self.model_psr_loss_path = './model/model_psr_iteration/loss_his'

        self.log_path = './log'
