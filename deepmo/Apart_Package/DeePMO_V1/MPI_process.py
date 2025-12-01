from mpi4py import MPI
import sys, traceback, time
import numpy as np

from .DeePMO_IDT import GenOneDataIDT
from .DeePMO_LFS import GenOneDataLFS
from .DeePMO_PSRex import GenOneDataPSRex
from .DeePMO_IDT_PSR import GenOneDataIDTPSR, GenOneDataIDTPSR_plus_PSRex
from .DeePMO_IDT_HRR import GenOneDataIDT_HRR
from .DeePMO_IDT_PSRextinction import GenOneDataIDTPSRextinction
from .DeePMO_IDT_PSRex_HRR import GenOneDataIDT_HRR_PSRextinction
from .DeePMO_IDT_PSRex_LFS import GenOneDataIDT_LFS_PSRextinction
from .DeePMO_IDT_PSRex_LFS_concentration import GenOneDataIDT_LFS_PSRex_concentration
from .DeePMO_Exp_IDT_LFS import GenOneDataIDT_LFS
from .DeePMO_IDT_LFS_HRR import GenOneDataIDT_LFS_HRR
from .DeePMO_IDT_LFS_PSRconcentration import GenOneDataIDT_LFS_concentration

from utils.cantera_utils import *
from utils.setting_utils import *
from utils.yamlfiles_utils import * 

ct.suppress_thermo_warnings()

def GenIDT_MPI(IDT_condition: np.ndarray, samples:np.ndarray, eq_dict:dict, 
               reduced_mech:str, my_logger:Log, expect_sample_size:int, IDT_mode = 0,  fuel:str = None, oxidizer: str = None, 
               IDT_fuel:str = None, IDT_oxidizer: str = None, 
               idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, idt_cut_time_alpha = 1.5, save_path = './data/APART_data/tmp', **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT(index = index,
                            IDT_condition = IDT_condition,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            save_path = save_path,
                            cut_time = cut_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer, 
                            **kwargs)
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenLFS_MPI(LFS_condition: np.ndarray, samples:np.ndarray, eq_dict:dict, reduced_mech:str, my_logger:Log, expect_sample_size:int,
                fuel:str = None, oxidizer: str = None, LFS_fuel:str = None, LFS_oxidizer: str = None,
                save_path = './data/APART_data/tmp', **kwargs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataLFS(
                LFS_condition = LFS_condition,
                Alist = samples[index],
                eq_dict = eq_dict,
                fuel = fuel,
                oxidizer = oxidizer,
                index = index,
                reduced_mech = reduced_mech,
                my_logger = my_logger,
                save_path = save_path,
                LFS_fuel = LFS_fuel,
                LFS_oxidizer = LFS_oxidizer,
                **kwargs
                
            )
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenPSRex_MPI(PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, samples:np.ndarray, eq_dict:dict,
                 reduced_mech:str, my_logger:Log, expect_sample_size:int,  fuel:str = None, oxidizer: str = None,
                PSR_fuel:str = None, PSR_oxidizer: str = None, PSRex_decay_exp = 0.5,
                psr_error_tol = 50, save_path = './data/APART_data/tmp', **kwargs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataPSRex(
                PSR_condition = PSR_condition,
                RES_TIME_LIST = RES_TIME_LIST,
                Alist = samples[index],
                eq_dict = eq_dict,
                fuel = fuel,
                oxidizer = oxidizer,
                index = index,
                reduced_mech = reduced_mech,
                my_logger = my_logger,
                save_path = save_path,
                PSR_fuel = PSR_fuel,
                PSR_oxidizer=PSR_oxidizer,
                psr_error_tol = psr_error_tol,
                PSRex_decay_exp=PSRex_decay_exp,
                **kwargs
            )
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_PSR_MPI(IDT_condition: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, samples:np.ndarray, 
                           eq_dict:dict, PSR_fuel:str, PSR_oxidizer: str,
                           reduced_mech:str, my_logger:Log,  expect_sample_size:int ,
                            IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                             fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                            idt_cut_time_alpha = 1.5, psr_error_tol = 50, save_path = './data/APART_data/tmp', **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDTPSR(index = index,
                            IDT_condition = IDT_condition,
                            PSR_condition = PSR_condition,
                            RES_TIME_LIST = RES_TIME_LIST,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            psr_error_tol = psr_error_tol,
                            
                            save_path = save_path,
                            cut_time = cut_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            PSR_fuel = PSR_fuel,
                            PSR_oxidizer = PSR_oxidizer,
                            **kwargs)
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_HRR_MPI(IDT_condition: np.ndarray, samples:np.ndarray, eq_dict:dict, reduced_mech:str, my_logger:Log,  expect_sample_size:int ,
                    fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                   IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                   idt_cut_time_alpha = 1.5,  save_path = './data/APART_data/tmp', **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT_HRR(
                            index = index,
                            IDT_condition = IDT_condition,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            save_path = save_path,
                            cut_time = cut_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            **kwargs)
            if isinstance(res, int) or res is None:
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_PSRex_MPI(IDT_condition: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, samples:np.ndarray, 
                           eq_dict:dict, reduced_mech:str, my_logger:Log,  expect_sample_size:int ,
                            fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                            PSR_fuel:str = None, PSR_oxidizer: str = None,
                            IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                            idt_cut_time_alpha = 1.5, psr_error_tol = 50, save_path = './data/APART_data/tmp', PSRex_decay_exp = 0.5,
                            init_res_time = 1,  **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDTPSRextinction(
                            index = index,
                            IDT_condition = IDT_condition,
                            PSR_condition = PSR_condition,
                            RES_TIME_LIST = RES_TIME_LIST,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            psr_error_tol = psr_error_tol,
                            save_path = save_path,
                            cut_time = cut_time,
                            PSRex_decay_exp = PSRex_decay_exp,
                            init_res_time=init_res_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            PSR_fuel = PSR_fuel,
                            PSR_oxidizer = PSR_oxidizer,
                            **kwargs)
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_HRR_PSRex_MPI(IDT_condition: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray,  samples:np.ndarray, 
                           eq_dict:dict, reduced_mech:str, my_logger:Log,  expect_sample_size:int ,
                            fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                            PSR_fuel:str = None, PSR_oxidizer: str = None,
                            IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                            idt_cut_time_alpha = 1.5, psr_error_tol = 50, save_path = './data/APART_data/tmp', PSRex_decay_exp = 2,
                            init_res_time = 1, **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT_HRR_PSRextinction(
                            index = index,
                            IDT_condition = IDT_condition,
                            PSR_condition = PSR_condition,
                            RES_TIME_LIST = RES_TIME_LIST,  
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            psr_error_tol = psr_error_tol,
                            save_path = save_path,
                            cut_time = cut_time,
                            PSRex_decay_exp = PSRex_decay_exp,
                            init_res_time = init_res_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            PSR_fuel = PSR_fuel,
                            PSR_oxidizer = PSR_oxidizer,
                            **kwargs)
            if isinstance(res, int) or res is None:
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_PSRex_LFS_MPI(IDT_condition: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, LFS_condition: np.ndarray, samples:np.ndarray, 
                           eq_dict:dict,  reduced_mech:str, my_logger:Log,  expect_sample_size:int,
                            fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                            PSR_fuel:str = None, PSR_oxidizer: str = None, LFS_fuel:str = None, LFS_oxidizer: str = None,
                            IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                            idt_cut_time_alpha = 1.5, psr_error_tol = 50, save_path = './data/APART_data/tmp', PSRex_decay_exp = 0.5,
                            init_res_time = 1, **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT_LFS_PSRextinction(
                            index = index,
                            IDT_condition = IDT_condition,
                            PSR_condition = PSR_condition,
                            RES_TIME_LIST = RES_TIME_LIST,
                            LFS_condition = LFS_condition,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            psr_error_tol = psr_error_tol,
                            save_path = save_path,
                            cut_time = cut_time,
                            PSRex_decay_exp = PSRex_decay_exp,
                            init_res_time=init_res_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            PSR_fuel = PSR_fuel,
                            PSR_oxidizer = PSR_oxidizer,
                            LFS_fuel = LFS_fuel,
                            LFS_oxidizer = LFS_oxidizer,
                            **kwargs)
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_PSRex_LFS_PSRconcentration_MPI(IDT_condition: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, LFS_condition: np.ndarray, 
                                          PSR_concentration_kwargs: dict, samples:np.ndarray, eq_dict:dict,  reduced_mech:str, my_logger:Log,  expect_sample_size:int,
                                          fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                                          PSR_fuel:str = None, PSR_oxidizer: str = None, LFS_fuel:str = None, LFS_oxidizer: str = None,
                                          IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                                          idt_cut_time_alpha = 1.5, psr_error_tol = 50, save_path = './data/APART_data/tmp', PSRex_decay_exp = 0.5,
                                          init_res_time = 1, **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT_LFS_PSRex_concentration(
                            index = index,
                            IDT_condition = IDT_condition,
                            PSR_condition = PSR_condition,
                            RES_TIME_LIST = RES_TIME_LIST,
                            LFS_condition = LFS_condition,
                            PSR_concentration_kwargs= PSR_concentration_kwargs,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            psr_error_tol = psr_error_tol,
                            save_path = save_path,
                            cut_time = cut_time,
                            PSRex_decay_exp = PSRex_decay_exp,
                            init_res_time=init_res_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            PSR_fuel = PSR_fuel,
                            PSR_oxidizer = PSR_oxidizer,
                            LFS_fuel = LFS_fuel,
                            LFS_oxidizer = LFS_oxidizer,
                            **kwargs)
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_LFS_PSRconcentration_MPI(IDT_condition: np.ndarray, LFS_condition: np.ndarray, PSR_concentration_kwargs: dict, samples:np.ndarray, eq_dict:dict,  reduced_mech:str, my_logger:Log,  expect_sample_size:int,
                                          fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                                          LFS_fuel:str = None, LFS_oxidizer: str = None,IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                                          idt_cut_time_alpha = 1.5, save_path = './data/APART_data/tmp', **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT_LFS_concentration(
                            index = index,
                            IDT_condition = IDT_condition,
                            LFS_condition = LFS_condition,
                            PSR_concentration_kwargs= PSR_concentration_kwargs,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            save_path = save_path,
                            cut_time = cut_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            LFS_fuel = LFS_fuel,
                            LFS_oxidizer = LFS_oxidizer,
                            **kwargs)
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_PSR_plus_PSRex_MPI(IDT_condition: np.ndarray, PSR_condition: np.ndarray, RES_TIME_LIST: np.ndarray, samples:np.ndarray, 
                           eq_dict:dict,reduced_mech:str, my_logger:Log,  expect_sample_size:int,
                            fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                            PSR_fuel:str = None, PSR_oxidizer: str = None,
                            IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                            idt_cut_time_alpha = 1.5, psr_error_tol = 50, save_path = './data/APART_data/tmp', **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDTPSR_plus_PSRex(index = index,
                            IDT_condition = IDT_condition,
                            PSR_condition = PSR_condition,
                            RES_TIME_LIST = RES_TIME_LIST,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            psr_error_tol = psr_error_tol,
                            save_path = save_path,
                            cut_time = cut_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            PSR_fuel = PSR_fuel,
                            PSR_oxidizer = PSR_oxidizer,
                            **kwargs)
            if isinstance(res, int):
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_LFS_MPI(IDT_condition: np.ndarray, LFS_condition: np.ndarray, samples:np.ndarray, eq_dict:dict,
                       reduced_mech:str, my_logger:Log,  expect_sample_size:int ,
                            fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                            LFS_fuel:str = None, LFS_oxidizer: str = None,
                   IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                   idt_cut_time_alpha = 1.5,  save_path = './data/APART_data/tmp', **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT_LFS(
                            index = index,
                            IDT_condition = IDT_condition,
                            LFS_condition = LFS_condition,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            save_path = save_path,
                            cut_time = cut_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            LFS_fuel = LFS_fuel,
                            LFS_oxidizer = LFS_oxidizer,
                            **kwargs)
            if isinstance(res, int) or res is None:
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


def GenIDT_LFS_HRR_MPI(IDT_condition: np.ndarray, LFS_condition: np.ndarray, samples:np.ndarray, eq_dict:dict, 
                       reduced_mech:str, my_logger:Log,  expect_sample_size:int ,
                            fuel:str = None, oxidizer: str = None, IDT_fuel:str = None, IDT_oxidizer: str = None,
                            LFS_fuel:str = None, LFS_oxidizer: str = None,
                   IDT_mode = 0,  idt_arrays:np.ndarray = 1, cut_time:np.ndarray = None, 
                   idt_cut_time_alpha = 1.5,  save_path = './data/APART_data/tmp', **kwargs):
    """
    使用 MPI 生成数据集的函数；关于 MPI 的设置请参考 setup.yaml(实际上没有什么设置)
    参数完全等于 GenOneDataIDT，只是多了一个 sample 参数。
    return:
        error_num: 没有成功计算点火的样本数目
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 总工作量
    sample_length = 0 # 已经采样并计算完毕的样本数目
    njob = samples.shape[0]

    if rank==0:    # this is head worker, jobs are arranged by this worker
        job_all_idx = list(range(njob))  
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root = 0)

    # 参数列表
    job_content = []
    for i in range(sample_length, sample_length + int(njob)):
        job_content.append(i)
    
    # 每个进程分配到的工作序号，多余的工作平均分配给前几个worker
    q, r = njob // size, njob % size    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]
    error_nums = 0; result_nums = 0 # 错误数
    for index in work_content:
        try:
            res = GenOneDataIDT_LFS_HRR(
                            index = index,
                            IDT_condition = IDT_condition,
                            LFS_condition = LFS_condition,
                            Alist = samples[index],
                            eq_dict = eq_dict,
                            fuel = fuel,
                            oxidizer = oxidizer,
                            reduced_mech = reduced_mech,
                            my_logger = my_logger,
                            IDT_mode = IDT_mode,
                            idt_arrays = idt_arrays,
                            cut_time_alpha = idt_cut_time_alpha,
                            save_path = save_path,
                            cut_time = cut_time,
                            IDT_fuel = IDT_fuel,
                            IDT_oxidizer = IDT_oxidizer,
                            LFS_fuel = LFS_fuel,
                            LFS_oxidizer = LFS_oxidizer,
                            **kwargs)
            if isinstance(res, int) or res is None:
                error_nums += 1
            else:
                result_nums += 1
        except Exception:
            exstr = traceback.format_exc()
            my_logger.warning(f'!!ERROR:{exstr}')
            error_nums += 1
        finally:
            if result_nums > expect_sample_size:
                my_logger.info(f"Finished GenDataMPI; error nums: {error_nums}; result nums: {result_nums}")
                break
    return error_nums


"""=============================================================================================================================================================="""
"""                                                             调用函数                                                                                          """
"""=============================================================================================================================================================="""


def GenData_IDT_MPI(circ, args:dict, samples = None, boarder_samples = None, idt_arrays = None,):
    """
    MPI 多进程单独做 GenData_MPI; main.py 中函数的迁移
    """
    logger = Log(f'./log/APART_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_PSRex_MPI(circ, args:dict, samples = None, boarder_samples = None):
    """
    MPI 多进程单独做 GenData_MPI; main.py 中函数的迁移
    """
    logger = Log(f'./log/APART_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenPSRex_MPI(my_logger = logger, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenPSRex_MPI(my_logger = logger,  samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_LFS_MPI(circ, args:dict, samples = None, boarder_samples = None):
    """
    MPI 多进程单独做 GenData_MPI; main.py 中函数的迁移
    """
    logger = Log(f'./log/APART_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenLFS_MPI(my_logger = logger, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenLFS_MPI(my_logger = logger,  samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_IDT_PSR_MPI(circ, args:dict, samples = None, boarder_samples = None, idt_arrays = None,):
    """
    MPI 多进程单独做 GenData_MPI; main.py 中函数的迁移
    """
    logger = Log(f'./log/APART_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_PSR_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_PSR_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_IDT_HRR_MPI(circ, args:dict, samples = None, boarder_samples = None,idt_arrays = None,):
    """
    MPI 多进程单独做 DeePMO_IDT_PSRex_HRR
    """
    logger = Log(f'./log/DeePMO_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_HRR_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_HRR_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_IDT_PSRex_MPI(circ, args:dict, samples = None, boarder_samples = None,idt_arrays = None,):
    """
    MPI 多进程单独做 GenData_MPI; main.py 中函数的迁移
    """
    logger = Log(f'./log/APART_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; Remove Chem:{MPIargs.get('remove_chem', None)} expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_PSRex_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_PSRex_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_IDT_PSRex_HRR_MPI(circ, args:dict, samples = None, boarder_samples = None,idt_arrays = None,):
    """
    MPI 多进程单独做 DeePMO_IDT_PSRex_HRR
    """
    logger = Log(f'./log/DeePMO_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_HRR_PSRex_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_HRR_PSRex_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_IDT_PSRex_LFS_MPI(circ, args:dict, samples = None, boarder_samples = None,idt_arrays = None,):
    """
    MPI 多进程单独做 DeePMO_IDT_PSRex_LFS
    """
    logger = Log(f'./log/DeePMO_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}; IDT mode: {MPIargs['IDT_mode']}")
    error_nums = GenIDT_PSRex_LFS_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_PSRex_LFS_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)


def GenData_IDT_PSRex_LFS_PSRconcentration_MPI(circ, args:dict, samples = None, boarder_samples = None, idt_arrays = None,):
    """
    MPI 多进程单独做 DeePMO_IDT_PSRex_LFS
    """
    logger = Log(f'./log/DeePMO_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_PSRex_LFS_PSRconcentration_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_PSRex_LFS_PSRconcentration_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, save_path = "./data/APART_data/boarder_tmp", **MPIargs)


def GenData_IDT_LFS_PSRconcentration_MPI(circ, args:dict, samples = None, boarder_samples = None, idt_arrays = None,):
    """
    MPI 多进程单独做 DeePMO_IDT_LFS
    """
    logger = Log(f'./log/DeePMO_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_LFS_PSRconcentration_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_LFS_PSRconcentration_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    


def GenData_IDT_PSR_plus_PSRex_MPI(circ, args:dict, samples = None, boarder_samples = None,idt_arrays = None,):
    """
    MPI 多进程单独做 GenData_MPI; main.py 中函数的迁移
    """
    logger = Log(f'./log/APART_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']};")
    error_nums = GenIDT_PSR_plus_PSRex_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_PSR_plus_PSRex_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_IDT_LFS_MPI(circ, args:dict, samples = None, boarder_samples = None,idt_arrays = None,):
    """
    MPI 多进程单独做 DeePMO_IDT_PSRex_HRR
    """
    logger = Log(f'./log/DeePMO_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_LFS_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_LFS_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")


def GenData_IDT_LFS_HRR_MPI(circ, args:dict, samples = None, boarder_samples = None,idt_arrays = None,):
    """
    MPI 多进程单独做 DeePMO_IDT_PSRex_HRR
    """
    logger = Log(f'./log/DeePMO_GenData_circ={circ}.log')
    mkdirplus('./data/APART_data/tmp')
    # 计算真实机理的IDT，并加载 APART_args + chem_info 参数
    sample_size = args['sample_size'][circ] if isinstance(args['sample_size'], Iterable) else args['sample_size']
    MPIargs = args.copy()
    eq_dict = read_json_data(f"./data/APART_data/reduced_data/eq_dict_circ={circ}.json")
    MPIargs.update(eq_dict)
    MPIargs.update({
        'expect_sample_size': sample_size,
    })

    if idt_arrays is None:
        reduced_idt_data = np.load('./data/true_data/true_idt.npz')['reduced_idt_data']
        true_idt_data = np.load('./data/true_data/true_idt.npz')['true_idt_data']
        idt_arrays = np.maximum(reduced_idt_data, true_idt_data)
    if samples is None:
        samples = np.load(f'./data/APART_data/Asamples_{circ}.npy')

    t1 = time.time(); 
    # 使用多进程生成数据
    logger.info(f"Start GenDataMPI; expect sample size: {MPIargs['expect_sample_size']}; " + 
                f"IDT_mode: {MPIargs['IDT_mode']}; idt_cut_time_alpha = {MPIargs.get('idt_cut_time_alpha', 10)}")
    error_nums = GenIDT_LFS_HRR_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = samples, **MPIargs)
    # 继续生成 boarder_samples 的数据
    if boarder_samples is None:
        boarder_samples = f'./data/APART_data/Aboarder_samples_{circ}.npy'
        if os.path.exists(boarder_samples):
            boarder_samples = np.load(boarder_samples)
            logger.info("Start the Boarder Sample Generation")
            error_nums2 = GenIDT_LFS_HRR_MPI(my_logger = logger, idt_arrays = idt_arrays, samples = boarder_samples, 
                                            save_path = "./data/APART_data/boarder_tmp", **MPIargs)
    logger.info(f"Finished GenDataMPI costs {time.time() - t1}s; error nums: {error_nums}")
