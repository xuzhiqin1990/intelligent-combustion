import sys
import numpy as np
import os
import time
try:
    from mpi4py import MPI
except:
    pass
import shutil
from dmgr.utils import *
from func_timeout import func_set_timeout, FunctionTimedOut

import scipy.sparse as sp
from scipy.sparse import csr_matrix

from multiprocessing import Pool


def one_core_task(
        worker_id: int, 
        vectors, 
        this_worker_job: list, 
        settings, 
        get_data, 
        working_dir: str, 
        count: int = None,
        log_path: str = None,
        save_path: str = None,
        save_result: bool = True,
        return_result: bool = False,
        ):
    '''
    单个cpu核需要完成的任务，接收一些简化机理向量，计算其indicator
    Parameters:
        index: cpu index
        vectors: reduced mechanisms one-hot vector
    Returns:
        None
    '''
    indicator_list = settings.indicators
    Datas = {}
    Datas['vector'] = []
    for indicator in indicator_list:
        Datas[indicator] = []
    
    # 创建日志文件
    if log_path is None:
        log_path = f'{working_dir}/log/gendata/iter_{count}.log'
    my_logger = Log(log_path)
    # my_logger.info(f'worker id: {worker_id}, worker jobs: {this_worker_job}')

    t0 = time.time()
    for i in range(np.size(vectors, 0)):
        vector = vectors[i]

        # 如果vector为稀疏矩阵，转化为稠密矩阵
        sparse_form = False
        if type(vector) == csr_matrix:
            vector = vector.toarray()[0]
            sparse_form = True

        job_id = this_worker_job[i]
        try:
            # 生成数据
            data = get_data(
                chem = vector, 
                settings = settings,
                mode = 'parallel', 
                job_id = job_id, 
                worker_id = worker_id,
                log_path = log_path,)
            
            # 保存数据，如果原来的数据类型为稀疏矩阵，转化为稀疏矩阵再保存
            if sparse_form:
                # 转化为稀疏矩阵
                vector_sp = sp.coo_matrix(vector)

                # 保存稀疏矩阵的数据
                Datas['vector_row'] = vector_sp.row
                Datas['vector_col'] = vector_sp.col
                Datas['vector_data'] = vector_sp.data
                Datas['vector_shape'] = vector_sp.shape
            else:
                Datas['vector'].append(data['vector'])
            
            for indicator in indicator_list:
                Datas[indicator].append(data[indicator])

        except FunctionTimedOut:
            my_logger.info('time out!')
        except Exception as e:
            my_logger.info('error')
            # if i == 0:
            #     my_logger.info(e)
                # print(e)

    # 如果没有tmp文件夹，则创建
    if save_path is None:
        tmp_path = f'{working_dir}/data/simulation_data/tmp'
    else:
        tmp_path = f'{save_path}/tmp'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path, exist_ok=True)

    # 保存数据
    if save_result:
        np.savez(f'{tmp_path}/{worker_id}.npz', **Datas)
    
    my_logger.info(f'finish jobs in worker {worker_id}, time cost: {time.time()-t0:.2f} s')

    if return_result:
        return Datas


def get_job_index(njob, rank, worker_num, batch_id=0, batch_num=1):
    '''
    每个进程分配到的工作序号，多余的工作平均分配给前几个worker。
    实验中为避免报错导致整个程序崩溃，故分batch_num次提交mpi任务，每次提交的任务数量为njob//batch_num
    Parameters:
        njob: total job num
        vectors: worker index
        worker_num: total worker num
        batch_id: 当前提交任务的id
        batch_num: 总共提交任务的数量
    Returns:
        this_worker_job: this worker job
    '''
    # 本次任务中需要计算的任务编号
    njob_this_batch = njob // batch_num
    job_all_idx = list(range(batch_id * njob_this_batch, (batch_id + 1) * njob_this_batch))

    q, r = njob_this_batch // worker_num, njob_this_batch % worker_num    # 商和余数
    this_worker_job = [job_all_idx[x] for x in range(rank * q, (rank + 1) * q)]
    if rank < r:
        this_worker_job.append(job_all_idx[-(rank + 1)])
    return this_worker_job



# 使用 multiprocessing 生成数据
def gendata_multiprocessing( 
        settings, 
        get_data, 
        count: int = None,
        log_path: str = None,
        save_path: str = None
        ):
    working_dir = settings.working_dir
    cpu_num = settings.cpu_num

    vector_path = f'{working_dir}/data/vector_data/vector_{count}.npz'
    vector = np.load(vector_path)['vector']

    p = Pool(cpu_num)
    for index in range(cpu_num):
        this_worker_job = get_job_index(njob=np.size(vector, 0), rank=index, worker_num=cpu_num)
        print('worker index:', index, 'jobs in this worker', len(this_worker_job))

        p.apply_async(
            func=one_core_task, 
            args=(index, vector[this_worker_job], this_worker_job, settings,
                get_data, working_dir, count, log_path, save_path))
    p.close()
    p.join()


def gather_data(
        settings, 
        count: int = None, 
        sparse_form: bool = False, 
        save_path: str = None,
        save_name: str = 'simulation.npz',
        del_tmp_path: bool = True,
        return_datas: bool = False,
        ):
    working_dir = settings.working_dir

    # 创建保存数据的字典，确定数据的格式
    Datas = {}
    if count is not None:
        Datas['count'] = count
    if sparse_form:
        Datas['vector_row'] = []
        Datas['vector_col'] = []
        Datas['vector_data'] = []
        Datas['vector_shape'] = []
        vector_size = 0
    else:
        Datas['vector'] = []
    for indicator in settings.indicators:
        Datas[indicator] = []

    if save_path is None:
        tmp_data_save_path = f'{working_dir}/data/simulation_data/tmp'
    else:
        tmp_data_save_path = f'{save_path}/tmp'
        
    # 读取并保存数据
    for files in os.listdir(tmp_data_save_path):
        try:
            datas = np.load(
                f'{tmp_data_save_path}/{files}', allow_pickle=True)
            
            if sparse_form:
                Datas['vector_row'].extend(datas['vector_row'] + vector_size)
                Datas['vector_col'].extend(datas['vector_col'])
                Datas['vector_data'].extend(datas['vector_data'])
                vector_size += datas['vector_shape'][0]
            else:
                Datas['vector'].extend(datas['vector'])
            
            for indicator in settings.indicators:
                Datas[indicator].extend(datas[indicator])
        except Exception as r:
            print(files)

    if sparse_form:
        Datas['vector_shape'] = np.array([vector_size, datas['vector_shape'][1]])

    # 保存数据
    if save_path is None:
        np.savez(f'{working_dir}/data/simulation_data/simulation_{count}.npz', **Datas)
    else:
        np.savez(f'{save_path}/{save_name}', **Datas)

    # 删除中间文件
    if del_tmp_path:
        shutil.rmtree(tmp_data_save_path, ignore_errors=True) 

    if return_datas:
        return Datas



def gendata_MPI(
        settings,
        get_data, 
        count: int = None,
        vector_path: str = None,
        log_path: str = None,
        save_path: str = None,
        save_name: str = 'simulation.npz',
        sparse_form: bool = False,
        need_gather: bool = False,
        batch_id: int = 0,
        batch_num: int = 1,
        ):
    r'''
    使用MPI生成数据并自动保存。
    第一个worker负责加载01向量，然后将任务传播至其他worker。最后一个worker负责收集并保存数据。
    Args:
        ``global_settings``: 全局设置
        ``get_data``: 生成数据的函数
        ``count``: 当前的迭代次数
        ``log_path``: 生成数据日志文件的保存路径，没有则默认保存在 working_dir/log/gendata/iter_{count}.log
        ``save_path``: 生成数据的保存路径，没有则默认保存在 working_dir/data/simulation_data/simulation_{count}.npz
        ``save_name``: 生成数据的保存文件名，保存至 save_path/save_name，save_path=None时无效
        ``sparse_form``: 是否以稀疏矩阵的形式保存数据
        ``need_gather``: 是否需要收集数据
        ``batch_id``: 当前批次的id
        ``batch_num``: 总共的批次数。实验中为避免报错导致整个程序崩溃，故分batch_num次提交mpi任务，每次提交的任务数量为njob//batch_num
    '''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    working_dir = settings.working_dir

    if rank == 0:
        # 创建日志文件，如果已存在则清空内容
        if log_path == None:
            log_path = f'{working_dir}/log/gendata/iter_{count}.log'
        if batch_id == 0:
            my_logger = Log(log_path, mode = 'w')
        else:
            time.sleep(5)

    # 其他进程等待第一个进程广播log_path，这实际上创建了一个进程锁
    # 防止其他进程在第一个进程清空日志文件之前写入日志
    log_path = comm.bcast(log_path, root=0) 
    my_logger = Log(log_path)
    
    # 加载数据
    if vector_path is None:
        vector_path = f'{working_dir}/data/vector_data/vector_{count}.npz'
    
    if sparse_form:
        vector = load_vector_sp(vector_path, return_sparse = True)
    else:
        vector = np.load(vector_path)['vector']

    # 如果在程序中直接gather，则不需要保存临时数据
    if need_gather:
        save_result = False
    else:
        save_result = True

    vector_size = np.size(vector, 0)
    this_worker_job = get_job_index(njob = vector_size, rank = rank, worker_num = size, 
                                    batch_id = batch_id, batch_num = batch_num)

    worker_id = batch_id * size + rank

    # 太大就不输出了，占用资源
    if vector_size < 100000:
        my_logger.info(f'worker id: {worker_id}, worker jobs: {this_worker_job}')
    vector = vector[this_worker_job]
    datas = one_core_task(worker_id, vector, this_worker_job, settings,
                        get_data, working_dir, count, 
                        log_path=log_path, save_path=save_path,
                        save_result=save_result, return_result=True)
    
    if need_gather:
        gathered_datas = comm.gather(datas, root=0)
    
    if rank == 0 and need_gather:
        my_logger.info(f'recieving data start...')
        # 接收并合并其他进程的运行结果
        Datas = {}
        
        if count is not None:
            Datas['count'] = count

        if sparse_form:
            Datas['vector_row'] = []
            Datas['vector_col'] = []
            Datas['vector_data'] = []
            Datas['vector_shape'] = []
            vector_size = 0
        else:
            Datas['vector'] = []
        for indicator in settings.indicators:
            Datas[indicator] = []

        # 接收并合并其他进程的运行结果
        for worker_i in range(size):
            # 接收数据
            datas = gathered_datas[worker_i]
            # 合并结果
            if sparse_form:
                Datas['vector_row'].extend(datas['vector_row'] + vector_size)
                Datas['vector_col'].extend(datas['vector_col'])
                Datas['vector_data'].extend(datas['vector_data'])
                vector_size += datas['vector_shape'][0]
            else:
                Datas['vector'].extend(datas['vector'])
            
            for indicator in settings.indicators:
                Datas[indicator].extend(datas[indicator])

        if sparse_form:
            Datas['vector_shape'] = np.array([vector_size, datas['vector_shape'][1]])

        # 保存数据
        if save_path is None:
            np.savez(f'{working_dir}/data/simulation_data/simulation_{count}.npz', **Datas)
        else:
            np.savez(f'{save_path}/{save_name}', **Datas)
        
        my_logger.info(f'recieve data finish')


