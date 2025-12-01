import os, sys
sys.path.append('..')
import numpy as np
from .vector_generator import delete_combination
from func_timeout import func_set_timeout, FunctionTimedOut
import time
import shutil
try:
    from mpi4py import MPI
except:
    pass
from dmgr.utils import output_loss, compare_loss, Log
from dmgr.data import get_job_index



def get_best_vector_info(settings, best_data, get_species_name, get_reduced_sp_reac):
    '''
    在log中记录本次实验获得的机理的信息
    Args:
        settings: 配置文件
        best_data: 本次实验获得的最优机理
        get_species_name: 从settings.chemfile_path文件中获取组分名称的函数
        get_reduced_sp_reac: 从best_vector和settings.chemfile_path文件中获取组分与反应的函数
    '''
    working_dir = settings.working_dir

    # 计算误差
    true_data = np.load(f'{working_dir}/data/true_data.npz')
    loss, loss_list = output_loss(settings, best_data, true_data, return_loss_list = True)

    # 加载原始机理
    Data = np.load(os.path.join(f'{working_dir}/ini_vector.npz'))
    ini_vector = Data['vector']

    # 加载上一轮的机理
    try:
        Data = np.load(f'{working_dir}/best_vector.npz')
        original_vector = Data['vector']
    except:
        original_vector = np.array(ini_vector)

    # 记录保留与删除的组分
    species_name = get_species_name(settings.chemfile_path)

    best_vector = best_data['vector']
    # 与上一次相比删除的组分
    deleted_species_index1 = np.nonzero(original_vector - best_vector)[0]
    deleted_species1 = []
    for i in deleted_species_index1:
        deleted_species1.append(species_name[i])

    # 从初始时刻开始总共删除的组分
    deleted_species_index2 = np.nonzero(ini_vector - best_vector)[0]
    deleted_species2 = []
    for i in deleted_species_index2:
        deleted_species2.append(species_name[i])

    # 保留的组分
    sp_num = np.sum(best_vector)
    species, reactions = get_reduced_sp_reac(best_vector, settings.chemfile_path)
    saved_species = [s.name for s in species]
    sp_num, reac_num = len(species), len(reactions)

    my_logger = Log(f'{working_dir}/log/dsc_log.log')
    my_logger.info('-'*100)
    my_logger.info(f'species num: {sp_num}, reaction num: {reac_num}')
    my_logger.info('weights: %s' % settings.dsc_weight)
    my_logger.info('delete species max: %s' % settings.del_per_iteration_max)
    my_logger.info('best loss: {:.2f}'.format(loss))
    # 对每个指标输出误差
    for i, indicator in enumerate(settings.indicators):
        config = getattr(settings, f'{indicator}_config')
        if config.loss_form == 'relative':
            my_logger.info(f'{indicator} loss: {100*loss_list[i]:.2f} %, loss form: {config.loss_form}')
        elif config.loss_form == 'absolute':
            my_logger.info(f'{indicator} loss: {loss_list[i]:.2f}, loss form: {config.loss_form}')
    my_logger.info('save path: %sspecies.npz' % int(np.sum(best_vector)))
    my_logger.info('deleted species in this loop: %s' % deleted_species1)
    my_logger.info('deleted species in all loop: %s' % deleted_species2)
    my_logger.info('saved species: %s' % saved_species)



def one_core_task(
        worker_id: int, 
        vectors, 
        this_worker_job: list,
        settings,
        get_data,
        ):
    '''
    单个cpu核需要完成的任务，接收一些简化机理向量，计算其indicator
    Parameters:
        index: cpu index
        vectors: reduced mechanisms one-hot vector
    Returns:
        None
    '''
    working_dir = settings.working_dir
    Datas = []

    # 创建日志文件
    log_path = f'{working_dir}/log/dsc_GenData.log'
    my_logger = Log(log_path)

    t0 = time.time()
    for i in range(np.size(vectors, 0)):
        vector = vectors[i]

        job_id = this_worker_job[i]
        try:
            # 生成数据
            datas = get_data(
                chem = vector, 
                settings = settings,
                mode = 'parallel', 
                job_id = job_id, 
                worker_id = worker_id,
                log_path = log_path,)
            
            Datas.append(datas)

        except FunctionTimedOut:
            my_logger.info('time out!')
        except Exception as e:
            my_logger.info('error!')
            # my_logger.info(e)
    
    # 计算误差最小的机理
    if len(Datas) > 0:
        best_datas = Datas[0]
        for i in range(len(Datas)):
            best_datas = compare_loss(settings, best_datas, Datas[i])
        
        # 保存最优机理
        np.savez(f'{working_dir}/data/dsc_tmp/{worker_id}.npz', **best_datas)
    else:
        best_datas = None


    my_logger.info(f'finish jobs in worker {worker_id}, time cost: {time.time()-t0:.2f} s')

    return best_datas



def update_reduced_mechanism_mpi(settings, get_data, get_species_name, get_reduced_sp_reac, gather_data = False):
    r'''
    使用MPI生成数据，计算并自动保存和更新简化机理。
    worker_0负责生成01向量，然后将任务传播至其他worker。最后worker_0负责收集并保存数据
    '''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    working_dir = settings.working_dir

    if rank == 0:
        # 创建日志文件
        my_logger = Log(f'{working_dir}/log/dsc_GenData.log', mode = 'w')

        # 生成所有可能的情况
        vector = delete_combination(settings)
        my_logger.info(np.size(vector, 0))

        os.makedirs(f'{working_dir}/data/dsc_tmp', exist_ok=True)
    else:
        vector = None

    vector = comm.bcast(vector, root=0) 

    vector_size = np.size(vector, 0)
    this_worker_job = get_job_index(njob = vector_size, rank = rank, worker_num = size)

    datas = one_core_task(rank, vector[this_worker_job], this_worker_job, settings, get_data)
    

    if gather_data:
        gathered_datas = comm.gather(datas, root=0)
        
        if rank == 0:
            my_logger = Log(f'{working_dir}/log/dsc_GenData.log')
            my_logger.info(f'recieving data start...')

            # 接收并合并其他进程的运行结果
            best_data = gathered_datas[0]
            for worker_i in range(1, size):
                # 接收数据
                datas = gathered_datas[worker_i]
                # 比较
                if datas is not None:
                    best_data = compare_loss(settings, best_data, datas)

            my_logger.info(f'recieve data finish')

            # 保存并输出best_datas的详细信息
            get_best_vector_info(settings, best_data, get_species_name, get_reduced_sp_reac)
            species_num = int(np.sum(best_data['vector']))
            np.savez(f'{working_dir}/data/dsc_data/{species_num}species.npz', **best_data)
            np.savez(f'{working_dir}/best_vector.npz', **best_data)


def gather_data(settings, get_species_name, get_reduced_sp_reac):
    working_dir = settings.working_dir

    if os.path.exists(f'{working_dir}/data/dsc_tmp'):
        best_data = None
        for file in os.listdir(f'{working_dir}/data/dsc_tmp'):
            try:
                datas = np.load(f'{working_dir}/data/dsc_tmp/{file}', allow_pickle=True)
                if best_data is None:
                    best_data = datas
                else:
                    best_data = compare_loss(settings, best_data, datas)
            except:
                pass
        
        # 保存并输出best_datas的详细信息
        get_best_vector_info(settings, best_data, get_species_name, get_reduced_sp_reac)
        species_num = int(np.sum(best_data['vector']))
        np.savez(f'{working_dir}/data/dsc_data/{species_num}species.npz', **best_data)
        np.savez(f'{working_dir}/best_vector.npz', **best_data)

        # 删除临时文件夹
        shutil.rmtree(f'{working_dir}/data/dsc_tmp')
            