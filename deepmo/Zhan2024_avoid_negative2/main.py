
import sys
import time
import argparse

sys.path.append('..')
from Apart_Package.DeePMO_V1.DeePMO_Exp_IDT_LFS import DeePMO_IDT_LFS
from Apart_Package.utils.cantera_utils import *
from Apart_Package.utils.setting_utils import *
from Apart_Package.utils.yamlfiles_utils import *
from Apart_Package.APART_base import gatherAPARTdata

def gather_data(circ = 0, shrink_data = False, cover = True):
    cpu_nums = os.cpu_count() - 1
    gatherAPARTdata(npz_keys = ['IDT', 'T','Alist', 'LFS'], 
                         apart_data_keys = ['idt', 'T', 'Alist', 'lfs'],
                            save_path = "./data/APART_data/tmp", rm_tree = True, 
                            save_file_name = f"./data/APART_data/apart_data_circ={circ}.npz",
                            cpu_nums = cpu_nums, logger = Log(f"./log/Gatherdata_circ={circ}.log")
                         )
    mkdirplus("./data/APART_data/boarder_tmp")
    if shrink_data:
        try:
            apart_data_path = f"./data/APART_data/apart_data_circ={circ}.npz"
            aboarder_data_path = os.path.dirname(apart_data_path) + f"/Aboarder_apart_data_circ={circ}.npz"
            load_file_name = [
                apart_data_path,
                aboarder_data_path,
            ]
            if cover or not os.path.exists(aboarder_data_path) or os.path.getsize(aboarder_data_path) /1024 /1024 <= 1:
                gatherAPARTdata(npz_keys = ['IDT', 'T','Alist', 'LFS'], 
                         apart_data_keys = ['idt', 'T', 'Alist', 'lfs'],
                            save_path = "./data/APART_data/boarder_tmp", rm_tree = True, 
                            save_file_name = aboarder_data_path,
                            cpu_nums = cpu_nums, logger = Log(f"./log/Gatherdata_circ={circ}.log")
                         )
            mkdirplus("./data/APART_data/boarder_tmp")
        except:
            pass
        


# 多进程生成数据 CPU
def gen_data(circ = 0):
    t0 = time.time()
    anet = DeePMO_IDT_LFS(circ = circ, need_write_json = True, )
    anet.GenAPARTDataLogger.info(f"GenTrueIDTData FINISHED cost {time.time() - t0}")
    sample_size = anet.APART_args['sample_size'][circ] if isinstance(anet.APART_args['sample_size'], list) else anet.APART_args['sample_size']
    sample_process_iter = 0
    while sample_size > 50:
        t1= time.time(); 
        samples = anet.ASample(sample_size = sample_size, if_PSR_filter = False)
        # 使用多进程生成数据
        sample_size -= anet.GenDataFuture(samples = samples, start_sample_index = int(sample_size * sample_process_iter))
        sample_process_iter += 1
        anet.GenDataFuture(samples = anet.boarder_samples,
                                save_path = "./data/APART_data/boarder_tmp"
                               )
        anet.GenAPARTDataLogger.info(f"Sample Process Iter {sample_process_iter} Have sampled {anet.APART_args['sample_size'] - sample_size} points, " + 
                                     f"GenAPARTData FINISHED cost {time.time() - t1}")
    anet.GenAPARTDataLogger.info(f"GenAPARTData FINISHED cost {time.time() - t1}")
    
    anet = DeePMO_IDT_LFS(circ = circ, need_write_json = True, )
    anet.gather_apart_data()

# 生成的数据用于DNN训练 GPU
def DNN_train(circ = 0):
    anet = DeePMO_IDT_LFS(circ = circ,  need_write_json = False,)
    anet.DeePMO_train(shrink_strategy = True, PSR_train_outside_weight = 0)

    # shutil.copyfile("./model/model_pth/settings_circ=0.json", "./tmp_json.json")

# 单独反问题 GPU
def inverse_problem(circ):
    anet = DeePMO_IDT_LFS(circ = circ,  need_write_json = False, )
    anet.SkipSolveInverse(save_dirpath = f"./inverse_skip/circ={circ}", 
                                  father_sample = f"./data/APART_data/father_sample_circ={circ}.npz", 
                                  IDT_reduced_threshold = None,)    
    anet.SortALISTStat()

# 多进程生成数据 CPU
def sample_prepare(circ = 0):
    t0 = time.time()
    # 计算真实机理的IDT，并加载APART参数
    anet = DeePMO_IDT_LFS(circ = circ,  need_write_json = True,) 
    samples = anet.ASample()
    anet.GenAPARTDataLogger.info(f"IDT_condition = {anet.IDT_condition}")
    anet.GenAPARTDataLogger.info(f"The size of SAMPLES is {anet.samples.shape}")
    anet.GenAPARTDataLogger.info(f"Sample FINISHED cost {time.time() - t0}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, help = 'The mode of main.py')
    parser.add_argument('--circ', type = int, help = 'The circulation of the nn_sample', default = 0)
    parser.add_argument('--root', type = str, help = 'The root of inverse analysis', default = './inverse')
    mode_args = parser.parse_args()
    if mode_args.mode == 'sample_prepare':
        sample_prepare(mode_args.circ)
    if mode_args.mode == 'nn_sample_gen':
        gen_data(mode_args.circ)
    if mode_args.mode == 'nn_sample_gen_MPI':
        circ = mode_args.circ
        args = read_json_data(f"./model/model_pth/settings_circ={circ}.json")
        from Apart_Package.DeePMO_V1.MPI_process import GenData_IDT_LFS_MPI
        GenData_IDT_LFS_MPI(circ, args)
    if mode_args.mode == 'gather_data':
        gather_data(mode_args.circ, shrink_data = True)
    if mode_args.mode == 'nn_sample_train':
        DNN_train(mode_args.circ)   
    if mode_args.mode == 'inverse':
        inverse_problem(mode_args.circ)


