import sys
import time
import argparse
sys.path.append('..')
from Apart_Package.utils.setting_utils import read_json_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, help = 'The mode of main.py')
    parser.add_argument('--circ', type = int, help = 'The circulation of the nn_sample', default = 0)
    parser.add_argument('--root', type = str, help = 'The root of inverse analysis', default = './inverse')
    mode_args = parser.parse_args()
    circ = mode_args.circ
    args = read_json_data(f"./model/model_pth/settings_circ={circ}.json")
    from Apart_Package.DeePMO_V1.MPI_process import GenData_IDT_LFS_MPI
    GenData_IDT_LFS_MPI(circ, args)