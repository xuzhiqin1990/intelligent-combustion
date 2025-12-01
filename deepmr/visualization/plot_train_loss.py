import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(file_path, save_path, iteration, xlog = False):
    r'''
    按顺序加载文件夹下所有loss的npz文件并画图
    Args:
        ``file_path``(``str``): 文件夹路径
        ``save_path``(``str``): 保存路径
        ``iteration``(``int``): 第几次迭代
        ``xlog``(``bool``): x轴是否使用log
    '''
    epoch_index = []
    train_loss_his = []
    test_loss_his = []

    # 加载文件夹下所有loss的npz文件
    l = [f'{file_path}/loss_his_{i}.npz' for i in range(1, iteration + 1)]
    for target_file in l:
        data = np.load(target_file)

        epoch_index.extend(data['epoch_index'])
        train_loss_his.extend(data['train_his'])
        test_loss_his.extend(data['test_his'])

    plt.plot(train_loss_his, lw=1.2, label='train')
    plt.plot(test_loss_his, 'r--', lw=1.2, label='test')

    plt.xlabel('epoch')
    plt.ylabel('loss (log scale)')

    if xlog:
        plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/loss_his.png')
    plt.close()