import numpy as np
import torch
import os
import time
import copy
import shutil
from dmgr.utils import *
from dmgr.base.DeePGR_base import DeePGR_base_class
from dmgr.visualization.plot_train_loss import plot_loss
from dmgr.data import load_dnn_data_x, load_dnn_data_y

import scipy.sparse as sp
from scipy.sparse import csr_matrix



def dnn_training(
        count,
        settings,
        sparse_form = False,
        ):
    r'''
    加载DNN训练数据
    Args:
        ``gr``(``DeePGR_base_class``): DeePGR的基类
        ``count``(``int``): 第几次迭代
        ``global_config``(``GlobalConfig``): 全局配置文件
        ``indicator_config_list``(``list``): 指标配置文件列表
        ``sparse_form``(``bool``): 输入是否使用稀疏矩阵
    '''
    gr = DeePGR_base_class(settings)


    # 找出需要训练的指标对应的配置文件
    indicators = []
    DNN_assist_config_list = []
    for i, indicator in enumerate(settings.indicators):
        if settings.DNN_assist[i] == True:
            config = getattr(settings, f'{indicator}_config')
            indicators.append(indicator)
            DNN_assist_config_list.append(config)

    # 加载输入数据
    x_train, x_test = load_dnn_data_x(gr.dnn_input_data_path + f'/dnn_{count}.npz', sparse_form)

    # 对每个需要训练的指标对应的DNN进行训练
    for i, indicator in enumerate(indicators):
        # 加载配置文件
        config = DNN_assist_config_list[i]

        # 加载模型
        model = config.model

        # 加载数据
        y_train, y_test = load_dnn_data_y(gr.dnn_data_path[indicator] + f'/dnn_{count}.npz')

        pth_path = gr.model_pth_path[indicator]
        loss_path = gr.loss_his_path[indicator]
        json_path = gr.model_json_path[indicator]

        config.input_dim = np.size(x_train, 1)
        config.output_dim = np.size(y_train, 1)
        config.train_size = np.size(x_train, 0)
        config.test_size = np.size(x_test,  0)
        config.data_path = gr.dnn_data_path[indicator] + f'/data_{count}.npz'

        device = config.device

        # 判断第一次训练还是有预训练模型
        config.train_index = count + 1
        if count != 0:
            json_data = read_json_data(
                f'{json_path}/settings_{count}.json')
            start_epoch = int(json_data['stop_epoch'])
            end_epoch = int(start_epoch + config.epoch)
            # 实例化DNN，加载已训练的网络参数
            my_model = model.to(device)
            checkpoint = torch.load(
                f'{pth_path}/model_{count}.pth', map_location=device)
            my_model.load_state_dict(checkpoint['model'])
        else:  # 没有预训练网络的情形
            start_epoch, end_epoch = 0, int(config.epoch)
            my_model = model.to(device)

        # 定义损失和优化器
        criterion = config.criterion
        optimizer = config.get_optimizer(model)
        scheduler = config.get_scheduler()

        # 保存初始化模型
        if count == 0:
            state = {'model': my_model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'epoch': 0}
            torch.save(state, f'{pth_path}/model_0.pth')  # 保存初始化模型

        ''' 训练神经网络 '''
        t0 = time.time()
        train_logger = Log(gr.train_log_his_path[indicator] + f'/train_{config.train_index}.log', mode='w')
        train_logger.info('dnn training...')
        epoch_index, train_his, test_his = [], [], []
        config.batch_num = np.size(x_train, 0) // config.batch_size

        os.makedirs(f'{pth_path}/tmp', exist_ok=True)  # 创建临时保存网络参数的文件夹

        for epoch in range(start_epoch, end_epoch):
            my_model.eval()    # 预测
            with torch.no_grad():
                if sparse_form:
                    x_test_row, x_test_col, x_test_data = sp.find(x_test)
                    indices = torch.LongTensor([x_test_row.tolist(), x_test_col.tolist()])
                    values = torch.FloatTensor(x_test_data.tolist())
                    x = torch.sparse.FloatTensor(indices, values, torch.Size(x_test.shape))
                    x = x.float().to(device)
                else:
                    x = torch.FloatTensor(x_test).to(device)
                
                y_dnn_test = my_model(x)

                test_loss = float(
                    criterion(y_dnn_test, torch.FloatTensor(y_test).to(device)))
                
            my_model.train()   # 训练
            train_loss = 0
            for i in range(config.batch_num):  # 按照 batch 进行训练
                x_train_batch = x_train[i * config.batch_size: (i+1) * config.batch_size, :]
                y_train_batch = y_train[i * config.batch_size: (i+1) * config.batch_size, :]
                if sparse_form:
                    x_train_batch_row, x_train_batch_col, x_train_batch_data = sp.find(x_train_batch)
                    indices = torch.LongTensor([x_train_batch_row.tolist(), x_train_batch_col.tolist()])
                    values = torch.FloatTensor(x_train_batch_data.tolist())
                    x_train_batch = torch.sparse.FloatTensor(indices, values, torch.Size(x_train_batch.shape))
                    x_train_batch = x_train_batch.float().to(device)
                else:
                    x_train_batch = torch.FloatTensor(x_train_batch).to(device)

                y_dnn_train_batch = my_model(x_train_batch)

                train_loss_batch = criterion(
                    y_dnn_train_batch, torch.FloatTensor(y_train_batch).to(device))

                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()
                train_loss += float(train_loss_batch)

            scheduler.step()  # 调整学习率

            train_loss /= config.batch_num
            epoch_index.append(epoch)
            train_his.append(train_loss)
            test_his.append(test_loss)

            if (epoch - start_epoch) % 5 == 0:
                # 保存信息
                train_logger.info('epoch: {}\t train loss: {:.4e}   test loss: {:.4e}   time cost: {} s   lr:{:.2e}'
                                  .format(epoch, train_loss, test_loss, int(time.time()-t0),
                                          optimizer.param_groups[0]['lr']))
            if (epoch - start_epoch == 0) or ((epoch - start_epoch - 25) % 50 == 0):
                # 保存DNN模型
                state = {'model': my_model.state_dict(
                ), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(
                    state, f'{pth_path}/tmp/my_model_{epoch-start_epoch}.pth')

        train_logger.info(
            f'train dnn done! time cost: {time.time()-t0} second.')

        # early stopping，找出loss最低的那个点以及对应的网络参数
        Test_Loss = np.array(test_his).reshape(-1, 50)
        test_loss_sum = np.sum(Test_Loss, 1)
        stop_index = 50 * np.argmin(test_loss_sum) + 25

        shutil.copy(f'{pth_path}/tmp/my_model_{stop_index}.pth',
                    f'{pth_path}/model_{config.train_index}.pth')
        shutil.rmtree(f'{pth_path}/tmp', ignore_errors=True)  # 删除中间文件

        ''' 保存实验结果 '''
        # 保存DNN的loss
        np.savez(f'{loss_path}/loss_his_{config.train_index}.npz', 
                 epoch_index=epoch_index[: stop_index],
                 train_his=train_his[: stop_index],
                 test_his=test_his[: stop_index])
        
        # 更新loss图像
        plot_loss(loss_path, gr.model_path[indicator], config.train_index)
        config.stop_epoch = int(start_epoch + stop_index)

        config_dict = copy.deepcopy(vars(config))
        config_dict.pop('device')
        config_dict.pop('criterion')
        config_dict.pop('optimizer')
        config_dict.pop('scheduler')
        config_dict.pop('model')

        write_json_data(
            f'{json_path}/settings_{config.train_index}.json', config_dict)  # 保存DNN的超参数
