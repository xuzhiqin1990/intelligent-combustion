import numpy as np
import torch
import torch.optim as optim
import os
import time
import copy
import shutil
from dmgr.utils import *
from dmgr.models import *
from dmgr.base.DeePMR_base import DeePMR_base_class
from dmgr.visualization.plot_train_loss import plot_loss
from dmgr.data import load_dnn_data_x, load_dnn_data_y


def dnn_training(
        count,
        settings,
        parallel_train: bool = False,
        ):
    r'''
    加载DNN训练数据
    Args:
        ``mr``(``DeePMR_base_class``): DeePMR的基类
        ``count``(``int``): 第几次迭代
        ``settings``: DeePMR的配置文件
    '''
    mr = DeePMR_base_class(settings)
    
    # 找出需要训练的指标对应的配置文件
    indicators = []
    DNN_assist_config_list = []
    for i, indicator in enumerate(settings.indicators):
        if settings.DNN_assist[i] == True:
            config = getattr(settings, f'{indicator}_config')
            indicators.append(indicator)
            DNN_assist_config_list.append(config)

    # 加载输入数据
    x_train, x_test, input_dim = load_dnn_data_x(mr.dnn_input_data_path + f'/dnn_{count}.npz')

    # device = torch.device(settings.cuda_device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 对每个需要训练的指标对应的DNN进行训练
    if parallel_train and len(indicators) > 1:
        from multiprocessing import Pool
        p = Pool(processes=len(indicators))
        
        for i, indicator in enumerate(indicators):
            # 加载配置文件
            config = DNN_assist_config_list[i]
            # 训练神经网络
            p.apply_async(
                func=train_dnn_indicator, 
                args=(config, indicator, count, mr, input_dim, x_train, x_test, device))
        p.close()
        p.join()
    else:
        for i, indicator in enumerate(indicators):
            # 加载配置文件
            config = DNN_assist_config_list[i]
            train_dnn_indicator(config, indicator, count, mr, input_dim, x_train, x_test, device)




def train_dnn_indicator(config, indicator, count, mr, input_dim, x_train, x_test, device):
    config.train_index = count + 1
    train_logger = Log(mr.train_log_his_path[indicator] + f'/train_{config.train_index}.log', mode='w')
    train_logger.info('dnn training...')

    try:
        # 加载数据
        y_train, y_test, output_dim = load_dnn_data_y(mr.dnn_data_path[indicator] + f'/dnn_{count}.npz')

        # 加载模型
        model = load_dnn_model(config.model_name, config.hidden_units, input_dim, output_dim)

        pth_path = mr.model_pth_path[indicator]
        loss_path = mr.loss_his_path[indicator]
        json_path = mr.model_json_path[indicator]

        config.input_dim = np.size(x_train, 1)
        config.output_dim = np.size(y_train, 1)
        config.train_size = np.size(x_train, 0)
        config.test_size = np.size(x_test,  0)
        config.data_path = mr.dnn_data_path[indicator] + f'/data_{count}.npz'

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
        criterion = get_criterion(config.criterion)
        optimizer = get_optimizer(config.optimizer, config.lr, model)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

        # 保存初始化模型
        if count == 0:
            state = {'model': my_model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'epoch': 0}
            torch.save(state, f'{pth_path}/model_0.pth')  # 保存初始化模型

        ''' 训练神经网络 '''
        t0 = time.time()
        epoch_index, train_his, test_his = [], [], []
        config.batch_num = np.size(x_train, 0) // config.batch_size

        os.makedirs(f'{pth_path}/tmp', exist_ok=True)  # 创建临时保存网络参数的文件夹

        for epoch in range(start_epoch, end_epoch):
            my_model.eval()    # 预测
            with torch.no_grad():
                y_dnn_test = my_model(torch.FloatTensor(x_test).to(device))
                test_loss = float(
                    criterion(y_dnn_test, torch.FloatTensor(y_test).to(device)))
            my_model.train()   # 训练
            train_loss = 0
            for i in range(config.batch_num):  # 按照 batch 进行训练
                x_train_batch = x_train[i *
                                        config.batch_size: (i+1) * config.batch_size, :]
                y_train_batch = y_train[i *
                                        config.batch_size: (i+1) * config.batch_size, :]
                y_dnn_train_batch = my_model(
                    torch.FloatTensor(x_train_batch).to(device))
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
        plot_loss(loss_path, mr.model_path[indicator], config.train_index)
        config.stop_epoch = int(start_epoch + stop_index)

        config_dict = copy.deepcopy(vars(config))

        write_json_data(
            f'{json_path}/settings_{config.train_index}.json', config_dict)  # 保存DNN的超参数
    
    except Exception as e:
        train_logger.info(e)