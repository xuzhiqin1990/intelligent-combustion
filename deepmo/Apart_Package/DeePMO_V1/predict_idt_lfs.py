import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from DeePMO_V1_Network import Network_PlainDoubleHead
import os
import pandas as pd
# 定义保存目录
SAVE_DIR = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35_alpha0.1_from_circ3"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_pth_path = '/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35_alpha0.1_from_circ3/model/model_pth/early_stopping/model_best_stopat_775_circ=3.pth'
data_path = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35_alpha0.1_from_circ3/data/APART_data/apart_data_circ=3.npz"
#A数据路径
data1_path = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35/inverse_skip/circ=3/0/IDT_data.npz"
#LFS的数据路径
data2_path = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35/data/true_data/true_lfs.npz"
#IDT数据路径
data3_path = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35/data/true_data/true_idt.npz"
Data = np.load(data_path)
Data1 = np.load(data1_path)
Data2 = np.load(data2_path)
Data3 = np.load(data3_path)
Alist_data = Data1['Alist'].astype(np.float32)  # 保持与训练一致的精度
all_idt_data = Data3['true_idt_data']
all_lfs_data = Data2['true_lfs_data']



# 原代码对IDT取了对数
processed_idt = np.log10(all_idt_data.astype(np.float32))  # 与_TrainDataProcess一致
processed_idt = processed_idt.reshape(1, -1)  # 确保IDT数据是二维的
all_lfs_data = all_lfs_data.reshape(1, -1)  # 确保LFS数据是二维的
Alist_data = Alist_data.reshape(1, -1)  # 确保Alist数据是二维的
# 打印数据维度以确认
print(processed_idt.shape)
print(all_lfs_data.shape)
print(Alist_data.shape)

dataset = TensorDataset(
    torch.tensor(Alist_data), 
    torch.tensor(processed_idt),
    torch.tensor(all_lfs_data.astype(np.float32))
)

# 从训练数据中获取网络维度参数
input_dim = Alist_data.shape[1]
output_dim = [processed_idt.shape[1], all_lfs_data.shape[1]]  # IDT和LFS的输出维度
hidden_units = [3000, 2000, 2000]  # 必须与训练时的参数一致（需要确认实际参数）

# 初始化网络并加载权重
model = Network_PlainDoubleHead(input_dim, hidden_units, output_dim).to(device)
checkpoint = torch.load(best_pth_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()  # 设置为评估模式

# 执行预测（保持与训练相同的batch机制）
batch_size =1000  # 需要确认原训练参数
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 存储预测结果
idt_preds, lfs_preds = [], []

with torch.no_grad():
    for batch in loader:
        A_batch, _, _ = batch
        A_batch = A_batch.to(device)
        
        # 双头预测
        batch_idt = model.forward_Net1(A_batch)
        batch_lfs = model.forward_Net2(A_batch)
        
        idt_preds.append(batch_idt.cpu().numpy())
        lfs_preds.append(batch_lfs.cpu().numpy())

# 合并批次结果
idt_pred = np.concatenate(idt_preds, axis=0)
lfs_pred = np.concatenate(lfs_preds, axis=0)


def save_pred_to_csv(features, predictions, pred_name, columns_prefix="A"):
    """
    修复列名生成和数据合并问题
    """
    # 生成特征列名
    feature_columns = [f"{columns_prefix}{i}" for i in range(features.shape[1])]
    
    # 生成预测列名（处理多输出情况）
    if predictions.ndim == 1:
        pred_columns = [pred_name]
    else:
        pred_columns = [f"{pred_name}_{i}" for i in range(predictions.shape[1])]
    
    # 合并数据（确保维度对齐）
    combined_data = np.hstack([
        features.reshape(len(features), -1), 
        predictions.reshape(len(predictions), -1)
    ])
    
    # 创建DataFrame
    df = pd.DataFrame(
        data=combined_data,
        columns=feature_columns + pred_columns
    )
    
    # 保存到指定目录
    save_path = os.path.join(SAVE_DIR, f"{pred_name}.csv")
    df.to_csv(save_path, index=False)
    print(f"文件已保存至: {save_path}")

save_pred_to_csv(
    features=Alist_data, 
    predictions=idt_pred, 
    pred_name="IDT_pred_log10"
)

save_pred_to_csv(
    features=Alist_data,
    predictions=lfs_pred,
    pred_name="LFS_pred"
)

# 对IDT预测结果进行指数运算（撤销log10变换）
idt_pred_actual = np.power(10, idt_pred)

# LFS预测不需要逆变换（原代码未做处理）
lfs_pred_actual = lfs_pred

# 计算误差指标（与原数据对比）
idt_rmse = np.sqrt(np.mean((idt_pred_actual - all_idt_data)**2))
lfs_rmse = np.sqrt(np.mean((lfs_pred_actual - all_lfs_data)**2))

print(f"IDT预测RMSE: {idt_rmse:.4e} | LFS预测RMSE: {lfs_rmse:.4e}")
print(f"预测结果维度：IDT {idt_pred_actual.shape}，LFS {lfs_pred_actual.shape}")