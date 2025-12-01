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
# 创建TensorDataset（与训练时相同的数据结构）
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

def sensitivity_analysis_LFS(model, Alist_data, device, eps=1e-8):
    """
    LFS敏感性分析（无需对数变换）
    返回: (num_features, num_LFS_outputs) 的敏感性矩阵
    """
    model.eval()
    
    # 确保输入形状 (1, num_features)
    Alist_data = Alist_data.to(device)
    if Alist_data.dim() == 1:
        Alist_data = Alist_data.unsqueeze(0)
    
    # 获取原始LFS预测值
    with torch.no_grad():
        _, LFS_original = model(Alist_data)  # 假设第二个输出头是LFS
    
    # 生成扰动数据 (num_features, num_features)
    num_features = Alist_data.shape[1]
    perturbation_factor = torch.eye(num_features).to(device) * 0.05 + 1.0
    perturbed_data = Alist_data * perturbation_factor

    # 批量预测扰动后的LFS
    with torch.no_grad():
        _, LFS_perturbed = model(perturbed_data)
    
    # 计算敏感性矩阵
    delta_LFS = LFS_perturbed - LFS_original
    sensitivity_matrix = (delta_LFS / (LFS_original + eps)) / 0.05
    
    return sensitivity_matrix.cpu().numpy().squeeze()

# 修改主程序部分
if __name__ == "__main__":
    # 确保使用相同设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 转换输入数据
    Alist_tensor = torch.tensor(Alist_data, dtype=torch.float32).to(device)
    
    # 执行LFS敏感性分析
    lfs_sensitivity = sensitivity_analysis_LFS(model, Alist_tensor, device)
    
    # 输出形状验证
    print(f"LFS敏感性矩阵形状: {lfs_sensitivity.shape}")
    
    # 保存每个LFS输出的敏感性结果
    for output_idx in range(lfs_sensitivity.shape[1]):
        df = pd.DataFrame({
            "Feature Index": range(lfs_sensitivity.shape[0]),
            "Sensitivity": lfs_sensitivity[:, output_idx],
            "Effect": ["Positive" if x > 0 else "Negative" for x in lfs_sensitivity[:, output_idx]]
        })
        
        save_path = os.path.join(SAVE_DIR, f"LFS_sensitivity_output_{output_idx}.csv")
        df.to_csv(save_path, index=False)
        print(f"LFS敏感性结果已保存: {save_path}")

    # 打印示例
    sample = pd.read_csv(os.path.join(SAVE_DIR, "LFS_sensitivity_output_0.csv"))
    print("\nLFS敏感性示例（前5个特征）:")
    print(sample.head())