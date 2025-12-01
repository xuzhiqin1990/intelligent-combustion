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

def sensitivity_analysis(model, Alist_data, device, eps=1e-8):
    """
    执行敏感性分析并返回排序后的结果（修复元组错误）
    返回格式: [(特征索引, 平均敏感性), ...]
    """
    model.eval()
    
    # 确保输入形状正确 (batch_size=1, num_features)
    Alist_data = Alist_data.to(device)
    if Alist_data.dim() == 1:
        Alist_data = Alist_data.unsqueeze(0)
    
    #---------------- 1. 获取原始IDT预测值 ----------------#
    with torch.no_grad():
        outputs = model(Alist_data)
        # 明确提取IDT预测头（假设 outputs 是 (IDT_pred, LFS_pred)）
        IDT_original = outputs[0] if isinstance(outputs, tuple) else outputs
    
    #---------------- 2. 生成扰动数据（向量化） ----------------#
    num_features = Alist_data.shape[1]
    perturbation_factor = torch.eye(num_features) * 0.05 + 1.0
    perturbation_factor = perturbation_factor.to(device)
    perturbed_data = Alist_data * perturbation_factor  # 形状: (num_features, num_features)
    
    #---------------- 3. 批量预测扰动后的IDT ----------------#
    with torch.no_grad():
        outputs_perturbed = model(perturbed_data)
        # 明确提取IDT预测头
        IDT_perturbed = outputs_perturbed[0] if isinstance(outputs_perturbed, tuple) else outputs_perturbed
    
    #---------------- 4. 验证输出类型 ----------------#
    # 确保 IDT_original 和 IDT_perturbed 是张量
    assert isinstance(IDT_original, torch.Tensor), f"IDT_original 应为张量，实际类型: {type(IDT_original)}"
    assert isinstance(IDT_perturbed, torch.Tensor), f"IDT_perturbed 应为张量，实际类型: {type(IDT_perturbed)}"
    
    #---------------- 5. 计算敏感性指标 ----------------#
    IDT_deviation = IDT_perturbed - IDT_original
    eta = (IDT_deviation / (-1) * (IDT_original + eps)) / 0.05
    
    # 计算每个特征的平均敏感性
    average_eta = eta.mean(dim=1)  # 沿IDT指标维度平均
    
    #---------------- 6. 排序并返回结果 ----------------#
    sorted_indices = torch.argsort(average_eta, descending=True)
    return [(idx.item(), average_eta[idx].item()) for idx in sorted_indices]

# 示例用法
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Alist_tensor = torch.tensor(Alist_data, dtype=torch.float32).to(device)
    
    # 执行敏感性分析
    sensitivity_results = sensitivity_analysis(model, Alist_tensor, device)
    
    # 保存结果到CSV
    df = pd.DataFrame(sensitivity_results, columns=["Reaction Index", "Average Sensitivity"])
    df.to_csv(os.path.join(SAVE_DIR, "sensitivity_ranking.csv"), index=False)
    
    # 打印结果
    print("敏感性排序（从高到低）:")
    print(df.to_string(index=False))