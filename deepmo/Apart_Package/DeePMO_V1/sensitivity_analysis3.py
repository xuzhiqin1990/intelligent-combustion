import numpy as np
import torch
import os
import json
import time
from DeePMO_V1_Network import Network_PlainDoubleHead

# 路径配置（请按你的实际路径修改）
SAVE_DIR = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35_alpha0.1_from_circ3/sensitivity_results"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_pth_path = '/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35_alpha0.1_from_circ3/model/model_pth/early_stopping/model_best_stopat_775_circ=4.pth'
data1_path = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35/inverse_skip/circ=3/0/IDT_data.npz"
data3_path = "/home/linpengxiao_pro/NUIGreduced_18sp_alpha0.35/data/true_data/true_idt.npz"

# 加载数据
Data1 = np.load(data1_path)
Data3 = np.load(data3_path)

Alist_data = Data1['Alist'].astype(np.float32)
all_idt_data = Data3['true_idt_data']

print("Alist_data shape:", Alist_data.shape)
print("all_idt_data.shape =", all_idt_data.shape)

# 输入 reshape 为 (1, num_reactions)
Alist_data = Alist_data.reshape(1, -1)  # (1, N)
input_dim = Alist_data.shape[1]

# 假设模型输出为双头：[IDT, LFS]，IDT为53个工况，LFS为1（如实际LFS是其他长度请修改）
output_dim = [53, 21]
hidden_units = [3000, 2000, 2000]  # 与训练保持一致

# 网络加载
model = Network_PlainDoubleHead(input_dim, hidden_units, output_dim).to(device)
checkpoint = torch.load(best_pth_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

def sensitivity_analysis_per_case_log(model, Alist_data, device, delta=0.001):
    """
    神经网络敏感性分析，输出 (num_reactions, num_conditions) 的敏感度矩阵
    假设输入是log(A)！
    """
    model.eval()
    Alist_data = Alist_data.to(device)
    if Alist_data.dim() == 1:
        Alist_data = Alist_data.unsqueeze(0)
    num_reactions = Alist_data.shape[1]

    with torch.no_grad():
        # 1. 原始预测 (log10(IDT))
        outputs = model(Alist_data)
        original_log_IDT = outputs[0] if isinstance(outputs, tuple) else outputs
        original_IDT = torch.pow(10, original_log_IDT)    # (1, 53)
        base_lnIDT = torch.log(original_IDT)              # (1, 53)

        # 2. 对每个反应加扰动
        delta_log = np.log(1 + delta)
        perturbed_Alist = Alist_data.repeat(num_reactions, 1)
        for i in range(num_reactions):
            perturbed_Alist[i, i] += delta_log  # 只扰动第i个反应的log(A)

        # 3. 批量预测扰动后IDT
        outputs_perturbed = model(perturbed_Alist)
        perturbed_log_IDT = outputs_perturbed[0] if isinstance(outputs_perturbed, tuple) else outputs_perturbed
        perturbed_IDT = torch.pow(10, perturbed_log_IDT)        # (num_reactions, 53)
        ln_perturbed_IDT = torch.log(perturbed_IDT)             # (num_reactions, 53)

        # 4. 灵敏度公式
        sensitivity_matrix = (ln_perturbed_IDT - base_lnIDT) / np.log(1 + delta)  # (num_reactions, 53)

    return sensitivity_matrix.cpu().numpy()


if __name__ == "__main__":
    # 时间统计起点
    time_start = time.perf_counter()

    Alist_tensor = torch.tensor(Alist_data, dtype=torch.float32).to(device)

    # 敏感度分析
    delta = 0.01  # 建议与Cantera一致
    sensitivity_matrix = sensitivity_analysis_per_case_log(model, Alist_tensor, device, delta=delta)
    num_reactions, num_conditions = sensitivity_matrix.shape

    print(f"sensitivity_matrix.shape = {sensitivity_matrix.shape}")

    # 保存为与Cantera一致的json格式
    for cond_idx in range(num_conditions):
        sensitivity_json = {f"R{feat_idx}": float(sensitivity_matrix[feat_idx, cond_idx]) for feat_idx in range(num_reactions)}
        save_path = os.path.join(SAVE_DIR, f"nn_sensitivity_case_{cond_idx:02d}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sensitivity_json, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        print(f"Saved: {save_path}")

    # 时间统计终点
    time_end = time.perf_counter()
    total_seconds = time_end - time_start
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print(f"\n神经网络敏感性分析总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")