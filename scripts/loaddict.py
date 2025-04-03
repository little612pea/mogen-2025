import torch

# 指定 .ckpt 文件路径
ckpt_path = "motionx.ckpt"

# 加载 checkpoint 文件
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

# 打印所有键
print("Checkpoint keys:", checkpoint.keys())

# 打印模型权重的键
if 'state_dict' in checkpoint:
    print("\nState dict keys:")
    for key in checkpoint['state_dict'].keys():
        print(key)

# # 打印优化器状态
# if 'optimizer_states' in checkpoint:
#     print("\nOptimizer states:", checkpoint['optimizer_states'])

# # 打印全局步数和当前轮数
# if 'global_step' in checkpoint:
#     print("\nGlobal step:", checkpoint['global_step'])

# if 'epoch' in checkpoint:
#     print("Epoch:", checkpoint['epoch'])

# # 打印超参数
# if 'hyper_parameters' in checkpoint:
#     print("\nHyper parameters:", checkpoint['hyper_parameters'])