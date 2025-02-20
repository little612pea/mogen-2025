import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()  # 保持激活函数
        # self._initialize_weights()

    def forward(self, x):
        residual = x  # 残差连接
        out = self.activation(self.fc1(x))  # fc1 + 激活
        out = self.activation(self.fc2(out))  # fc2 映射回原维度
        return out + residual  # 加上残差并激活

    def _initialize_weights(self):
        # fc1 和 fc2 近似恒等映射
        # nn.init.kaiming_uniform_(self.fc1.weight, a=1)  # 保证足够的小扰动
        nn.init.zeros_(self.fc1.weight)  # 保证足够的小扰动
        nn.init.zeros_(self.fc1.bias)

        # fc2 精确初始化为恒等映射
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


class ExtractFC(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, num_blocks=2,disable = False):
        super(ExtractFC, self).__init__()

        # 原子动作分支
        self.atomic_action_branch = nn.Sequential(
            *[ResidualBlock(input_dim, hidden_dim) for _ in range(num_blocks)],
        )

        # 隐潜特征分支
        self.latent_feature_branch = nn.Sequential(
            *[ResidualBlock(input_dim, hidden_dim) for _ in range(num_blocks)],
        )
        self.disable_latent = disable

    def forward(self, x):
        # 两个分支解耦
        atomic_action_feature = self.atomic_action_branch(x)
        if not self.disable_latent:
            latent_feature = self.latent_feature_branch(x)
        else:
            latent_feature = torch.zeros_like(x) 
        # 拼接
        # combined_feature = torch.cat([atomic_action_feature, latent_feature], dim=-1)
        return atomic_action_feature, latent_feature


if __name__ == "__main__":
    # 假设输入的 text_embedding 是 512 维
    batch_size = 32
    text_embedding = torch.randn(batch_size, 512)
    print("text_embedding: ",text_embedding)
    # 实例化解耦网络
    model = ExtractFC()
    model.disable_latent= True
    # 前向传播
    atomic_action_feature, latent_feature = model(text_embedding)
    print("atomic_action_feature: ", atomic_action_feature)
    print("latent_feature: ",latent_feature)
    # def cosine_similarity(a, b):
    #     # 计算余弦相似度
    #     a_norm = F.normalize(a, dim=-1)
    #     b_norm = F.normalize(b, dim=-1)
    #     return torch.sum(a_norm * b_norm, dim=-1)

    # # 验证相似性
    # cos_sim = cosine_similarity(text_embedding, combined_feature)
    # print("Cosine Similarity (mean):", cos_sim.mean().item())
    def count_parameters_detailed(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

    count_parameters_detailed(model)
