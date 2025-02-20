import torch
import torch.nn as nn
import torch.nn.functional as F

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        residual = x  # 残差连接
        out = self.activation(self.fc1(x))
        out = self.fc2(out)  # 第二层不经过激活函数，直接映射回原维度
        return out + residual  # 加上残差

    def _initialize_weights(self):
        # 初始化权重为零，使残差块初始为恒等映射
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


# 改进的PriorNetwork
class PriorNetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, num_blocks=2):
        super(PriorNetwork, self).__init__()

        # 残差分支：分别处理两个输入向量
        self.branch1 = nn.Sequential(
            *[ResidualBlock(input_dim, hidden_dim) for _ in range(num_blocks)],
        )
        self.branch2 = nn.Sequential(
            *[ResidualBlock(input_dim, hidden_dim) for _ in range(num_blocks)],
        )

        # 融合层
        self.fusion_layer = nn.Linear(output_dim * 2, output_dim)
        self._initialize_fusion()

    def _initialize_fusion(self):
        # 初始化融合层为仅传递 branch1 的输出
        nn.init.zeros_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)
        with torch.no_grad():
            self.fusion_layer.weight[:, :self.fusion_layer.out_features].fill_diagonal_(1)  # 对 branch1 保留50%权重
            # self.fusion_layer.weight[:, self.fusion_layer.out_features:].fill_diagonal_(0)  # 对 branch2 保留50%权重

    def forward(self, vector1, vector2):
        # 分别通过两个分支处理两个输入向量
        branch1_out = self.branch1(vector1)
        branch2_out = self.branch2(vector2)

        # 融合两个分支的输出
        combined = torch.cat([branch1_out, branch2_out], dim=-1)
        fused_output = self.fusion_layer(combined)
        return fused_output


if __name__ == "__main__":
    input_dim = 512
    hidden_dim = 256
    output_dim = 512
    batch_size = 32

    # 创建模型实例
    model = PriorNetwork()

    # 测试输入向量
    vector1 = torch.randn(batch_size, input_dim)
    vector2 = vector1.clone()  # 初始冻结的向量

    # 前向传播
    output = model(vector1, vector2)

    print("Vector1 Input:", vector1)
    print("Vector2 Input (Frozen):", vector2)
    print("Output Vector:", output)
    print("Is output equal to vector1?", torch.allclose(vector1, output, atol=1e-6))

    # 统计参数数量
    def count_parameters_detailed(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

    count_parameters_detailed(model)
