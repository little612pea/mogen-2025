import torch
import torch.nn as nn
import torch.nn.functional as F

class AtomicActionMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim = 256):
        super(AtomicActionMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.disable = False
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        atomic_action_embedding = self.fc3(x)  # 输出为原子动作的 embedding
        return atomic_action_embedding


class DetailDescriptionMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim = 256):
        super(DetailDescriptionMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.disable = False
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        detail_description_embedding = self.fc3(x)  # 输出为细节描述的 embedding
        return detail_description_embedding


class ExtractFC_initial_exp(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256):
        super(ExtractFC_initial_exp, self).__init__()
        
        # 实例化原子动作和细节描述的 MLP 模块
        self.atomic_action_mlp = AtomicActionMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.detail_description_mlp = DetailDescriptionMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        
    def forward(self, x):
        # 使用两个 MLP 模块分别提取原子动作和细节描述的特征
        atomic_action_embedding = self.atomic_action_mlp(x)  # 输出形状为 (batch, 256)
        detail_description_embedding = self.detail_description_mlp(x)  # 输出形状为 (batch, 256)
        # cosine_similarity = F.cosine_similarity(atomic_action_embedding, detail_description_embedding, dim=-1)
        # 将两个嵌入拼接在一起，得到 512 维的输出
        combined_embedding = torch.cat([atomic_action_embedding, detail_description_embedding], dim=-1)
        # print("Cosine Similarity:", cosine_similarity)
        return combined_embedding

if __name__ == "__main__":
    # 假设输入的 text_embedding 是 512 维
    text_embedding = torch.randn(1, 512)  # 示例文本嵌入

    # 实例化 ExtractFC_initial_exp 模块
    extract_fc = ExtractFC_initial_exp(input_dim=512, hidden_dim=256, output_dim=256)

    # 获取 512 维的拼接嵌入
    combined_embedding = extract_fc(text_embedding)
    print("Combined Embedding Shape:", combined_embedding.shape)  # 输出形状为 (batch, 512)
    def count_parameters_detailed(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

    count_parameters_detailed(extract_fc)
