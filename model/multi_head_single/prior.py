import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorNetwork_multi_head_single(nn.Module):
    def __init__(self, vector_dim=512, num_heads=4):
        super(PriorNetwork_multi_head_single, self).__init__()
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        
        # 使用MultiheadAttention层
        self.multihead_attention = nn.MultiheadAttention(embed_dim=vector_dim, num_heads=num_heads)
        
        # 初始化多头注意力机制的权重矩阵
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 初始化Q, K, V线性变换的权重矩阵为单位矩阵
        for name, param in self.multihead_attention.named_parameters():
            if 'in_proj_weight' in name:
                with torch.no_grad():
                    param.copy_(torch.eye(self.vector_dim).repeat(3, 1))
            elif 'in_proj_bias' in name:
                with torch.no_grad():
                    param.zero_()
            elif 'out_proj.weight' in name:
                with torch.no_grad():
                    param.copy_(torch.eye(self.vector_dim))
            elif 'out_proj.bias' in name:
                with torch.no_grad():
                    param.zero_()
    
    def forward(self, v1, v2):
        """
        输入:
            v1: 第一个向量 (batch_size, vector_dim)
            v2: 第二个向量 (batch_size, vector_dim)
        输出:
            融合后的向量 (batch_size, vector_dim)
        """
        # 将两个向量堆叠起来，形成形状为 (batch_size, 2, vector_dim) 的张量
        stacked_vectors = torch.stack([v1, v2], dim=1)  # (batch_size, 2, vector_dim)
        
        # 调整输入形状以适应MultiheadAttention的要求
        query = key = value = stacked_vectors.transpose(0, 1)  # (2, batch_size, vector_dim)
        
        # 使用MultiheadAttention层进行注意力计算
        attn_output, _ = self.multihead_attention(query, key, value)  # (2, batch_size, vector_dim)
        
        # 获取融合后的向量
        fused_vector = attn_output[0]  # (batch_size, vector_dim)
        
        return fused_vector

# 示例使用
if __name__ == "__main__":
    # 假设向量维度为8，使用4个注意力头
    vector_dim = 512
    num_heads = 4
    
    # 创建两个相同的随机向量
    v1 = torch.randn(1, vector_dim)  # (1, 8)
    v2 =  v1.clone()  # (1, 8)
    
    # 创建Transformer注意力融合模型实例
    fusion_model = PriorNetwork_multi_head_single(vector_dim, num_heads)
    
    # 进行融合
    fused_vector = fusion_model(v1, v2)
    
    print("Input Vector v1:", v1)
    print("Input Vector v2:", v2)
    print("Fused Vector:", fused_vector)
    def count_parameters_detailed(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

    count_parameters_detailed(fusion_model)