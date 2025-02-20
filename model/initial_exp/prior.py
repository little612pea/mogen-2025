import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorNetwork_initial_exp(nn.Module):
    def __init__(self, embedding_dim=512, num_layers=4, num_heads=8, ff_dim=256, dropout=0.1, max_seq_len=10):
        super(PriorNetwork_initial_exp, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # 可学习的位置编码
        self.position_embeddings = nn.Parameter(torch.randn(max_seq_len, embedding_dim))
        
        # Transformer Encoder Layer
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu'
        )
        
        # 多层 Transformer Encoder 堆叠
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        
        # 输出投影层将 Transformer 输出映射到目标嵌入空间
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, text_embedding):
        # 扩展输入维度以适配 Transformer 的输入 (batch, seq_len, embed_dim)
        if text_embedding.dim() == 2:
            text_embedding = text_embedding.unsqueeze(1).repeat(1, self.max_seq_len, 1)  # Repeat到max_seq_len长度
        
        # 添加位置编码
        text_embedding = text_embedding + self.position_embeddings.unsqueeze(0)

        # 调整维度以适配 Transformer 的输入 (seq_len, batch, embed_dim)
        text_embedding = text_embedding.permute(1, 0, 2)
        
        # 全局注意力通过 Transformer 编码器
        transformer_output = self.transformer_encoder(text_embedding)
        
        # 调整维度回到 (batch, seq_len, embed_dim)
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # 使用全序列的注意力，将所有位置的表示进行平均
        motion_embedding = transformer_output.mean(dim=1)  # 全局平均池化
        motion_embedding = self.output_layer(motion_embedding)  # 投影至目标嵌入空间
        
        return motion_embedding
        
if __name__ == "__main__":
    # 假设 text_embedding 是 (batch, embedding_dim) 维度‘】
    text_embedding = torch.randn(64, 512)  # 示例文本嵌入

    # 实例化 Prior Network
    prior_network = PriorNetwork_initial_exp()

    # 获取 motion embedding
    motion_embedding = prior_network(text_embedding)
    print("Motion Embedding Shape:", motion_embedding.shape)  # 应该输出 (batch, embedding_dim)
    def count_parameters_detailed(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

    count_parameters_detailed(prior_network)