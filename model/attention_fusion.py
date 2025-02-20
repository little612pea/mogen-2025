import torch 
import torch.nn as nn
import torch.nn.functional as F
from mlp_fc import ExtractFC

class TrainablePositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_sequence_length, d_model)
        nn.init.constant_(self.embedding.weight, 0.)  # 初始化为 0

    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.embedding.weight.device).unsqueeze(0)
        return self.embedding(positions)  # (1, seq_len, d_model)

class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim=512, num_layers=4, num_heads=8, ff_dim=256, dropout=0.1, max_seq_len=2):
        super(FusionNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len  # 序列长度（2表示两个输入）

        # 可学习的位置编码
        self.position_embeddings = TrainablePositionEncoding(max_seq_len, embedding_dim)

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

        # 输出投影层
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
        self._initialize_model()
    
    def _initialize_model(self):                
        # 初始化位置编码为零
        nn.init.constant_(self.position_embeddings.embedding.weight, 0.)

        # 初始化 Transformer Encoder 为等值映射
        for layer in self.transformer_encoder.layers:
            # 自注意力部分初始化为仅关注自身
            with torch.no_grad():
                # 初始化 Query、Key 和 Value 的投影矩阵为单位矩阵
                layer.self_attn.in_proj_weight.zero_()
                layer.self_attn.in_proj_weight[:self.embedding_dim, :self.embedding_dim] = torch.eye(self.embedding_dim)
                layer.self_attn.in_proj_weight[self.embedding_dim:2*self.embedding_dim, :self.embedding_dim] = torch.eye(self.embedding_dim)
                layer.self_attn.in_proj_weight[2*self.embedding_dim:, :self.embedding_dim] = torch.eye(self.embedding_dim)
                layer.self_attn.in_proj_bias.zero_()

                # 初始化输出投影矩阵为单位矩阵
                layer.self_attn.out_proj.weight.copy_(torch.eye(self.embedding_dim))
                layer.self_attn.out_proj.bias.zero_()

                # 初始化前向传播层为恒等映射
                layer.linear1.weight.zero_()
                layer.linear1.weight[:, :self.embedding_dim] = torch.eye(layer.linear1.weight.size(0), layer.linear1.weight.size(1))
                layer.linear1.bias.zero_()

                layer.linear2.weight.zero_()
                layer.linear2.weight[:self.embedding_dim, :] = torch.eye(layer.linear2.weight.size(0), layer.linear2.weight.size(1))
                layer.linear2.bias.zero_()

        # 初始化输出层为恒等映射
        nn.init.eye_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, embedding1, embedding2):
        """
        Args:
            embedding1: Tensor of shape (batch, embedding_dim)
            embedding2: Tensor of shape (batch, embedding_dim)
        Returns:
            Fused embedding of shape (batch, embedding_dim)
        """
        # 构造输入 (batch, seq_len, embed_dim)，seq_len 为 2
        embeddings = torch.stack([embedding1, embedding2], dim=1)  # (batch, seq_len=2, embed_dim)

        # 添加位置编码
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.position_embeddings(seq_len)

        # 调整维度以适配 Transformer 的输入 (seq_len, batch, embed_dim)
        embeddings = embeddings.permute(1, 0, 2)
        print("EMBEDDINGS: ",embeddings)
        # Transformer 编码
        transformer_output = self.transformer_encoder(embeddings)
        print("OUTPUT: ",transformer_output)
        # 调整维度回到 (batch, seq_len, embed_dim)
        transformer_output = transformer_output.permute(1, 0, 2)
        print("OUTPUT PERMUTE: ",transformer_output)
        # 对两个向量使用注意力融合
        attention_weights = F.softmax(transformer_output.mean(dim=2), dim=1)  # (batch, seq_len)
        fused_embedding = (attention_weights.unsqueeze(-1) * transformer_output).sum(dim=1)  # 加权求和

        # 输出层映射至目标嵌入空间
        fused_embedding = self.output_layer(fused_embedding)

        return fused_embedding


if __name__ == "__main__":
    # 假设 embedding1 和 embedding2 是两个 (batch, embedding_dim) 维度的嵌入
    batch_size = 32
    text_embedding = torch.randn(batch_size, 512)

    # 实例化解耦网络
    model = ExtractFC()

    # 前向传播
    embedding1,embedding2 = model(text_embedding)
    print("embedding 1: ", embedding1)
    print("embedding 2: ", embedding2)
    # 实例化 Fusion Network
    fusion_network = FusionNetwork()

    # 获取融合后的 embedding
    fused_embedding = fusion_network(embedding1, embedding2)
    print("Fused Embedding Shape:", fused_embedding.shape)  # 应输出 (batch, embedding_dim)
    print("fused embedding: ", fused_embedding)