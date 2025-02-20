import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorNetwork_multi_head_comp(nn.Module):
    def __init__(self, vector_dim=512, num_heads=4, num_layers=4, ff_dim=512, dropout=0.1, max_seq_len=2,disable = False):
        super(PriorNetwork_multi_head_comp, self).__init__()
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.disable_latent = False
        # Transformer Encoder for single vector Full-Attention
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=vector_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # Multihead Attention for vector fusion
        self.multihead_attention = nn.MultiheadAttention(embed_dim=vector_dim, num_heads=num_heads)

        # Output projection
        self.output_layer = nn.Linear(vector_dim, vector_dim)

        # Initialize weights
        # self._initialize_weights()

    def _initialize_weights(self):
        # Initialize multihead attention weights for identity mapping
        for name, param in self.multihead_attention.named_parameters():
            if 'in_proj_weight' in name:
                with torch.no_grad():
                    param.copy_(torch.eye(self.vector_dim).repeat(3, 1))  # Q, K, V identity matrices
            elif 'in_proj_bias' in name or 'out_proj.bias' in name:
                with torch.no_grad():
                    param.zero_()
            elif 'out_proj.weight' in name:
                with torch.no_grad():
                    param.copy_(torch.eye(self.vector_dim))

        # Initialize Transformer encoder layer weights
        def initialize_identity_transformer_layer(layer, vector_dim):
            """
            Initialize a TransformerEncoderLayer to act as an identity mapping.
            Args:
                layer: An instance of TransformerEncoderLayer.
                vector_dim: Dimension of the input/output vectors.
            """
            with torch.no_grad():
                # Initialize self-attention weights and biases
                layer.self_attn.in_proj_weight.copy_(torch.eye(vector_dim).repeat(3, 1))  # Q, K, V -> Identity
                layer.self_attn.in_proj_bias.zero_()
                layer.self_attn.out_proj.weight.copy_(torch.eye(vector_dim))
                layer.self_attn.out_proj.bias.zero_()

                # Initialize feedforward network weights and biases
                layer.linear1.weight.zero_()  # Zero-out first FFN layer
                layer.linear1.bias.zero_()
                layer.linear2.weight.zero_()  # Zero-out second FFN layer
                layer.linear2.bias.zero_()

                # Initialize layer normalization parameters
                layer.norm1.weight.fill_(1.0)  # Scale factor = 1
                layer.norm1.bias.zero_()
                layer.norm2.weight.fill_(1.0)
                layer.norm2.bias.zero_()

        # Initialize all layers in TransformerEncoder
        for layer in self.transformer_encoder.layers:
            initialize_identity_transformer_layer(layer, self.vector_dim)

        # Output layer initialization
        with torch.no_grad():
            self.output_layer.weight.copy_(torch.eye(self.vector_dim))
            self.output_layer.bias.zero_()

    def forward(self, v1, v2):
        """
        Args:
            v1: First vector of shape (batch_size, vector_dim)
            v2: Second vector of shape (batch_size, vector_dim)
        Returns:
            Fused vector of shape (batch_size, vector_dim)
        """
        # Full-Attention: Process each vector independently
        batch_size = v1.size(0)

        def process_vector(vector):
            text_embedding = vector.unsqueeze(1).repeat(1, batch_size, 1)  # (batch_size, seq_len, vector_dim)
            transformer_input = text_embedding.permute(1, 0, 2)  # (seq_len, batch_size, vector_dim)
            transformer_output = self.transformer_encoder(transformer_input)  # (seq_len, batch_size, vector_dim)
            return transformer_output.mean(dim=0)  # Mean pooling (batch_size, vector_dim)

        v1_processed = process_vector(v1)  # (batch_size, vector_dim)
        if self.disable_latent:
            v2_processed = torch.zeros_like(v1)
        else:
            v2_processed = process_vector(v2)  # (batch_size, vector_dim)
        # Multihead Attention: Fusion
        stacked_vectors = torch.stack([v1_processed, v2_processed], dim=0)  # (2, batch_size, vector_dim)
        query = key = value = stacked_vectors  # (2, batch_size, vector_dim)
        attn_output, _ = self.multihead_attention(query, key, value)  # (2, batch_size, vector_dim)
        fused_vector = attn_output.mean(dim=0)  # Average the two outputs (batch_size, vector_dim)

        # Output projection
        fused_vector = self.output_layer(fused_vector)

        return fused_vector


if __name__ == "__main__":
    # Example usage
    vector_dim = 512
    num_heads = 4
    num_layers = 4
    ff_dim = 512
    max_seq_len = 2
    dropout = 0.1

    # Two input vectors
    v1 = torch.randn(2, vector_dim)  # (batch_size=8, vector_dim)
    v2 = v1.clone()  # Identical for testing

    # Initialize model
    fusion_model = PriorNetwork_multi_head_comp()
    fusion_model.disable_latent = True

    # Compute fused vector
    fused_vector = fusion_model(v1, v2)

    print("Input Vector v1:", v1)
    print("Input Vector v2:", v2)
    print("Fused Vector:", fused_vector)

    # Count parameters
    def count_parameters_detailed(model):
        for name, param in model.named_parameters():
            print(f"{name}: {param.numel()} parameters")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

    count_parameters_detailed(fusion_model)
    def cosine_similarity(a, b):
        # Normalize the embeddings along the feature dimension
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        # Compute the cosine similarity
        cos_sim = torch.sum(a_norm * b_norm, dim=-1)
        return cos_sim

    # 计算相似度
    cos_sim = cosine_similarity(v1,fused_vector)
    print(cos_sim)
