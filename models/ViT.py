import torch
from torch import nn


class MSA(torch.nn.Module):
    def __init__(self, dim_in: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.1):
        super(MSA, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.normalization_factor = dim_head ** -0.5
        self.w_qkv = nn.Linear(dim_in, dim_head * 3 * heads, bias=False)
        self.merge_heads = nn.Sequential(
            nn.Linear(heads * dim_head, dim_in),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.w_qkv(x) # shape (B, N, heads * 3 * dim_head)
        qkv = qkv.reshape(B, N, self.heads, 3, self.dim_head).permute(3, 0, 2, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # each has shape (B, heads, N, dim_head)
        attention = (q @ k.transpose(-1, -2)) * self.normalization_factor # shape (B, heads, N, N)
        attention = torch.softmax(attention, dim=-1) # shape (B, heads, N, N)
        attention = attention @ v # shape (B, heads, N, dim_head)

        heads = attention.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head) # shape (B, N, heads * dim_head)
        out = self.merge_heads(heads) # shape (B, N, dim_in)
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, dim_in: int, hidden_dim: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim_in),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim_in: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.msa = MSA(dim_in, heads, dim_head, dropout)
        self.norm1 = nn.LayerNorm(dim_in)
        self.ffn = FeedForward(dim_in, mlp_dim, dropout)
        self.norm2 = nn.LayerNorm(dim_in)

    def forward(self, x):
        x = x + self.msa(self.norm1(x)) # shape (B, N, dim_in)
        x = x + self.ffn(self.norm2(x)) # shape (B, N, dim_in)
        return x


class ViT(torch.nn.Module):
    def __init__(self, dim_in: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.1):
        super(ViT, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                TransformerBlock(dim_in, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) # shape (B, N, dim_in)
        return x




if __name__ == "__main__":
    from utils.utils import get_model_size_mb
    DIM_IN = 64
    BATCH_SIZE = 32

    # test ViT
    input_tensor = torch.randn(BATCH_SIZE, 16, DIM_IN)  # (B, N, C)
    print(f"Input tensor shape: {input_tensor.shape}")
    vit = ViT(dim_in=DIM_IN, depth=6, heads=8, dim_head=64, mlp_dim=128, dropout=0.1)
    output_tensor = vit(input_tensor)
    print(f"ViT output shape: {output_tensor.shape}")  # Expected: (B, 16, DIM_IN)
    print(f"ViT model size: {get_model_size_mb(vit):.2f} MB")

