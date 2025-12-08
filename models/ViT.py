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
    def __init__(
        self,
        image_size : int   = 512,
        patch_size : int   = 16,
        dim_in     : int   = 768,
        depth      : int   = 12,
        heads      : int   = 8,
        mlp_dim    : int   = 2048,
        channels   : int   = 1,
        dim_head   : int   = 64,
        dropout    : float = 0.1
    ):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2

        # patch embedding
        self.patch_to_embedding = nn.Sequential(
            nn.Conv2d(channels, dim_in, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2) # shape (B, dim_in, N)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim_in))
        self.dropout = nn.Dropout(p=dropout)


        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                TransformerBlock(dim_in, heads, dim_head, mlp_dim, dropout)
            )

        self.ln_final = nn.LayerNorm(dim_in)

    def forward(self, x):
        # image shape: (B, C, H, W)

        # patch embedding
        x = self.patch_to_embedding(x) # shape (B, dim_in, N)
        x = x.permute(0, 2, 1) # shape (B, N, dim_in)
        B, N, _ = x.shape

        # add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1) # shape (B, 1, dim_in)
        x = torch.cat((cls_tokens, x), dim=1) # shape (B, N + 1, dim_in)

        # add positional embedding
        x = x + self.positional_embedding[:, :N + 1, :] # shape (B, N + 1, dim_in)
        x = self.dropout(x) # shape (B, N + 1, dim_in)

        # transformer blocks
        for layer in self.layers:
            x = layer(x) # shape (B, N, dim_in)
        x = self.ln_final(x) # shape (B, N, dim_in)
        return x


if __name__ == "__main__":
    from utils.utils import get_model_size_mb
    DIM_IN = 64
    BATCH_SIZE = 32

    # test ViT
    input_tensor = torch.randn((BATCH_SIZE, 1, 512, 512))
    print(f"Input tensor shape: {input_tensor.shape}")
    vit = ViT(
        image_size=512,
        patch_size=16,
        dim_in=DIM_IN,
        depth=6,
        heads=8,
        mlp_dim=256,
        channels=1,
        dim_head=8,
        dropout=0.1
    )

    output_tensor = vit(input_tensor)
    print(f"ViT output shape: {output_tensor.shape}")  # Expected: (B, N, DIM_IN)
    print(f"ViT model size: {get_model_size_mb(vit):.2f} MB")

