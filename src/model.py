import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # Escalo los pasos de (0,1) por un enfoque de DDPM
        time = time.view(-1) * 1000.0

        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttentionBlock(nn.Module):
    # Mejora a (CBAM).
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.group_norm = nn.GroupNorm(32, channels)

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        norm_x = self.group_norm(x)

        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)

        attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(attn_output)

    def get_attention_map(self, x):
        B, C, H, W = x.shape
        norm_x = self.group_norm(x)

        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)

        attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(attn_output), attn_output

class DDPMResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))

        emb_out = self.time_emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class CIFAR10UNet(nn.Module):
    def __init__(self, channels: List[int] = [128, 256, 256, 256], num_res_blocks: int = 2, t_embed_dim: int = 512, num_classes: int = 11):
        super().__init__()
        self.channels = channels

        self.time_embedder = SinusoidalPositionEmbeddings(t_embed_dim)
        self.y_embedder = nn.Embedding(num_embeddings=num_classes, embedding_dim=t_embed_dim)

        self.emb_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim)
        )

        self.init_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)

        # 3. Encoder
        self.downs = nn.ModuleList([])
        in_c = channels[0]
        current_res = 32

        for i, out_c in enumerate(channels):
            is_last = (i == len(channels) - 1)
            blocks = nn.ModuleList([])
            for _ in range(num_res_blocks):
                blocks.append(DDPMResBlock(in_c, out_c, t_embed_dim))
                in_c = out_c
                if current_res in [16, 8]:
                    blocks.append(SelfAttentionBlock(in_c))

            downsample = nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1) if not is_last else nn.Identity()
            self.downs.append(nn.ModuleDict({'blocks': blocks, 'downsample': downsample}))
            if not is_last: current_res //= 2

        # 4. Midcoder
        self.mid_block1 = DDPMResBlock(in_c, in_c, t_embed_dim)
        self.mid_attn = SelfAttentionBlock(in_c)
        self.mid_block2 = DDPMResBlock(in_c, in_c, t_embed_dim)

        # 5. Decoder
        self.ups = nn.ModuleList([])
        reversed_channels = list(reversed(channels))

        for i, out_c in enumerate(reversed_channels):
            is_last = (i == len(reversed_channels) - 1)
            blocks = nn.ModuleList([])
            for _ in range(num_res_blocks):
                blocks.append(DDPMResBlock(in_c + out_c, out_c, t_embed_dim))
                in_c = out_c
                if current_res in [16, 8]:
                    blocks.append(SelfAttentionBlock(in_c))

            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
            ) if not is_last else nn.Identity()

            self.ups.append(nn.ModuleDict({'blocks': blocks, 'upsample': upsample}))
            if not is_last: current_res *= 2

        # 6. Salida
        self.final_norm = nn.GroupNorm(32, in_c)
        self.final_conv = nn.Conv2d(in_c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        t_emb = self.time_embedder(t)
        y_emb = self.y_embedder(y)
        emb = self.emb_mlp(t_emb + y_emb)

        x = self.init_conv(x)
        skips = []

        # 3. Encoder
        for level in self.downs:
            for block in level['blocks']:
                if isinstance(block, DDPMResBlock):
                    x = block(x, emb)
                    skips.append(x)
                else:
                    x = block(x)
            x = level['downsample'](x)

        # 4. Midcoder
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # 5. Decoder
        for level in self.ups:
            for block in level['blocks']:
                if isinstance(block, DDPMResBlock):
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = block(x, emb)
                else:
                    x = block(x)
            x = level['upsample'](x)

        x = F.silu(self.final_norm(x))
        return self.final_conv(x)

    def sample_hidden_states(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        t_emb = self.time_embedder(t)
        y_emb = self.y_embedder(y)
        emb = self.emb_mlp(t_emb + y_emb)

        x = self.init_conv(x)
        skips = []

        # 3. Encoder
        for level in self.downs:
            for block in level['blocks']:
                if isinstance(block, DDPMResBlock):
                    x = block(x, emb)
                    skips.append(x)
                else:
                    x = block(x)
            x = level['downsample'](x)

        # 4. Midcoder
        x = self.mid_block1(x, emb)
        # x = self.mid_attn(x)
        x, attn_map = self.mid_attn.get_attention_map(x)
        x = self.mid_block2(x, emb)
        h = x.clone().detach()

        # 5. Decoder
        for level in self.ups:
            for block in level['blocks']:
                if isinstance(block, DDPMResBlock):
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = block(x, emb)
                else:
                    x = block(x)
            x = level['upsample'](x)

        x = F.silu(self.final_norm(x))
        return self.final_conv(x), h, attn_map