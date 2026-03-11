import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """
    将图像分割成补丁并映射到嵌入空间
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层进行补丁嵌入
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x)  # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    """
    为每个补丁添加位置编码
    """
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 用于 CLS 标记
    
    def forward(self, x):
        # x: (batch_size, num_patches, embed_dim)
        batch_size = x.shape[0]
        # 添加 CLS 标记的位置编码
        cls_token = torch.zeros(batch_size, 1, x.shape[-1], device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # (batch_size, num_patches + 1, embed_dim)
        x = x + self.pos_embed  # 添加位置编码
        return x

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # 线性投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, embed_dim=768, hidden_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer 块
    """
    def __init__(self, embed_dim=768, num_heads=8, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # 自注意力子层
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.attn(x)
        x = self.dropout1(x)
        x = residual + x
        
        # 前馈子层
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x, attn_weights

class VisionTransformer(nn.Module):
    """
    Vision Transformer 模型
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, 
                 num_heads=8, hidden_dim=3072, num_layers=12, num_classes=10, dropout=0.1):
        super().__init__()
        
        # 补丁嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 位置编码
        num_patches = self.patch_embed.num_patches
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # 补丁嵌入
        x = self.patch_embed(x)
        
        # 添加位置编码
        x = self.pos_embed(x)
        
        # 经过 Transformer 层
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_list.append(attn_weights)
        
        # 分类
        x = self.norm(x)
        x = x[:, 0]  # 取 CLS 标记
        x = self.cls_head(x)
        
        return x, attn_weights_list
