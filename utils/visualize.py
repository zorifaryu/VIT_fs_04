import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(img, attn_weights, patch_size=16):
    """
    可视化注意力权重
    
    Args:
        img: 输入图像 (3, H, W)
        attn_weights: 注意力权重 (num_heads, seq_len, seq_len)
        patch_size: 补丁大小
    """
    # 选择第一个注意力头
    attn = attn_weights[0].detach().cpu().numpy()
    
    # 只关注 CLS 标记对其他补丁的注意力
    cls_attn = attn[0, 1:]  # 跳过 CLS 标记本身
    
    # 计算补丁数量
    num_patches = cls_attn.shape[0]
    side_length = int(np.sqrt(num_patches))
    
    # 重塑注意力权重为二维
    cls_attn = cls_attn.reshape(side_length, side_length)
    
    # 调整注意力图大小以匹配原始图像
    attn_map = plt.imshow(cls_attn, cmap='viridis', interpolation='nearest')
    
    # 显示原始图像和注意力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原始图像
    img = img.permute(1, 2, 0).numpy()
    img = (img + 1) / 2  # 反归一化
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 显示注意力图
    im = ax2.imshow(cls_attn, cmap='viridis', interpolation='nearest')
    ax2.set_title('Attention Map')
    ax2.axis('off')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight')
    
    plt.tight_layout()
    plt.savefig('attention_map.png')
    plt.close()
