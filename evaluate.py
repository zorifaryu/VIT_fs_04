import torch
from data.cifar10 import get_cifar10_dataloaders
from models.vit import VisionTransformer
from utils.evaluate import evaluate_model, plot_confusion_matrix
from utils.visualize import visualize_attention

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    print('Loading data...')
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=32)
    
    # 初始化模型
    print('Initializing model...')
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_heads=8,
        hidden_dim=3072,
        num_layers=12,
        num_classes=10,
        dropout=0.1
    ).to(device)
    
    # 加载模型权重
    print('Loading model weights...')
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    # 评估模型
    print('Evaluating model...')
    accuracy, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f'Accuracy: {accuracy:.2f}%')
    
    # 绘制混淆矩阵
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(y_true, y_pred, classes)
    print('Confusion matrix saved as confusion_matrix.png')
    
    # 可视化注意力权重
    print('Visualizing attention...')
    # 获取一个批次的图像
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        # 取第一张图像
        img = inputs[0]
        # 前向传播
        _, attn_weights_list = model(inputs)
        # 取最后一层的注意力权重
        attn_weights = attn_weights_list[-1]
        # 可视化注意力
        visualize_attention(img, attn_weights[0])
        print('Attention map saved as attention_map.png')
        break

if __name__ == '__main__':
    main()
