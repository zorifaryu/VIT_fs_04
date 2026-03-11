import torch
import torch.nn as nn
from data.cifar10 import get_cifar10_dataloaders
from models.vit import VisionTransformer
from utils.train import train_epoch, validate, get_optimizer, get_lr_scheduler

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数
    batch_size = 32
    lr = 1e-4
    num_epochs = 100
    
    # 加载数据
    print('Loading data...')
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
    
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
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr=lr)
    scheduler = get_lr_scheduler(optimizer, T_max=num_epochs)
    
    # 训练模型
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')
    
    print(f'Final best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
