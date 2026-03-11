import torch
import torch.nn as nn
import torch.optim as optim
from data.cifar10 import get_cifar10_dataloaders
from models.vit import VisionTransformer
import time

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {running_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.3f}%')
    
    return running_loss/len(train_loader), 100.*correct/total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/len(test_loader), 100.*correct/total

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数
    batch_size = 128
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 50
    
    # 数据加载
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size, num_workers=8)
    
    # 模型初始化
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        scheduler.step()
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}%')
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}%')
        print('-' * 50)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.3f}%')
    
    print(f'Final best accuracy: {best_acc:.3f}%')

if __name__ == '__main__':
    main()