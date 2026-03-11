import torch
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练模型一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    
    Returns:
        train_loss: 训练损失
        train_acc: 训练准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, test_loader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        val_loss: 验证损失
        val_acc: 验证准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(test_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def get_optimizer(model, lr=1e-3):
    """
    获取优化器
    
    Args:
        model: 模型
        lr: 学习率
    
    Returns:
        optimizer: 优化器
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

def get_lr_scheduler(optimizer, T_max=100):
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        T_max: 最大迭代次数
    
    Returns:
        scheduler: 学习率调度器
    """
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return scheduler
