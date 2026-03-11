import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloaders(batch_size=32):
    """
    获取 CIFAR-10 数据集的训练和测试数据加载器
    
    Args:
        batch_size: 批处理大小
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VIT 通常使用 224x224 输入
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=2)
    
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=False, num_workers=2)
    
    return train_loader, test_loader
