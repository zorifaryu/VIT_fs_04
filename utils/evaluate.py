import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
    
    Returns:
        accuracy: 准确率
        y_true: 真实标签
        y_pred: 预测标签
    """
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
