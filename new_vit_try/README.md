# Vision Transformer (VIT) 图像分类项目

## 项目简介

本项目基于 Vision Transformer (VIT) 模型实现 CIFAR-10 数据集的图像分类任务，旨在为考研复试提供一个难度适中、贴合 VIT 核心知识点、易阐述的项目。项目分为两个版本：

1. **完整代码版本**：包含完整的 VIT 模型实现和训练流程，适合深入理解 VIT 核心原理
2. **轻量级测试版本**：针对时间限制优化，控制运行时间在 2 小时内，适合快速验证和演示

## 技术栈

- **框架**：PyTorch
- **数据集**：CIFAR-10
- **硬件要求**：GPU 环境（充分利用 GPU 资源）

## 核心知识点

### 1. Vision Transformer 核心原理

- **Patch Embedding**：将图像分割为固定大小的 patches，并通过线性投影转换为向量表示
- **位置编码**：为每个 patch 添加位置信息，使模型能够理解空间关系
- **多头自注意力**：捕获不同尺度的特征信息，增强模型的表达能力
- **前馈神经网络**：对注意力输出进行非线性变换，进一步提取特征
- **Layer Normalization**：稳定训练过程，加速收敛

### 2. 模型架构

```
VisionTransformer
├── PatchEmbedding
├── Positional Encoding
├── Transformer Blocks (depth=6)
│   ├── MultiHeadAttention
│   ├── FeedForward Network
│   └── Layer Normalization
└── Classification Head
```

### 3. 训练策略

- **优化器**：AdamW 优化器，结合权重衰减
- **学习率调度**：余弦退火学习率调度
- **数据增强**：随机裁剪、水平翻转
- **批次大小**：128（完整版本）/ 256（轻量版本）
- **训练轮数**：50（完整版本）/ 20（轻量版本）

## 项目结构

```
new_vit_try/
├── complete_code/         # 完整代码版本
│   ├── data/              # 数据加载模块
│   │   └── cifar10.py     # CIFAR-10 数据加载和预处理
│   ├── models/            # 模型定义
│   │   └── vit.py         # VIT 核心模型实现
│   ├── train.py           # 训练和测试脚本
│   └── best_model.pth     # 训练完成的最佳模型（运行后生成）
├── lightweight_test/      # 轻量级测试版本
│   ├── test.py            # 集成的训练和测试脚本
│   └── lightweight_best_model.pth  # 轻量级模型（运行后生成）
└── README.md              # 项目说明文档
```

## 运行说明

### 环境配置

1. 确保安装了 PyTorch 和 torchvision
2. 确保 GPU 环境可用

### 运行完整版本

```bash
cd complete_code
python train.py
```

### 运行轻量级版本

```bash
cd lightweight_test
python test.py
```

## 性能指标

- **完整版本**：在 CIFAR-10 数据集上可达约 90%+ 的准确率
- **轻量级版本**：在 2 小时内可达约 85%+ 的准确率

## 复试口述要点

1. **项目动机**：选择 VIT 模型是因为它代表了深度学习的前沿方向，将 NLP 中的 Transformer 架构成功应用到计算机视觉领域

2. **核心创新**：
   - 采用 patch embedding 将图像转换为序列数据
   - 使用自注意力机制捕获全局特征
   - 避免了传统 CNN 中的归纳偏置，提供了更强的表达能力

3. **技术难点**：
   - 如何处理图像的空间信息（通过位置编码解决）
   - 如何平衡模型复杂度和训练效率（通过深度和嵌入维度的选择）
   - 如何提高 GPU 利用率（通过 batch size 优化和数据加载优化）

4. **实验结果**：
   - 完整模型在 CIFAR-10 上的表现
   - 轻量级模型的时间效率和精度权衡

5. **项目价值**：
   - 理解 Transformer 架构在视觉任务中的应用
   - 掌握深度学习模型的训练和优化技巧
   - 实践 GPU 加速计算的方法

## 简历呈现要点

- **项目名称**：基于 Vision Transformer 的图像分类系统
- **技术栈**：PyTorch, GPU 加速, 深度学习
- **核心贡献**：
  - 实现了完整的 Vision Transformer 模型，包括 patch embedding、多头自注意力等核心组件
  - 优化了模型训练策略，提高了 GPU 利用率
  - 设计了轻量级版本，在时间限制内完成训练和验证
- **成果**：在 CIFAR-10 数据集上实现了约 90% 的分类准确率

## 总结

本项目通过实现 Vision Transformer 模型，深入理解了 Transformer 架构在计算机视觉领域的应用，同时通过优化训练策略和模型结构，确保了在 GPU 环境下的高效运行。项目难度适中，涵盖了 VIT 的核心知识点，适合在考研复试中进行阐述和展示。