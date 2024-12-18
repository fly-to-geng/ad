# 基于激光点云的目标检测

点云数据的特点：

1. Permutation Invariance，顺序不变性，点云没有天然的顺序，无论以怎样的顺序输入，输出结果都应该保持不变
2. Transformation Invariance， 旋转不变性，点云整体平移旋转之后，代码点的坐标的数值会发生变化，但是输出结果应该保持不变
3. 不规则，稀疏，位置上临近的点不是独立的，往往多个点构成一个语义上的物体。

处理点云数据的方法大致上可以分为两类：
1. 把3D空间划分为小的立方体，点云依照小的立方体分组，最后形成 H * W * C * D 的三维空间特征， 例如 VoxelNet， SECOND
2. 以BEV的视角把点云拍平到H*W上，高度这一维度收缩到特征维度，或者直接使用maxpooling聚合, 例如 PointPillars


**Voxel-Based**

VoxelNet --> SECOND -->PointPillar --> CenterPoint;

**Point-Based**

PointNet --> PointNet++ --> Point-RCNN --> 3D-SSD

- (2017) PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- (2017) PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- (2017) VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
- (2018) Frustum PointNets for 3D Object Detection from RGB-D Data
- (2019) PointPillars: Fast Encoders for Object Detection from Point Clouds
- (2020) SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving
- (2021) PointAugmenting: Cross-Modal Augmentation for 3D Object Detection
- (2021) CenterPoint: Center-based 3D Object Detection and Tracking


## PointNet

https://arxiv.org/pdf/1612.00593

输入： 没有经过排序的点云坐标， 每个点(x, y, z, ...) + 其他属性，例如强度

输出：

针对 object classification 任务， 有k个类别，输出每个类别的得分。

针对 semantic segmentation 任务，输出 n * m, n 为点的个数， m为语义的类别数量，也就是为每个点输出属于每个类别的概率

点云的特点：

1. Permutation Invariance， 无序， 要求模型可以以任意序列输入，不影响模型的输出结果
2. Transformation Invariance， 整体旋转和平移点云坐标，不应该对模型输出的类别和分割结果造成影响，模型应该关注空间位置相对关系，而不是绝对坐标
3. 虽然没有顺序，但是位置上相邻的点不是孤立的，通常有关系，例如共同组成一个物体

为了使模型达到第一点，有三种方案：
1. 输入排序，每次都用固定排好序的输入进入模型处理
2. 数据增强，训练的时候以各种置换之后的数据输入模型，增强模型对数据置换不变性的理解
3. 使用对输入顺序不敏感的函数，例如加法，乘法，对输入数据的顺序都没有要求，都会得到一样的结果

PointNet 选择使用第三种方案，先用共享的MLP分别处理每个点， 再用 max pooling 处理成一个单独的特征，最后使用这个特征作为分类的输入，得到K个类别的结果。取最大值的操作对输入顺序不敏感。

为了使模型达到第二点， PointNet 借鉴了  Spatial Transformer Networks 中的方法，使用一个小的网络预测一个固定的转换矩阵，输入的特征 乘以 这个转换矩阵之后，把位置正则化标准化，然后
使用标准化之后的特征作为预测的输入

为了使模型达到第三点， PointNet 使用全局特征拼接到每个点的特征的后面，然后使用这个特征预测这个点的类别归属，完成分割任务，简洁有效。

## PointNet++

pointnet 无法捕获不同尺度的局部特征，PointNet++就是为了解决这个问题。

pointnet 只是把每个点的特征 拼接上 全局特征作为最终特征，缺少了不同尺度的局部特征，所以这方面比较差。

PointNet++ 先使用最远点采样找出多个中心点，在把中心点附近的点作为一个 group, 按照 pointnet 的方法得到这些点的整体特征，拼接到某个点的特征上，这样

每个点既有这个点的特征，也有这个点局部的特征，也有全局的特征，可以更好的适应不同的场景。

## PointPillars

按照 x, y 坐标划分点云中的点，落到同一个方格内的一起处理，最终处理成 CHW 格式的伪图片特征，利用图像检测的各种算法处理这个特征，最终输出检测框。

## VoxelNet

点云映射到体素(立方体小方格)，按照体素为单位提取每个点的特征和整体特征，简单拼接后作为最终的特征。
