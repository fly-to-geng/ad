# 计算机视觉

## 公开数据集

- MNIST
- CIFAR-10/100
- (2010) ImageNet: A large-scale hierarchical image database
- (2014) Microsoft COCO: Common Objects in Context
- (2018) Semantic Understanding of Scenes Through the ADE20K Dataset

## 图像分类

- regnet
- (1998) LeNet
- (2012) AlexNet
- (2013) NIN
- (2014) VGG
- (2014) GoogleNet
- (2015) ResNet
- (2016) DenseNet
- (2017) SENet
- (2018) ShuffleNet
- (2018) MobileNetV2
- (2019) EfficientNet

- (2020) ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- (2021) Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- (2021) Masked Autoencoders Are Scalable Vision Learners
- (2022) Vision Transformer with Deformable Attention

## 2D 图像检测(出检测框)

目标检测就是在一张图像中找到我们关注的目标，并确定它的类别和位置，这是计算机视觉领域最核心的问题之一

传统目标检测方法：

SIFT（尺度不变特征变换）、HOG（方向梯度直方图）、DPM（一种基于组件的图像检测算法）

基于深度学习的目标检测方法：

RPN（region proposal network）

**二阶段方法**

先生成区域候选框，再通过卷积神经网络进行分类和回归修正。

常见算法有 RCNN、SPPNet、Fast RCNN，Faster RCNN 和 RFCN 等。二阶算法检测结果更精确

(2014) RCNN：Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation

是将深度学习应用到目标检测领域的开山之作，凭借卷积神经网络出色的特征提取能力，大幅度提升了目标检测的效果

(2015) Fast R-CNN

(2015) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

(2017) Feature Pyramid Networks # 提出多尺度特征

**一阶段方法**

不生成候选框，直接在网络中提取特征来预测物体的分类和位置。

常见算法有 SSD、YOLO系列 和 RetinaNet 等。一阶算法检测速度与更快

(2016) SSD: Single Shot MultiBox Detector

(2017) RetinaNet: Focal Loss for Dense Object Detection # Focal Loss 提出

(2017) YOLOv1: You Only Look Once: Unified, Real-Time Object Detection
(2017) YOLOv2: YOLO9000: Better, Faster, Stronger
(2018) YOLOv3: An Incremental Improvement.
(2020) YOLOv4: Optimal Speed and Accuracy of Object Detection

(2019) CenterNet: Objects as Points

**基于transformer的方法**

(2020) DETR: End-to-End Object Detection with Transformers

(2020) Deformable DETR: Deformable Transformers for End-to-End Object Detection

(2021) Conditional DETR for Fast Training Convergence

(2022) Anchor DETR: Query Design for Transformer-Based Detector

(2022) DAB-DETR: DYNAMIC ANCHOR BOXES ARE BETTER QUERIES FOR DETR
(2022) DN-DETR: Accelerate DETR Training by Introducing Query DeNoising
(2022) DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection


## 分割

- (2022) Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers
- (2023) LISA: REASONINGSEGMENTATION  VIALARGELANGUAGEMODEL (通过语言大模型进行分割任务)
- UNet (MICCAI'2016/Nat. Methods'2019)
- SegFormer (NeurIPS'2021)

## 3D 图像检测（Lidar点云和多角度图像）

激光雷达的问题：

1. 逆光丢点，激光的波长和可见光相似，需要过滤，逆光时过滤有问题出现激光点云丢失
2. 雨雾噪声，高反膨胀（反射率比较高的物体，在雨天激光雷达被雨水附着的情况下会发生物体膨胀变大的效果）
3. 多径反射，周围环境存在反射镜面的时候会在对面出现一个物体
4. 黑车点云缺失， 黑车下雨反射率低导致没有点云，检测不出物体来。

点云检测方案：


**Voxel-Based**

VoxelNet/SECOND/PointPillar/CenterPoint

**Point-Based**

PointNet/PointNet++/Point-RCNN

VoxelNet --> SECOND -->PointPillar;

PointNet --> PointNet++ --> Point-RCNN --> 3D-SSD


- (2017) [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation]()
- (2017) [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)
- (2017) VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
- (2018) [Frustum PointNets for 3D Object Detection from RGB-D Data](https://readpaper.com/pdf-annotate/note?pdfId=4542544466673295361&noteId=2052111209889635328)
- (2019) [PointPillars: Fast Encoders for Object Detection from Point Clouds]()
- (2020) [SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving]()
- (2021) [PointAugmenting: Cross-Modal Augmentation for 3D Object Detection]()
- (2021) CenterPoint: Center-based 3D Object Detection and Tracking

Lidar + 视觉 + BEV方案:

caddn： Categorical Depth Distribution Network for Monocular 3D Object Detection
- (2019) Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving
- (2019) Disentangling Monocular 3D Object Detection
- (2020) LSS: Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D
- (2020) RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving # 关键点集合约束
- (2021) FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection
- (2021) DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries
- (2022) BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers
- (2022) BEVFormer ++ : Improving BEVFormer for 3D Camera-only Object Detection: 1st Place Solution for Waymo Open Dataset Challenge 2022
- (2022) BEVSegFormer: Bird's Eye View Semantic Segmentation From Arbitrary Camera Rigs
- (2022) BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving
- (2022) BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation

带物体跟踪的方案：

传统方法先用检测方法检测出每帧的物体，在用后处理 matching 每帧的物体。
Simple online and realtime tracking with a deep association metric
Tracking without bells and whistles
IoU-matching: High-Speed tracking-by-detection without using image information
Re-ID similarity: Towards Real-Time Multi-Object Tracking


- (2018) Mono-Camera 3D Multi-Object Tracking Using Deep Learning Detections and PMBM Filtering
- (2020) 3D Multi-Object Tracking: A Baseline and New Evaluation Metrics

- (2021) DEFT: Detection Embeddings for Tracking
- (2021) SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking
- (2020) CenterTrack: Tracking Objects as Points
- (2021) CenterTrack: Center-based 3D Object Detection and Tracking
- (2021) MOTR: End-to-End Multiple-Object Tracking with TRansformer

- (2022) QD3DT: Monocular Quasi-Dense 3D Object Tracking
- (2022) MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries

TransTrack: Multiple-Object Tracking with Transformer.
TrackFormer: Multi-Object Tracking with Transformers.

带构建地图的方法：

- (2021) HDMapNet: An Online HD Map Construction and Evaluation Framework
- (2022) VectorMapNet: End-to-end Vectorized HD Map Learning

静态感知：

主要识别红绿灯，标识牌，车道标识，车道线等静态道路元素。

1. 传统方法：分别检测 + 后处理 + 跟踪, example: MobileEye, EyeQ3, EyeQ4

2. 主流方法： 传感器前融合 + BEV， example: EyeQ5, tesla, 理想HDMapNet

3. 下一代：BEV范式 + 时空记忆模块


处理高精地图显示变更的方法：

Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection