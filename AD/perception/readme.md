# perception

## 数据集

1. https://www.cityscapes-dataset.com/dataset-overview/

点云检测：

1. PointPillars: Fast Encoders for Object Detection From Point Clouds

3D 目标检测数据集：

1. KITTI： http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
2. Nuscene: https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any
3. waymo: https://waymo.com/open
4. lyftLevel 5: https://level5.lyft.com/dataset/?source=post_page

语义分割数据集：

1. http://semantic-kitti.org/

## 3D 点云检测

- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- PointNet++ (NeurIPS'2017)
- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
- PointPillars: Fast Encoders for Object Detection from Point Clouds
- Center-based 3D Object Detection and Tracking
- (2017) [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation]()
- (2017) [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)
- (2017) VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
- (2018) [Frustum PointNets for 3D Object Detection from RGB-D Data]()
- (2019) [PointPillars: Fast Encoders for Object Detection from Point Clouds]()
- (2020) [SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving]()
- (2021) [PointAugmenting: Cross-Modal Augmentation for 3D Object Detection]()
- (2021) CenterPoint: Center-based 3D Object Detection and Tracking


## BEV 感知

以2D视角的多张图片为输入，转换到BEV空间，再输出检测和跟踪结果的一类方法

检测方法：

1. Disentangling Monocular 3D Object Detection
2. FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection
3. Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving
4. DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries
5. BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers

带物体跟踪的方法：

- Mono-Camera 3D Multi-Object Tracking Using Deep Learning Detections and PMBM Filtering
- DEFT: Detection Embeddings for Tracking
- QD3DT: Monocular Quasi-Dense 3D Object Tracking
- MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries
- 3D Multi-Object Tracking: A Baseline and New Evaluation Metrics
- SimpleTrack: Understanding and Rethinking 3D Multi-object Tracking
- CenterTrack: Tracking Objects as Points
- CenterTrack: Center-based 3D Object Detection and Tracking
- MOTR: End-to-End Multiple-Object Tracking with TRansformer

- (2020) LSS
- (2020) DETR
- (2021) deformer DETR
- (2021) DETR3D
- (2022) PETR
- (2022) PETRv2
- (2022) BEVFormer
- (2022) BEVSegFormer
- (2022) BEVerse
- (2022) BEVDet
- (2021) BEVFusion
- (2021) FUTR3D

## 地图构建

- (2021) HDMapNet: An Online HD Map Construction and Evaluation Framework
- (2022) VectorMapNet: End-to-end Vectorized HD Map Learning
- Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection