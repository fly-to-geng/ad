# prediction

## 综述

- Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review

## 数据集

- (2019) Argoverse: 3D Tracking and Forecasting With Rich Maps
- (2021) Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset

## 传统方法

- CoverNet: Multimodal Behavior Prediction using Trajectory Sets
- MultiPath: Multiple Probabilistic Anchor TrajectoryHypotheses for Behavior Prediction
- TNT:
- VectorNet: Encoding HD Maps and Agent Dynamics fromVectorized Representation
- LaneGCN: Learning Lane Graph Representations for Motion Forecasting
- LaneRCNN: Distributed Representations for Graph-Centric Motion Forecasting
- scene transformer
- mmTransformer: Multimodal_Motion_Prediction_with_Stacked_Transformers
- Trajectory Prediction with Linguistic Representations
- TPCN: Temporal Point Cloud Networks for Motion Forecasting
- HOME: Heatmap Output for future Motion Estimation
- THOMAS: Trajectory Heatmap Output with learned Multi-Agent Sampling
- GOHOME: Graph-Oriented Heatmap Output for future Motion Estimation
- Wayformer: Motion Forecasting via Simple & Efficient Attention Networks
- DCMS: Motion Forecasting with Dual Consistency and Multi-Pseudo-Target Supervision
- Path-Aware Graph Attention for HD Maps in Motion Prediction
- (2019) Rules of the Road:Predicting Driving Behavior with a Convolutional Model of Semantic Interactions # 小火苗
- (2022) HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction
- (2021) mmTransformer: Multimodal Motion Prediction with Stacked Transformers
- (2021) Scene Transformer: A unified architecture for predicting multiple agent trajectories
- (2021) TPCN: Temporal Point Cloud Networks for Motion Forecasting # 表示成点云信息，用点云算法处理
- (2021) MultiModalTransformer: Multi-modal Motion Prediction with Transformer-based Neural Network for Autonomous Driving
- (2022) DCMS: Motion Forecasting with Dual Consistency and Multi-Pseudo-Target Supervision
- (2022) Wayformer: Motion Forecasting via Simple & Efficient Attention Networks
- (2022) Trajectory Prediction with Linguistic Representations # 涉及自然语言处理，思路新颖
- (2022) VectorFlow: Combining Images and Vectors for Traffic Occupancy and Flow Prediction # 基于流的预测，特斯拉2022
- (2022) Importance is in your attention: agent importance prediction for autonomous driving #
- (2022) ScePT: Scene-consistent, Policy-based Trajectory Predictions for Planning
- (2022) Vehicle trajectory prediction works, but not everywhere # 地图数据增强方法
- (2023) QC-net: Query-Centric Trajectory Prediction
- (2024) FutureNet-LOF: Joint Trajectory Prediction and Lane Occupancy Field Prediction with Future Context Encoding

### 场景一致性预测

(2022) M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction
(2022) THOMAS: Trajectory Heatmap Output with learned Multi-Agent SamplingDownload PDF

### 端到端预测

(2021) FIERY: Future Instance Prediction in Bird's-Eye View From Surround Monocular Cameras
(2022) ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries
(2022) BEVerse : Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving
IntentNet: Learning to Predict Intention from Raw Sensor Data
Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net
PnPNet: End-to-End Perception and Prediction with Tracking in the Loop
ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries
Deep Multi-Task Learning for Joint Localization, Perception, and Prediction
FIERY: Future Instance Prediction in Bird's-Eye View from Surround Monocular Cameras
ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries
End-to-end Prediction of Driver Intention using 3D Convolutional Neural Networks
BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving

### 行人轨迹预测

Goal-driven Long-Term Trajectory Prediction

(2016) Social LSTM:Human Trajectory Prediction in Crowded Spaces
(2018) Social Attention: Modeling Attention in Human Crowds
(2018) Social GAN: Socially Acceptable Trajectorieswith Generative Adversarial Networks
(2018) SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints
(2019) IE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction
(2019) SR-LSTM: State Refinement for LSTM towards Pedestrian Trajectory Prediction
(2020) Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data
(2022) Distilling Knowledge for Short-to-Long Term Trajectory Prediction
(2016) Social LSTM: Human Trajectory Prediction in Crowded Spaces
(2020) Goal-driven Long-Term Trajectory Prediction

### AVP预测

ParkPredict+: Multimodal Intent and Motion Prediction for Vehicles in Parking Lots with CNN and Transformer

## 大模型

- Traj-LLM: A New Exploration for Empowering Trajectory Prediction with Pre-trained Large Language Models
- LARGE TRAJECTORY MODELS ARE SCALABLE MOTION PREDICTORS AND PLANNERS

## 轨迹的运动学优化

Deep Kinematic Models for Kinematically Feasible Vehicle Trajectory Predictions

## 预测范式

1. TPCN (CVPR2021) 基于点云的预测
2. RTM3D (ECCV2020) 引入关键点之间的几何约束
3. M2I: From Factored Marginal Trajectory to Interactive Prediction
4. Thoms(ICLR 2022) 场景一致性预测
5. Vehicle trajectory prediction, but not everywhere. 数据增强，通过对地图旋转拉伸变化做数据增强