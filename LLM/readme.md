# 语言模型

## 早期阶段

- [2013] Word2Vec : Efficient Estimation of Word Representations in Vector Space
- [2017] transformer : Attention is All your need
- [2018] ELMo: Deep contextualized word representations
- [2018] GPT: Language Models are Unsupervised Multitask Learners
- [2018] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 大模型(纯语言模型)

- (2023.02.27) LLaMA: Open and Efficient Foundation Language Models
- (2023.09.28) Qwen Technical Report
- (2024.02.01) OLMo: Open Language Model

## 多模态大模型

### 综述

- (2023.06.23) A Survey on Multimodal Large Language Models

### 图像和语言

- (2021.02.26) CLIP: Learning Transferable Visual Models From Natural Language Supervision
- (2022.01.28) BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
- (2022.12.14) openCLIP: Reproducible scaling laws for contrastive language-image learning
- (2023.01.30) BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- (2023.04.17) LLaVA: Large Language and Vision Assistant
- (2023.04.20) MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- (2023.06.12) Valley: Video Assistant with Large Language model Enhanced abilitY

#### CLIP

用互联网规模的图文对预训练一个大的模型，实现给出图像，找到最接近的匹配的文本，以此为基础，应用到各种图像分类任务中都超过了
为此专门训练的模型的性能

它训练的能力是图像和文本的匹配能力，模型可以判断哪个文本和图像最匹配。

#### BLIP

研究机构： Salesforce Research

github: https://github.com/salesforce/LAVIS/tree/main

##### (2022.01.28) BLIP

与CLIP类似，从互联网规模的图文对训练一个大的模型，实现图像任务，不同的是更复杂，有三个优化任务
第一个优化目标：给出图像和若个描述文本，选择一个相似的
第二个优化目标：给出图像和一个描述文本，二分类是否匹配
第三个优化目标：给出图像，生成图像的描述文本，和真值取交叉熵损失

训练出来的模型不仅能胜任 图像分类的任务，还能胜任给出图像，生成描述文本这种生成式任务。

所以BLIP比CLIP多一个能力，就是根据给出的图像生成文本描述

##### (2023.01.30) BLIP2

冻结大语言模型和视觉模型参数，中间加入Q-transformer, 桥接视觉特征到语言模型特征空间，实验多模态大模型

##### (2023.05.11) InstructBLIP

InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

##### (2023.05.24) BLIP-Diffusion

BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing

文本生成图像领域的相关工作，可以根据输入的文本生成图像

##### (2024.09.09) X-InstructBLIP

X-InstructBLIP: A Framework for Aligning Image, 3D, Audio, Video to LLMs and its Emergent Cross-modal Reasoning

类似 GPT4o 的 多模态大模型， 可以处理多个模态，不只是图像和文本

#### LLaVA

背景：刘浩天（Liu Haotian）是一位在人工智能和自然语言处理领域有显著贡献的研究者。他目前是斯坦福大学的博士生，专注于多模态学习、视觉-语言预训练模型以及大规模语言模型的研究。他在多个顶级学术会议和期刊上发表了多篇论文，并且参与了多个知名开源项目的开发。

github: https://github.com/haotian-liu/LLaVA

##### (2023.04.17) LLaVa

使用大语言模型生成图片-文本指令微调数据，训练多模态大模型，最终结果是可以输入语言+图片，模型给出问题的回答

##### (2023.06.01) LLaVA-Med

LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day

生物医学方面的大模型问答

##### (2023.09.26 )LLaVA-RLHF

LLaVA-RLHF: Aligning Large Multimodal Models with Factually Augmented RLHF

使用人类反馈强化学习之后强化的 LLaVa 模型

##### (2023.10.15) LLaVA-1.5

##### (2023.11.10) LLaVA-Plus

##### (2024.05.10) LLaVA-NeXT

https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/

LLaVA-OneVision: Easy Visual Task Transfer

#### MiniGPT-4

背景：Vision CAIR Research Group 是 KAUST 的一个重要研究小组，专注于计算机视觉、自然语言处理、多模态学习和机器人技术的研究。

沙特阿拉伯阿卜杜拉国王科技大学（King Abdullah University of Science and Technology, KAUST）

##### (2023.04.20) MiniGPT-4

实现的也是图像问答，不包含视频，方法类似，使用的是QFormer

##### (2023.10.14) MiniGPT-v2

MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-task Learning

##### MiniGPT4-Video

MiniGPT4-Video: Advancing Multimodal LLMs for Video Understanding with Interleaved Visual-Textual Tokens

Goldfish: Vision-Language Understanding of Arbitrarily Long Videos

#### Valley

能力是可以进行视频问答，在处理图像的基础上增加了处理视频的能力

## 世界模型

- (2024.10.28) Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability
- Emu3: Next-Token Prediction is All You Need
- Segment Anything
- SAM 2: Segment Anything in Images and Videos
- Self-Supervised Vision Transformers with DINO
- DINOv2: Learning Robust Visual Features without Supervision

## 模型微调

降低模型微调过程中灾难性遗忘的方法

- 微调过程中加入通用的指令数据(主流方法)
- Lora
- (2024.01.04) LLaMA Pro: Progressive LLaMA with Block Expansion
- (2024.02.21) Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning

## 序列学习

- (2023.10.30) STR: Large Trajectory Models are Scalable Motion Predictors and Planners
- (2023.12.01) LVM: Sequential Modeling Enables Scalable Learning for Large Vision Models
- (2024.09.27) Emu3: Next-Token Prediction is All You Need
