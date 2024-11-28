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

与CLIP类似，从互联网规模的图文对训练一个大的模型，实现图像任务，不同的是更复杂，有三个优化任务
第一个优化目标：给出图像和若个描述文本，选择一个相似的
第二个优化目标：给出图像和一个描述文本，二分类是否匹配
第三个优化目标：给出图像，生成图像的描述文本，和真值取交叉熵损失

训练出来的模型不仅能胜任 图像分类的任务，还能胜任给出图像，生成描述文本这种生成式任务。

所以BLIP比CLIP多一个能力，就是根据给出的图像生成文本描述

#### BLIP2

冻结大语言模型和视觉模型参数，中间加入Q-transformer, 桥接视觉特征到语言模型特征空间，实验多模态大模型

#### LLaVA

使用大语言模型生成图片-文本指令微调数据，训练多模态大模型，最终结果是可以输入语言+图片，模型给出问题的回答

### MiniGPT-4

实现的也是图像问答，不包含视频，方法类似，使用的是QFormer

#### Valley

能力是可以进行视频问答，在处理图像的基础上增加了处理视频的能力