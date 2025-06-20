# 大语言模型学习笔记


## 前言
- 阅读建议与知识预备
- 贡献者
- 文档架构说明

## 目录
- [大语言模型学习笔记](#大语言模型学习笔记)
  - [前言](#前言)
  - [目录](#目录)
  - [1. 大语言模型基础概念](#1-大语言模型基础概念)
    - [1.1 什么是大语言模型？](#11-什么是大语言模型)
    - [1.2 Transformer回顾](#12-transformer回顾)
    - [1.3 模型类型：Encoder-Only、Decoder-Only、Encoder-Decoder](#13-模型类型encoder-onlydecoder-onlyencoder-decoder)
    - [1.4 非Transformer架构的LLM](#14-非transformer架构的llm)
    - [1.5 参考文献与相关内容](#15-参考文献与相关内容)
  - [2. 大语言模型底层架构优化](#2-大语言模型底层架构优化)
    - [2.1 注意力机制及其变体](#21-注意力机制及其变体)
    - [2.2 位置编码与长上下文支持](#22-位置编码与长上下文支持)
    - [2.3 混合专家模型（MoE）](#23-混合专家模型moe)
    - [其他架构创新](#其他架构创新)
    - [参考文献与相关内容](#参考文献与相关内容)
  - [3. 模型预训练](#3-模型预训练)
    - [3.1 数据准备](#31-数据准备)
    - [3.2 预训练任务设计](#32-预训练任务设计)
    - [3.3 分布式训练与优化](#33-分布式训练与优化)
    - [3.4 新的预训练范式](#34-新的预训练范式)
    - [3.5 参考文献与相关内容](#35-参考文献与相关内容)
  - [4. 模型微调与知识蒸馏](#4-模型微调与知识蒸馏)
    - [4.1 监督微调（Supervised Fine-tuning, SFT）](#41-监督微调supervised-fine-tuning-sft)
    - [4.2 知识蒸馏（Knowledge Distillation）](#42-知识蒸馏knowledge-distillation)
    - [4.3 参考文献与相关内容](#43-参考文献与相关内容)
  - [5. 强化学习与深度思考能力](#5-强化学习与深度思考能力)
    - [5.1 人类反馈强化学习（RLHF）](#51-人类反馈强化学习rlhf)
    - [5.2 深度思考模型](#52-深度思考模型)
    - [5.3 群体相对策略优化（GRPO）](#53-群体相对策略优化grpo)
    - [5.4 过程奖励模型（PRM）与蒙特卡洛树搜索（MCTS）](#54-过程奖励模型prm与蒙特卡洛树搜索mcts)
    - [5.5 DAPO与VAPO](#55-dapo与vapo)
    - [5.6 生成式奖励模型](#56-生成式奖励模型)
    - [5.7 参考文献与相关内容](#57-参考文献与相关内容)
  - [6. 小模型优化与测试时扩展](#6-小模型优化与测试时扩展)
    - [6.1 模型压缩技术](#61-模型压缩技术)
    - [6.2 测试时扩展TTS](#62-测试时扩展tts)
    - [6.3 参考文献与相关内容](#63-参考文献与相关内容)
  - [7. 工程实现与优化](#7-工程实现与优化)
    - [7.1 我需要多少显存？](#71-我需要多少显存)
    - [7.2 分布式训练框架](#72-分布式训练框架)
    - [7.3 推理加速工具](#73-推理加速工具)
    - [7.4 训练与推理工具链](#74-训练与推理工具链)
    - [7.5 超参数调整](#75-超参数调整)
    - [7.6 参考文献与相关内容](#76-参考文献与相关内容)
  - [8. 多模态](#8-多模态)
    - [8.1 多模态模型概述](#81-多模态模型概述)
    - [8.2 多模态混合生成](#82-多模态混合生成)
    - [8.3 参考文献与相关内容](#83-参考文献与相关内容)
  - [9. LLM Based Agent](#9-llm-based-agent)
    - [9.1 Agent 概述](#91-agent-概述)
    - [9.2 Manus与自主智能体](#92-manus与自主智能体)
    - [9.3 DeepResearch](#93-deepresearch)
    - [9.4 MCP协议](#94-mcp协议)
    - [9.5 参考文献与相关内容](#95-参考文献与相关内容)
  - [10. LLM工程前沿速览](#10-llm工程前沿速览)
    - [10.1 OpenAI](#101-openai)
    - [10.2 DeepSeek](#102-deepseek)
    - [10.3 Qwen](#103-qwen)
    - [10.4 Anthropic](#104-anthropic)
    - [10.5 Seed](#105-seed)
    - [10.6 Meta](#106-meta)
    - [10.7 ERNIE](#107-ernie)
    - [10.8 其他前沿模型](#108-其他前沿模型)
  - [附录](#附录)
    - [术语表（Glossary）](#术语表glossary)
    - [致谢与贡献指南（CONTRIBUTING.md）](#致谢与贡献指南contributingmd)

## 1. 大语言模型基础概念
### 1.1 什么是大语言模型？
### 1.2 Transformer回顾
### 1.3 模型类型：Encoder-Only、Decoder-Only、Encoder-Decoder
### 1.4 非Transformer架构的LLM
### 1.5 参考文献与相关内容

## 2. 大语言模型底层架构优化
### 2.1 注意力机制及其变体
    - Multi-head Attention (MHA)
    - Multi-Query Attention (MQA)
    - Grouped Query Attention (GQA)
    - Flash Attention 及其性能优势
    - 多头潜在注意力（MLA）
    - 原生稀疏注意力（NSA）
### 2.2 位置编码与长上下文支持
    - RoPE（旋转位置编码）
    - YaRN、ALiBi 等扩展方法
    - 超长上下文处理策略
### 2.3 混合专家模型（MoE）
    - MoE 基本原理
    - DeepSeek MoE 实现
    - 负载均衡策略
### 其他架构创新
    - 多Token预测（MTP）
### 参考文献与相关内容

## 3. 模型预训练
### 3.1 数据准备
    - 数据来源与清洗
    - Tokenizer 训练与词表构建
    - 上下文拼接策略
### 3.2 预训练任务设计
    - Causal Language Modeling（CLM）
    - Masked Language Modeling（MLM）
    - Prefix LM
    - Sequence-to-Sequence LM
### 3.3 分布式训练与优化
    - ZeRO（DeepSpeed）、Tensor Parallelism（Megatron-LM）
    - 混合并行策略
    - 梯度检查点（Gradient Checkpointing）
### 3.4 新的预训练范式
    - 强化学习预训练
### 3.5 参考文献与相关内容

## 4. 模型微调与知识蒸馏
### 4.1 监督微调（Supervised Fine-tuning, SFT）
    - 全参数微调
    - 参数高效微调PEFT
### 4.2 知识蒸馏（Knowledge Distillation）
    - 蒸馏的基本思想
    - 软标签 vs 硬标签
    - DeepSeek-R1 的蒸馏策略
### 4.3 参考文献与相关内容

## 5. 强化学习与深度思考能力
### 5.1 人类反馈强化学习（RLHF）
    - RL 基础知识
    - PPO（Proximal Policy Optimization）
    - DPO（Direct Preference Optimization）
    - RLOO、Reinforce++ 等变体
### 5.2 深度思考模型
    - 长思维链（Long Chain-of-Thought, CoT）概念
    - 长思维链的影响因素
    - 思考过度与思考不足
### 5.3 群体相对策略优化（GRPO）
### 5.4 过程奖励模型（PRM）与蒙特卡洛树搜索（MCTS）
### 5.5 DAPO与VAPO
### 5.6 生成式奖励模型
### 5.7 参考文献与相关内容

## 6. 小模型优化与测试时扩展
### 6.1 模型压缩技术
    - 量化
    - 剪枝
### 6.2 测试时扩展TTS
### 6.3 参考文献与相关内容

## 7. 工程实现与优化
### 7.1 我需要多少显存？
    - 模型部署
    - 模型微调
    - 模型预训练
### 7.2 分布式训练框架
    - DeepSpeed
    - Megatron-LM
### 7.3 推理加速工具
    - vLLM
    - GGUF（Llama.cpp）
### 7.4 训练与推理工具链
    - Transformers
    - Unsloth
    - VeRL
    - LlamaFactory
### 7.5 超参数调整
### 7.6 参考文献与相关内容

## 8. 多模态
### 8.1 多模态模型概述
    - 模型架构：CLIP等
    - 多模态预训练任务
### 8.2 多模态混合生成
### 8.3 参考文献与相关内容

## 9. LLM Based Agent
### 9.1 Agent 概述
### 9.2 Manus与自主智能体
### 9.3 DeepResearch
### 9.4 MCP协议
### 9.5 参考文献与相关内容

## 10. LLM工程前沿速览
### 10.1 OpenAI
### 10.2 DeepSeek
### 10.3 Qwen
### 10.4 Anthropic
### 10.5 Seed
### 10.6 Meta
### 10.7 ERNIE
### 10.8 其他前沿模型

## 附录
### 术语表（Glossary）
### 致谢与贡献指南（CONTRIBUTING.md）
