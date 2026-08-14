# 🛠️ Build Your Own AI Stack

> Master AI by building it from scratch — tokenizer to agent.
> 从零手搓一整套 AI 技术栈：Tokenizer → Transformer → 训练 → 推理 → RAG → Agent。

![stars](https://img.shields.io/github/stars/XiaoGuan-AI/build-your-own-ai-stack)
![license](https://img.shields.io/github/license/XiaoGuan-AI/build-your-own-ai-stack)

**Every chapter is a working artifact.** No black boxes. No frameworks hiding the magic.
**每一章都是能跑的真代码。** 不调 LangChain，不装黑盒，每一步都从零手写。

---

## English

### Why this project?

`build-your-own-x` taught a generation of developers by making them recreate the tools they use every day. AI is the biggest stack most developers now depend on — yet almost nobody builds it themselves. Tutorials stop at the model (`LLMs-from-scratch`), or start at the agent (`ai-agents-from-scratch`). Nobody walks the whole road.

**This repo is the whole road**: you will hand-craft a complete AI application stack, and the final project is a working mini AI assistant that can chat *and* do real work.

### Roadmap (10 chapters)

| # | Chapter | What you build | Status |
|---|---------|----------------|--------|
| 0 | Setup & Minimal Math | Dev environment, tensors, gradients — the 5% of math you actually need | ✅ |
| 1 | Tokenizer | A real BPE tokenizer from scratch (like GPT-2's) | ✅ |
| 2 | Transformer | Multi-head attention, QKV, residual blocks, a forward pass you can trace | ⏳ |
| 3 | Pretraining | Train a 64M-param LLM on your own GPU from raw text | ⏳ |
| 4 | Inference Engine | KV cache, sampling strategies, a tiny serving loop | ⏳ |
| 5 | RAG | Embedding, retrieval, and grounded generation with your own docs | ⏳ |
| 6 | Agent Loop | ReAct, tool calling, planning — the brain of an agent | ⏳ |
| 7 | MCP & Tools | Model Context Protocol, file & terminal tools, safety rails | ⏳ |
| 8 | Capstone | Build a mini AI assistant that chats *and* executes tasks | ⏳ |
| 9 | Ship It | Packaging, performance, and the path beyond | ⏳ |

### How each chapter works

```
┌─────────────┐   ┌─────────────────────┐   ┌──────────┐   ┌────────────┐
│  Why it     │ → │  Minimal runnable   │ → │ Hands-on │ → │ Checkpoint │
│  works      │   │  code (< 300 lines) │   │  labs    │   │  quiz      │
│  (intuition)│   │  with comments      │   │          │   │  + project │
└─────────────┘   └─────────────────────┘   └──────────┘   └────────────┘
```

No GPU? Chapter 0–2, 4–9 run on CPU. Only Chapter 3 (pretraining) needs a GPU — a 64M model trains in ~2 hours on one 4090.

### Quick start

```bash
git clone https://github.com/XiaoGuan-AI/build-your-own-ai-stack.git
cd build-your-own-ai-stack
# Chapter 1 — run the tokenizer you built:
python code/ch01/bpe.py --demo
```

Full instructions in [docs/chapter-00-setup-and-math.md](docs/chapter-00-setup-and-math.md).

### Contributing

PRs welcome — especially: bug reports, better Chinese/English explanations, translated chapters, and lab exercises. See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).

### License

MIT © XiaoGuan-AI

---

## 中文

### 为什么做这个项目

`build-your-own-x` 让一代开发者靠「手搓自己每天都在用的技术」真正学会编程。今天 AI 是几乎所有开发者最依赖的技术栈——但几乎没有人亲手把它造出来。现有的教程要么止步于模型（如 rasbt/LLMs-from-scratch），要么从 Agent 半路开始（如 ai-agents-from-scratch），**没有人带你把整条路走完**。

**这个仓库就是整条路**：你将从零手搓一整套 AI 应用栈，最终大作业是一个真正能跑、能聊天、还能干活的迷你 AI 助手。

### 路线图（十章）

| # | 章节 | 你将手搓出什么 | 状态 |
|---|------|--------------|------|
| 0 | 环境与最小数学 | 开发环境、张量、梯度——你真正需要的 5% 数学 | ✅ |
| 1 | Tokenizer | 一个真实的 BPE 分词器（对标 GPT-2 的 tiktoken） | ✅ |
| 2 | Transformer | 多头注意力、QKV、残差块，一个能逐行追踪的前向传播 | ⏳ |
| 3 | 预训练 | 在你自己 GPU 上从原始文本训出一个 64M 参数的 LLM | ⏳ |
| 4 | 推理引擎 | KV Cache、采样策略、极简服务循环 | ⏳ |
| 5 | RAG | 向量化、检索、基于你自己文档的接地生成 | ⏳ |
| 6 | Agent 循环 | ReAct、工具调用、规划——Agent 的大脑 | ⏳ |
| 7 | MCP 与工具 | Model Context Protocol、文件/终端工具、安全护栏 | ⏳ |
| 8 | 大作业 | 手搓一个能聊天**还能干活**的迷你 AI 助手 | ⏳ |
| 9 | 上线 | 打包发布、性能优化，以及更远的路 | ⏳ |

### 每章结构

```
┌─────────────┐   ┌─────────────────────┐   ┌──────────┐   ┌────────────┐
│ 原理直觉    │ → │ 最小可运行代码      │ → │ 动手实验 │ → │ 检验点     │
│ （大白话）  │   │ （<300 行，全注释）  │   │          │   │ 练习+小项目 │
└─────────────┘   └─────────────────────┘   └──────────┘   └────────────┘
```

没有 GPU 也能学：第 0–2、4–9 章 CPU 就能跑，只有第 3 章（预训练）需要 GPU——64M 模型在一张 4090 上约 2 小时训完。

### 快速开始

```bash
git clone https://github.com/XiaoGuan-AI/build-your-own-ai-stack.git
cd build-your-own-ai-stack
# 第 1 章——跑你亲手写的分词器：
python code/ch01/bpe.py --demo
```

完整环境搭建见 [docs/chapter-00-setup-and-math.md](docs/chapter-00-setup-and-math.md)。

### 贡献

欢迎 PR——尤其是：提 bug、改进中英文讲解、翻译章节、补充练习。详见 CONTRIBUTING.md（即将上线）。

### License

MIT © XiaoGuan-AI
