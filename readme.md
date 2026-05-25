# 🧠 RAG-demo：从零搭建 RAG 知识库问答系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://www.langchain.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://platform.deepseek.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

> 使用 **LangChain + DeepSeek + FAISS** 搭建的 RAG（检索增强生成）知识库问答系统。支持 TXT / PDF 文档，适合 AI 初学者从零理解 RAG 架构思想并动手实践。
>
> **关键词**：RAG、检索增强生成、LangChain、DeepSeek、FAISS、向量检索、知识库问答、中文NLP、PDF问答、LLM应用开发

---

## 什么是 RAG？为什么需要它？

大语言模型（LLM）虽然强大，但存在两个根本问题：

1. **知识过时**：模型训练数据有截止日期，无法回答最新的问题
2. **幻觉问题**：当模型不知道答案时，会自信地"编造"一个看似合理的回答

**RAG（Retrieval-Augmented Generation，检索增强生成）** 的解决思路很直接：

> 不要让模型凭记忆回答，而是**先从你的文档中检索出相关内容**，把这些内容作为参考资料塞给模型，让它**基于真实资料来回答**。

这样，模型的回答就有据可依，大幅减少了幻觉，同时能回答你私有文档中的专业问题。

本项目提供了两个循序渐进的 Demo，帮你从零理解并实现一个完整的 RAG 问答系统。

---

## 系统架构全景图

本系统的数据流分为两个阶段：

```
┌─────────────────── 离线索引阶段（只跑一次） ───────────────────┐
│                                                                 │
│  知识库文档         文本提取          文本分块          向量化    │
│  (TXT/PDF)  ──→  (含页码元数据) ──→ (chunk) ──→  嵌入模型 ──→  │
│                                                         ↓       │
│                                              FAISS 向量库(本地)  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────── 在线问答阶段（每次提问） ───────────────────┐
│                                                                 │
│  用户提问 ──→ 嵌入模型 ──→ FAISS 相似度检索 ──→ Top-K 文本块   │
│                                                      ↓          │
│                              DeepSeek LLM ←── 问题 + 参考文本   │
│                                   ↓                              │
│                            生成答案 + 来源页码                    │
└─────────────────────────────────────────────────────────────────┘
```

**为什么分两个阶段？** 离线索引只需执行一次，向量库会持久化到本地磁盘。每次提问只需加载现有向量库做检索，不需要重复处理文档。

---

## 核心技术栈

| 组件            | 选型       | 角色                                                                    |
| --------------- | ---------- | ----------------------------------------------------------------------- |
| **LangChain**   | 应用框架   | 流程编排，把文档加载、分块、检索、问答串成一条链                        |
| **DeepSeek**    | 大语言模型 | 理解问题 + 参考文本，生成自然语言答案                                   |
| **FAISS**       | 向量数据库 | 高效存储和检索向量，毫秒级返回 Top-K 相似结果                           |
| **HuggingFace** | 嵌入模型   | 本地运行的中文向量模型（`text2vec-base-chinese`），**免费无需 API Key** |
| **PyMuPDF**     | PDF 解析   | 提取 PDF 文本并自动记录每页页码                                         |

---

## 核心模块解读

下面拆解每个环节，帮你理解**为什么这么设计**。

### 📄 文档加载 — 把文件变成程序能处理的文本

| 场景       | 加载器          | 说明                                                 |
| ---------- | --------------- | ---------------------------------------------------- |
| TXT 纯文本 | `TextLoader`    | 最简单的加载方式，直接读取文本内容                   |
| PDF 文档   | `PyMuPDFLoader` | 基于 PyMuPDF 解析，**自动记录每页的页码到 metadata** |

> 💡 **为什么 PDF 要记录页码？** 当系统给出答案时，你可以追溯到"这个答案来自第几页"，方便核实准确性。

### ✂️ 文本分块 — 把长文档切成合适大小的片段

LLM 的上下文窗口有限，不可能把整篇文档都塞进去。所以需要先切成小块（chunk），检索时只取最相关的几块。

| 分块器                           | 用于     | 策略                                              |
| -------------------------------- | -------- | ------------------------------------------------- |
| `CharacterTextSplitter`          | TXT Demo | 按固定字符数切分，简单直接                        |
| `RecursiveCharacterTextSplitter` | PDF Demo | 按 `\n\n` → `\n` → `。` → `，` 递归寻找最佳切分点 |

**关键参数**：

- **chunk_size**：每块的目标大小。太大 → 检索不精准且浪费 Token；太小 → 上下文不完整
- **chunk_overlap**：相邻块的重叠字符数。防止关键句子刚好被切在边界上

> 💡 **怎么调参？** 本项目 TXT Demo 用 chunk_size=100（文本很短），PDF Demo 用 chunk_size=500 + overlap=50（长文档需要更多上下文）。实际项目中需要根据文档类型和问答效果反复调试。

### 🔢 向量化 — 把文字变成计算机能比较的"坐标"

人类理解语义，计算机理解数字。**嵌入模型（Embedding Model）** 把一段文字映射为一个高维向量（如 768 维的浮点数数组），语义相近的文本在向量空间中距离更近。

本项目使用 `shibing624/text2vec-base-chinese`：

- ✅ **开源免费**的中文嵌入模型
- ✅ 运行在**本地**，无需联网调用 API
- ✅ 无需 API Key，**零成本**

> 💡 **为什么不用付费的嵌入 API？** 每次构建索引和每次提问都需要调用嵌入模型，使用本地模型可以完全避免这部分费用。

### 🗄️ 向量数据库 FAISS — 高效存储和检索向量

**FAISS**（Facebook AI Similarity Search）是 Meta 开源的向量检索引擎，核心能力是给定一个查询向量，从百万级向量库中**毫秒级**找出最相似的 Top-K 个结果。

本项目中 FAISS 的使用方式：

- **构建**：`FAISS.from_documents(docs, embeddings)` — 一行代码完成向量化 + 入库
- **持久化**：`db.save_local("faiss_index_pdf")` — 保存到本地目录，下次直接加载
- **检索**：`db.similarity_search_with_score(query, k=3)` — 返回最相关的 3 个文本块及相似度分数

### 🤖 大语言模型 — 理解问题并生成答案

检索到相关文本块后，系统会把它们和用户的问题一起交给 **DeepSeek LLM**，由模型基于这些参考资料生成自然语言答案。

整个过程由 LangChain 的 `RetrievalQA` 链自动编排：

```
qa.invoke("你的问题")  ← 你只需要这一行
   ↓
[自动] 将问题向量化 → 检索 Top-K 文本块 → 拼接 Prompt → 调用 LLM → 返回答案
```

---

## Demo 学习路径

建议按以下顺序学习，从简单到复杂：

### 📘 Demo 1：TXT 文本问答（`txt_rag_demo.ipynb`）

**学习目标**：理解 RAG 的最小完整流程

- 使用一段简短的 AI 概念文本作为知识库
- 涵盖：加载 → 分块 → 向量化 → 存储 → 问答的完整链路
- 代码简洁，适合初学者跑通第一个 RAG 应用

### 📕 Demo 2：PDF 文档问答（`pdf_rag_demo.ipynb`）

**学习目标**：掌握真实文档场景下的 RAG 实践

以「低空无人机集群通信.pdf」为知识库，在 Demo 1 的基础上新增：

| 步骤  | 新增能力       | 学习要点                                         |
| ----- | -------------- | ------------------------------------------------ |
| Step1 | 知识库管理     | 可配置的 PDF 路径，替换即可切换知识库            |
| Step2 | 页码元数据     | 提取文本同时记录每页页码，为答案溯源打基础       |
| Step3 | 高级分块       | `RecursiveCharacterTextSplitter` + 语义边界切分  |
| Step4 | 独立检索调试   | 先执行相似度搜索查看检索效果（**不消耗 Token**） |
| Step5 | 带来源的问答链 | `return_source_documents=True` 返回引用的源文档  |
| Step6 | 答案溯源       | 展示每个引用文档块的来源文件、页码和内容         |

---

## 项目目录结构

```
RAG-demo/
├── .env                    # 环境配置（API Key 等，不要提交到 Git）
├── .gitignore              # Git 忽略规则
├── readme.md               # 本文档
├── txt_rag_demo.ipynb      # Demo 1：TXT 文本问答
├── pdf_rag_demo.ipynb      # Demo 2：PDF 文档问答
├── test_doc.txt            # Demo 1 的示例文本（自动生成）
├── 低空无人机集群通信.pdf    # Demo 2 的示例 PDF
├── faiss_index/            # Demo 1 的向量库（自动生成）
├── faiss_index_pdf/        # Demo 2 的向量库（自动生成）
└── venv/                   # Python 虚拟环境
```

---

## 设计亮点

### 🆓 零成本嵌入

使用 HuggingFace 本地模型做向量化，嵌入过程完全在本地运行，**不调用任何付费 API**。只有最后一步调用 DeepSeek LLM 生成答案时才消耗 Token。

### 🔍 检索与生成分离

Step4（相似度搜索）可以独立运行，不调用 LLM、不消耗 Token。这意味着你可以：

- 反复调试检索效果（调整 chunk_size、Top-K 等参数）
- 确认检索结果满意后，再调用 LLM 生成答案
- **省钱 + 高效调试**

### 📖 答案可溯源

每个回答都附带引用文档块的**来源文件名和页码**，快速翻到原文验证，这是企业级 RAG 系统的必备能力。

### 💾 向量库持久化

FAISS 索引保存到本地磁盘，下次使用直接加载，**不需要重新处理文档和计算向量**。

### 🔒 配置与代码分离

所有敏感信息（API Key、镜像地址等）统一通过 `.env` 文件管理，代码中不硬编码任何密钥。既安全又方便切换环境。

---

## 环境搭建

### 1. 确认 Python 环境

请在终端里逐一执行以下命令，确认你的电脑已有 Python 3.9+：

```bash
python3 --version
pip3 --version
which python3
```

### 2. 创建虚拟环境

强烈建议给项目创建独立的虚拟环境，避免全局包版本冲突：

```bash
# 创建虚拟环境（只需一次）
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
# 激活成功后，终端左边会出现 (venv) 提示符

# 不用时退出虚拟环境
deactivate
```

### 3. 获取 DeepSeek API 密钥

1. **注册并登录**：打开浏览器访问 [platform.deepseek.com](https://platform.deepseek.com)，使用手机号或邮箱注册并登录
2. **创建密钥**：在左侧菜单找到 "API Keys"，点击 "创建 API key"，随便起个名字（如 "MyFirstRAG"）
3. **保存密钥**：系统会生成一串 `sk-xxxxxx` 字符，**务必立即复制保存**，关闭页面后将无法再查看

### 4. 配置环境变量

在项目根目录创建 `.env` 文件，写入：

```
DEEPSEEK_API_KEY=sk-你的密钥
HF_ENDPOINT=https://hf-mirror.com
```

> ⚠️ `.env` 文件包含敏感信息，已在 `.gitignore` 中排除，不会提交到 Git。

### 5. 安装依赖

```bash
python3 -m pip install langchain langchain-community langchainhub faiss-cpu sentence-transformers pymupdf
python3 -m pip install python-dotenv
python3 -m pip install langchain-deepseek
```

安装完成后看到 `Successfully installed ...` 就说明一切就绪了 ✅

---

## 常见问题

### ❌ OSError: We couldn't connect to 'https://huggingface.co'

**原因**：国内网络对 huggingface.co 不稳定，无法下载嵌入模型

**解决**：在 `.env` 文件中添加国内镜像地址：

```
HF_ENDPOINT=https://hf-mirror.com
```

### ❌ APIStatusError: Error code: 402

**原因**：DeepSeek API Key 可用但余额不足

**解决**：去 [DeepSeek 控制台](https://platform.deepseek.com) 给当前账号充值，或更换有余额的 Key

---

## 进阶方向

学完两个 Demo 后，你可以尝试：

- **支持更多格式**：Word、Markdown、HTML 等文档的加载与解析
- **优化检索精度**：尝试不同的 chunk_size、overlap、Top-K 参数组合
- **语义缓存**：对相似问题命中缓存，避免重复调用 LLM
- **多文档知识库**：同时加载多个文档，构建综合知识库
- **Web 界面**：用 Streamlit 或 Gradio 给系统加一个交互式前端

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！如果这个项目对你有帮助，请给一个 ⭐ Star，这是对我最大的鼓励。

## 📄 License

本项目基于 [MIT License](./LICENSE) 开源。
