# 项目介绍

LangChain+DeepSeek+Faiss搭建 RAG 知识库问答服务

## 流程

step1，收集整理知识库（客户经理考核办法.pdf只是示例，用你的PDF进行检索）
Step2，从PDF中提取文本并记录每行文本对应的页码
Step3，处理文本并创建向量存储
step4，执行相似度搜索，找到与查询相关的文档
Step5，使用问到链对用户问题进行回答（使用你的DASHSCOPE_APL_KEY）
step6，显示每个文档块的来源页码（当前页码来源有问题，可以用Cursor完善）

1、使用 TextLoader 来读取 .txt 文件内容，把其中的文字提取出来，变成 LangChain 能处理的 Document 对象（包含文本内容和元数据）

2、使用 CharacterTextSplitter 把长文本切分成小段（chunk），

    具体就是根据指定的字符数（chunk_size）把长文档切开，每段之间还可以有重叠（chunk_overlap），防止关键信息被截断。

3、使用 HuggingFaceEmbeddings 把文本转换成向量（Embedding）

    具体就是使用 Hugging Face 社区提供的开源模型（如 shibing624/text2vec-base-chinese），把中文句子变成一串 384 维或 768 维的浮点数向量。

4、使用 FAISS 作为向量数据库，用来存储第 3 步生成的向量，在提问时快速检索出相关的文本块。

    具体就是 FAISS 是 Facebook 开源的向量相似度搜索库，能高效计算余弦距离或欧氏距离，返回 Top-K 最相似的文档。

5、使用 ChatDeepSeek 调用 DeepSeek 大语言模型（LLM）然后根据 用户输入的问题 + 检索到的相关文本块 生成回答，

    具体就是 ChatDeepSeek 是 LangChain 官方为 DeepSeek 模型定制的包装类，方便你通过 API Key 调用 DeepSeek 的 deepseek-chat 模型（或 deepseek-coder）

6、使用 RetrievalQA 将上面的过程串成一条链（Chain），你只需要一行 qa.invoke("问题")，背后所有步骤都被这个链管理了

# 环境准备

## 先确认一下现状（很关键）

这个步骤不是绕弯子，而是为了搞清楚你的电脑到底是什么状况，避免后面走弯路。请在终端（Terminal）里逐一输入下面这几条命令，每输入一条就按一下回车：

```bash
python3 --version
pip3 --version
which python3
```

## 创建一个独立的虚拟环境

直接把包装在电脑全局环境里，可能会带来以后的版本冲突。所以强烈建议，还是给你的项目创建一个独立的虚拟环境

```bash
# 1. 创建一个叫 'venv' 的虚拟环境 (只用做一次)
python3 -m venv venv

# 2. 激活这个虚拟环境
source venv/bin/activate
# 激活成功后，你会看到终端左边出现了 (venv) 的提示符
```

## 💎 总结 & 验证

当你看到 Successfully installed ... 的提示时，就说明一切都大功告成了。之后你会看到终端的命令提示符前面多了一个 (venv)，这就表示你正处在安全的虚拟环境中。

等你不再需要这个环境时，用下面这个命令就能随时退出：

```bash
deactivate
```

## 获取 API 密钥

这是一个免费的步骤，但我们依然需要按流程操作。

注册并登录：打开浏览器，访问 platform.deepseek.com。使用你的手机号或邮箱完成账号注册并登录。

创建密钥：登录后，在左侧菜单栏找到并点击 "API Keys" 。点击 "创建 API key" 按钮（或 "Create API key"），为你的密钥随便起个名字（例如 "MyFirstRAG"）。

保存密钥：点击创建后，系统会生成一串类似 sk-xxxxxx 的字符。请你务必立即把那串字符复制并保存到一个记事本文件里，因为关闭页面后，你就再也看不到它了。

## 安装所有必备组件（依赖库）

所有组件（如 LangChain、FAISS 等）都是别人已经写好的工具包，我们直接用 Python 的包管理工具 pip 一键安装即可。

```bash
python3 -m pip install langchain langchain-community langchainhub faiss-cpu sentence-transformers pymupdf

python3 -m pip install python-dotenv

python3 -m pip install langchain-deepseek
```

# 遇到问题

## OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files

含义是：连不上官网，本地缓存里也没有完整模型，所以初始化嵌入模型失败

原因是：国内网络对 huggingface.co 不稳定、公司代理/防火墙拦截、或需要科学上网才能稳定访问

解决方案：使用国内镜像（常用）

在 创建 HuggingFaceEmbeddings 之前 设置环境变量（你已有 load_dotenv()，可写在 .env 里）：

`HF_ENDPOINT=https://hf-mirror.com`

## APIStatusError: Error code: 402

含义是：API Key 可用，但余额不足，所以 qa.invoke(query) 调用被拒绝

解决方案：去 DeepSeek 控制台给当前账号充值，或换一个有余额的 Key
