<h1 align="center">ChatPDF</h1>
<div align="center">
  <a href="https://github.com/shibing624/ChatPDF">
  </a>

<p align="center">
    <h3>基于本地 LLM 做检索知识问答(RAG)</h3>
    <p align="center">
      <a href="https://github.com/shibing624/ChatPDF/blob/main/LICENSE">
        <img alt="Tests Passing" src="https://img.shields.io/github/license/shibing624/ChatPDF" />
      </a>
      <a href="https://gradio.app/">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/Base-Gradio-fb7d1a?style=flat" />
      </a>
      <p>
        根据文件回答 / 开源模型 / 本地部署LLM
      </p>
    </p>
    <p align="center">
      <img alt="Animation Demo" src="https://github.com/shibing624/ChatPDF/blob/main/docs/snap.png" width="860" />
    </p>
  </p>
</div>


- 本项目支持多种开源LLM模型，包括ChatGLM3-6b、Chinese-LLaMA-Alpaca-2、Baichuan、YI等
- 本项目支持多种文件格式，包括PDF、docx、markdown、txt等
- 本项目优化了RAG准确率
  - Chinese chunk切分优化，适配中英文混合文档
  - embedding优化，使用text2vec的sentence embedding，支持sentence embedding/字面相似度匹配算法
  - 检索匹配优化，引入jieba分词的rank_BM25，提升对query关键词的字面匹配，使用字面相似度+sentence embedding向量相似度加权获取corpus候选集
  - 新增reranker模块，对字面+语义检索的候选集进行rerank排序，减少候选集，并提升候选命中准确率，用`rerank_model_name_or_path`参数设置rerank模型
  - 新增候选chunk扩展上下文功能，用`num_expand_context_chunk`参数设置命中的候选chunk扩展上下文窗口大小
  - RAG底模优化，可以使用200k的基于RAG微调的LLM模型，支持自定义RAG模型，用`generate_model_name_or_path`参数设置底模
- 本项目基于gradio开发了RAG对话页面，支持流式对话

## 原理

<img src="https://github.com/shibing624/ChatPDF/blob/main/docs/chatpdf.jpg" width="860" />

## 使用说明

#### 安装依赖

在终端中输入下面的命令，然后回车即可。
```shell
pip install -r requirements.txt
```

如果您在使用Windows，建议通过WSL，在Linux上安装。如果您没有安装CUDA，并且不想只用CPU跑大模型，请先安装CUDA。

如果下载慢，建议配置豆瓣源。

#### 本地调用

请使用下面的命令。取决于你的系统，你可能需要用python或者python3命令。请确保你已经安装了Python。
```shell
CUDA_VISIBLE_DEVICES=0 python chatpdf.py --gen_model_type auto --gen_model_name 01-ai/Yi-6B-Chat --corpus_files sample.pdf
```

#### 启动Web服务

```shell
CUDA_VISIBLE_DEVICES=0 python webui.py --gen_model_type auto --gen_model_name 01-ai/Yi-6B-Chat --corpus_files sample.pdf --share
```

如果一切顺利，现在，你应该已经可以在浏览器地址栏中输入 http://localhost:7860 查看并使用 ChatPDF 了。





## Reference
- [imClumsyPanda/langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)

#### 关联项目推荐
- [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)：训练自己的GPT大模型，实现了包括增量预训练、有监督微调、RLHF(奖励建模、强化学习训练)和DPO(直接偏好优化)
