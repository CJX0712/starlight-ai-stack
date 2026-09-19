# Starlight AI Stack

本地优先的检索增强生成（RAG）系统：**摄取 → 混合检索 → 引用生成**，全链路可在单台无 GPU 的 Windows 机器上跑通。

作者：晨星

---

## 1. 它解决什么问题

| 问题 | 本项目的做法 |
|---|---|
| 模型依赖云端、数据出网 | 全部本地推理，知识库落本地 SQLite 文件 |
| 大模型张口就来、无法追溯 | 答案强制带 `[n]` 引用，编号必须映射回真实文本块 |
| 中文检索效果差 | 稀疏 BM25（单字+二字组倒排）与稠密向量双路召回，RRF 融合互相兜底 |
| 换机器就装不起来 | 依赖版本锁定 + 仅二进制 wheel 安装 + 一键 bootstrap |
| 出问题不知道哪环坏了 | 每个模块独立自检，判据是机械不变量而非"看起来能跑" |

## 2. 模块划分与接口

上层只依赖 `contracts.py` 里的 Protocol，不依赖任何具体实现——因此模型后端、存储、检索策略都可以整层替换。

| 模块 | 单一职责 | 关键接口 | 独立验证方式 |
|---|---|---|---|
| `config` | 档位、环境变量、降级决策 | `Settings` / `resolve_profile` | 档位表完整性、内存守卫阈值 |
| `contracts` | 接口契约与数据类 | `Document / Chunk / ScoredChunk / Answer` | 类型协议一致性 |
| `models` | 模型服务适配 | `chat(messages)` / `embed(texts)` | 服务可达、维度恒定、工具调用可用 |
| `embedder` | 文本 → 定长向量 | `embed(texts) -> list[vector]` | 维度一致、向量已归一化 |
| `store` | 存与查（稠密+稀疏+原文） | `upsert_document` / `vector_search` / `sparse_search` | top1 自反、幂等、删除后不可检索 |
| `ingest` | 文件 → 文档 → 切块 | `ingest_path(path) -> IngestReport` | 重复摄取新增 = 0 |
| `retriever` | 双路召回与融合 | `retrieve(query,k) -> list[ScoredChunk]` | 已知查询 top1 命中关键词 |
| `generator` | 生成与引用解析 | `generate(query, ctx) -> Answer` | 越界引用编号被剔除 |
| `pipeline` | 装配与编排 | `ingest / search / ask` | 端到端问答产出非空引用 |
| `server` | HTTP 契约与 SSE | `/health /ingest /search /chat` | 契约测试 |
| `selftest` | 逐模块自检 | `run(modules)` | 全绿即链路可用 |

调用关系（箭头为依赖方向）：

```
server → pipeline → {retriever, generator, ingestor}
                        ↓            ↓         ↓
                     store       models     embedder → models
                        ↓
                   sqlite-vec + BM25 倒排
```

## 3. 快速开始

```bash
# 0) 先起模型服务（另开一个终端，或让脚本自动拉起）
ollama serve

# 1) 安装依赖（只装预编译 wheel，本机无编译器也能装）
pip install --only-binary=:all: -r requirements.txt

# 2) 一键引导：装依赖 + 准备模型 + 跑自检
python scripts/bootstrap.py --profile high

# 3) 启动服务与控制台
python scripts/start.py --profile high      # 浏览器会自动打开 http://127.0.0.1:8787
```

只跑验收不启服务：

```bash
python scripts/e2e.py                       # 内置语料：摄取 → 检索 → 问答 → 引用校验
python -m starlight.selftest                # 六模块自检
python -m starlight.selftest --module store # 只验一个模块
python -m pytest -q                         # 不依赖模型服务的单元测试
```

## 4. HTTP 接口

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 单文件控制台页面 |
| GET | `/health` | 后端状态、生效档位、降级说明、库统计 |
| POST | `/ingest` | `{"path": "D:/docs"}` 摄取目录或文件 |
| POST | `/ingest_text` | `{"text": "...", "source": "inline"}` 摄取一段文本 |
| GET | `/search?q=&k=5&debug=true` | 检索；debug 会返回稠密/稀疏两路原始排序 |
| POST | `/chat` | 非流式问答，返回答案与引用 |
| POST | `/chat/stream` | SSE 流式：`context` → `delta` → `done` |
| DELETE | `/documents/{doc_id}` | 删除整篇文档及其索引 |
| GET | `/stats` | 文档数、块数、词项数、向量维度 |

## 5. 模型档位

| 档位 | LLM | 嵌入 | 适用 |
|---|---|---|---|
| `high` | qwen2.5:7b-instruct-q4_K_M | bge-m3 | 内存充足（≥4GB 可用）时的最高质量档 |
| `balanced` | qwen3:4b | bge-m3 / nomic-embed-text | 常规档，CPU 上 4-5 tok/s |
| `lite` | qwen3:4b | nomic-embed-text | 内存紧张，缩短上下文与生成长度 |

**降级是自动的，且有据可查**：模型没拉下来、或可用内存低于 4GB，系统会自动切到能跑的组合，并在 `/health` 的 `notes` 字段写明原因。不会因为"配置里写了 7B"就直接崩。

## 6. 本机实测数据（AMD Ryzen 7 H 255 / 16GB / 无独显）

| 项 | 实测 |
|---|---|
| qwen3:4b 生成速度 | 4.2–5.3 tok/s（CPU，锁 4 线程最快；开 8 线程反而降到 4.4） |
| bge-m3 嵌入 | 1024 维，3 条短文本约 1.3s |
| nomic-embed-text 嵌入 | 768 维，单条约 0.06s |
| 六模块自检 | 6/6 通过 |
| 单元测试 | 13/13 通过（不依赖模型服务） |
| 端到端 | 摄取 → 检索 → 生成 → 引用，全链路通过 |

## 7. 踩过的坑（都已写进代码注释）

| 坑 | 现象 | 处置 |
|---|---|---|
| 全局 HTTP 代理拦截 localhost | 调 127.0.0.1:11434 返回 502 | httpx 客户端 `trust_env=False` |
| SQLite FTS5 trigram 分词 | 查询「检索」二字命中 0 条 | 弃用 FTS5，自建 BM25 倒排（单字+二字组） |
| Windows 临时目录建库 | `PermissionError` 文件被占用 | 自检库改放项目 `data/` 目录 |
| 思考型模型污染答案 | 答案前面一大段分析，甚至把指令本身拿去推理 | 输出格式契约 + 只在开头判定标记，宁可不切不可误切 |
| GitHub release 附件下不动 | llama.cpp 预编译包 curl 返回 000 | 改用已安装的 Ollama 作推理后端，接口层保持可替换 |
| huggingface.co 不可达 | 模型下载超时 | 走 Ollama 模型库，或 hf-mirror / ModelScope 镜像 |

## 8. 目录结构

```
starlight-ai-stack/
├─ src/starlight/        核心模块（按单一职责拆分）
├─ scripts/              bootstrap / start / e2e
├─ web/index.html        单文件控制台（内联样式脚本，无外部依赖）
├─ tests/                单元测试（不依赖模型服务）
├─ docs/ADR-*.md         架构决策记录
├─ requirements.txt      版本锁定的依赖清单
├─ DEPLOY.md             部署与故障排查
└─ README.md
```

## 9. 许可与署名

代码与文档作者：晨星。第三方依赖遵循各自开源许可（FastAPI、uvicorn、sqlite-vec、pypdf、numpy、openai SDK）；模型遵循 Ollama 模型库各自许可（Qwen 系列、BGE 系列、Nomic）。
