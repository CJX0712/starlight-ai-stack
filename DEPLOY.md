# 部署指南

作者：晨星

---

## 1. 前置条件

| 项 | 要求 | 本机实测 |
|---|---|---|
| 操作系统 | Windows 10/11 或 Linux / macOS | Windows 11 |
| Python | ≥ 3.10 | 3.13.14 |
| 编译器 | **不需要**（依赖全部走预编译 wheel） | 无 cmake / gcc / MSVC |
| 模型服务 | Ollama ≥ 0.5，或任意 OpenAI 兼容端点 | Ollama 0.34 |
| 内存 | 建议 ≥ 4GB 可用（跑 7B 档） | 15GB 总，实测可用 1.7GB 时已自动降级 |
| 磁盘 | 模型 1–5GB + 索引（视语料） | 剩余 36GB |

## 2. 安装

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate

# 关键：--only-binary=:all: 保证不会碰到需要本地编译的包
pip install --only-binary=:all: -r requirements.txt
```

想看完整传递依赖树，用锁定文件：

```bash
pip install --only-binary=:all: -r requirements.lock.txt
```

## 3. 准备模型

```bash
ollama serve                                   # 终端 A，常驻

ollama pull qwen2.5:7b-instruct-q4_K_M        # 高配档 LLM（约 4.7GB）
ollama pull bge-m3                             # 中文嵌入（约 1.2GB，1024 维）
# 可选降级组合
ollama pull qwen3:4b                           # 约 2.5GB
ollama pull nomic-embed-text                   # 约 274MB，768 维
```

网络慢时不必等齐：缺哪个模型，`resolve_profile` 会自动挑本机已有的同类模型顶上，并在 `/health` 的 `notes` 里说明。

## 4. 启动

```bash
python scripts/start.py --profile high         # 自动拉模型服务 + 启网关 + 开浏览器
```

等价手动方式：

```bash
set STARLIGHT_PROFILE=high
uvicorn starlight.server:app --host 127.0.0.1 --port 8787
```

访问 `http://127.0.0.1:8787` 使用控制台；`http://127.0.0.1:8787/docs` 看接口文档。

## 5. 配置

全部配置走环境变量（见 `.env.example`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `STARLIGHT_PROFILE` | high | 档位：high / balanced / lite |
| `STARLIGHT_OLLAMA_URL` | http://127.0.0.1:11434 | 模型服务地址 |
| `STARLIGHT_HOME` | ./data | 数据库与索引目录 |
| `STARLIGHT_CHUNK_SIZE` | 420 | 切块字数 |
| `STARLIGHT_CHUNK_OVERLAP` | 80 | 块间重叠字数 |
| `STARLIGHT_TOP_K` | 5 | 送入生成的块数 |
| `STARLIGHT_CANDIDATE_K` | 20 | 每路召回候选数 |
| `STARLIGHT_RRF_K` | 60 | RRF 融合平滑系数 |

换后端为任意 OpenAI 兼容端点（llama.cpp server / vLLM / 云端）：

```python
from starlight.pipeline import RAGPipeline
pipe = RAGPipeline(backend="openai")   # 配合 STARLIGHT_OLLAMA_URL 指向该端点
```

## 6. 容器化部署

```bash
docker compose up -d                          # 起 ollama + api
docker compose run --rm model-bootstrap       # 拉齐当前档位所需模型（幂等）
docker compose logs -f starlight              # 看服务日志
```

两个服务职责分离，各自用命名卷：`ollama-models` 存模型权重，`starlight-data` 存知识库。
重建应用容器不会丢模型，删掉 starlight 容器也不会丢知识库。

换档位：

```bash
STARLIGHT_PROFILE=high docker compose up -d starlight
```

只跑单元测试（不需要模型服务，适合 CI）：

```bash
docker run --rm -v "$PWD":/app -w /app python:3.13-slim \
  sh -c "pip install --only-binary=:all: -r requirements.txt && python -m pytest -q"
```

## 7. 评测门禁

```bash
python scripts/eval.py --mode rag             # 检索+生成，10 条金标集
python scripts/eval.py --mode agent           # 智能体路径
python scripts/eval.py --profile balanced     # 换档位对比效果
```

退出码：0 通过 / 1 指标未达标 / 2 环境问题。阈值与用例在 `eval/golden_set.json` 中维护，
语料放在 `eval/corpus/`。扩充语料后建议同步上调阈值。

CI 中的 `eval-gate` 任务默认不跑（需下载约 1.5GB 模型），在 Actions 页面手动触发即可。

## 8. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `/health` 返回 502 或后端离线 | 全局 HTTP 代理把 localhost 也代理了 | 代码已 `trust_env=False`；若自己改客户端请保持该设置 |
| 嵌入报 404 | 档位指定的嵌入模型本机没有 | 看 `/health` 的 `notes`，执行 `ollama pull <模型>` 或换档位 |
| 建库 `PermissionError` | Windows 临时目录被安全软件占用 | 设 `STARLIGHT_HOME` 到项目目录下 |
| 生成极慢（<1 tok/s） | 内存不足导致换页 | 换 `balanced` 档，或关闭占内存的容器后重启 |
| 装依赖慢到以分钟计 | 大模型常驻内存后磁盘换页 | `ollama stop <模型>` 腾出内存再装 |
| 答案前面一大段分析 | 用到了思考型模型（qwen3 系列） | 换 `qwen2.5` 系列；代码已做开头标记清洗作为兜底 |
| 答案说"没有检索到相关内容" | 库是空的 | 先在控制台摄取目录，或用 `scripts/e2e.py` 灌入示例语料 |
| `/agent` 一直调工具不返回 | 工具反复失败或问题本身无解 | 降低 `max_steps`（1–8），并看返回轨迹里每步的 observation |
| 装依赖报编译错误 | 某个包没有 wheel | 保持 `--only-binary=:all:`，换有 wheel 的替代库 |

## 9. 备份与迁移

整个知识库就是一个 SQLite 文件：`$STARLIGHT_HOME/starlight.db`。拷贝该文件即完成迁移；换机器后放到同一位置，改嵌入模型维度不一致时系统会自动按原文重建向量索引（原文始终保留，不会丢）。

## 10. 验收清单

```bash
python -m starlight.selftest     # 期望：通过 7/7
python -m pytest -q              # 期望：55 项全部通过
python scripts/e2e.py            # 期望：结论=全部通过
python scripts/eval.py --mode rag  # 期望：门禁=通过
python scripts/verify_env.py     # 期望：干净环境可一键复现
```
