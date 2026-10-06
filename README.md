# AI Production Lab v0.1

本地运行的个人技术知识库 RAG 系统。v0.1 已完成到评测。v0.2 当前完成到 **Phase 7：真实检索对比**。不选择模式时，问答仍使用向量检索。重排默认关闭。

可以在浏览器里导入 Markdown / TXT、查看文档列表、删除文档，并用自然语言提问。回答下方能看到引用、检索片段和本次耗时。评测页可以运行自带的 12 道题，并比较两次报告。

## 环境

- Python 3.11 或更高版本
- 在项目根目录执行下面的命令

本机若需要代理才能访问外网，安装依赖前在 PowerShell 中设置：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7892"
$env:HTTPS_PROXY="http://127.0.0.1:7892"
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

`.env` 含有密钥，不要提交到 Git。Phase 1 默认 `ENABLE_EXTERNAL_MODELS=false`，不填写模型密钥也能启动。

## 启动

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

另开一个终端启动页面：

```powershell
$env:RAG_API_BASE_URL="http://127.0.0.1:8000"
.\.venv\Scripts\python -m streamlit run frontend/app.py --server.port 8501
```

- 健康检查：<http://127.0.0.1:8000/api/v1/health>
- 接口文档：<http://127.0.0.1:8000/docs>
- 页面：<http://127.0.0.1:8501>

如果 8000 已被占用，把 `.env` 中的 `APP_PORT` 改成空闲端口，启动命令和 `RAG_API_BASE_URL` 使用同一端口。

上传、列表、删除、问答和评测都需要 `ENABLE_EXTERNAL_MODELS=true`，并分别填写 LLM 与 Embedding。只打开健康检查时可以保持 `false`。

已经配置模型时，也可以在项目根目录一键评测：

```powershell
.\.venv\Scripts\python -m app.evaluation
```

报告写到 `data/evaluation/reports/`。这个目录不提交。样例文档以 `eval/{文件名}` 写入当前向量库，不会删除你已经导入的资料。

健康检查会打开本地 Chroma，并核对已有集合的 Embedding 模型名和维度。集合还不存在时视为正常，也不会因此创建集合。它不调用大模型，也不计算向量。索引与当前配置不一致时，整体状态是 `degraded`。

## 测试

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check app tests frontend
```

## 目录职责

| 路径 | 职责 | 进度 |
| --- | --- | --- |
| `app/main.py` | FastAPI 入口、请求日志 | Phase 1 |
| `app/config.py` | 环境变量读取和启动校验 | Phase 1 |
| `app/api` | 健康检查、文档、问答和评测接口 | Phase 5、Phase 6 已实现 |
| `app/core` | 异常、日志、依赖装配 | Phase 1 |
| `app/ingestion` | 文档读取、解析、清洗、导入和写入索引 | Phase 2、Phase 3 |
| `app/chunking` | 按字符切分，记录 Markdown 标题 | Phase 2 已实现 |
| `app/embeddings` | OpenAI 兼容的文本向量化 | Phase 3 已实现 |
| `app/vectorstore` | 向量库抽象和 ChromaDB。可列出全部片段供 BM25 重建 | Phase 3 已实现 |
| `app/retrieval` | 向量检索、BM25、RRF、重排适配器和检索器工厂。问答默认仍是向量 | v0.2 Phase 5 |
| `app/llm` | OpenAI 兼容的对话客户端 | Phase 4 已实现 |
| `app/rag` | 上下文、提示词、引用和问答编排 | Phase 4 已实现 |
| `app/evaluation` | 评测集、文档级指标、检索对比和报告比较 | v0.2 Phase 5 |
| `app/observability` | 阶段耗时、Token 和错误分类 | Phase 6 已实现 |
| `frontend` | Streamlit 页面：文档、问答、检索模式、调试、引用和评测 | v0.2 Phase 6 |
| `data/evaluation` | 30 道评测题和样例语料 | v0.2 Phase 5 |
| `tests` | 单元测试和接口测试 | Phase 1 至 Phase 6 |

## Phase 1 的设计取舍

**配置和业务分开。** LLM 与 Embedding 使用两套地址、密钥和模型名。两者可以来自不同服务商；向量维度也必须和索引一致，以后换 Embedding 模型就要重建索引，不能把旧向量和新查询向量混在一起。

**密钥用 `SecretStr`。** 打印配置对象时不会带出明文 Key。日志过滤器会再扫一遍消息，防止密钥被拼进日志。

**异常和 HTTP 分开。** 业务代码抛 `DocumentValidationError` 这类领域异常。API 层统一变成 `{"error": {"code", "message"}}`。响应里没有堆栈。

**健康检查不打模型。** 存活探针如果每次都请求 LLM 或 Embedding，会又慢又花钱，模型服务抖动时还会把本来正常的 API 判成故障。现在只打开本地 Chroma，并在集合已经存在时核对模型签名。

**相对路径相对项目根目录。** 从别的工作目录启动服务时，`./data/chroma` 仍然落在项目里。

## Phase 2 的设计取舍

切分长度按 Python 字符数计算，也就是 Unicode 码位，不是模型 Token。「事务」是 2 个字符，在 Embedding 模型里可能对应不止 1 个 Token。`CHUNK_SIZE=800` 只是大约 800 个字符。

切分时优先在空行、换行和句号处断开。分隔符如果出现得太早，仍然按 `CHUNK_SIZE` 硬切，避免切出过短片段。下一块从上一块结尾往回退 `CHUNK_OVERLAP` 个字符，相邻块因此共享一段原文。

Markdown 标题放在 `metadata["heading"]`，不写进片段正文。正文保持原文上的连续切片，重叠才能和原文逐字对照。代码围栏里的 `#` 行不当成标题。Phase 4 组装上下文时再把标题加回去。

`document_id` 来自 `source_key`，不来自文件名。同一个文件名放在不同目录里，身份不同；同一份资料重复导入，身份不变。`content_hash` 对规范化后的正文做 SHA-256。正文变了，哈希和 `chunk_id` 都会变。不提供 Embedding 和向量库时，导入只停留在内存里。

在项目根目录可以这样试一次导入，它不会访问外网：

```powershell
.\.venv\Scripts\python -c "from pathlib import Path; from app.config import get_settings; from app.ingestion.pipeline import IngestionPipeline; result = IngestionPipeline(get_settings()).ingest_path(Path('tests/fixtures/spring_transaction.md'), source_key='fixtures/spring_transaction.md'); print(result.document.document_id, result.chunk_count)"
```

这份示例短于默认 `CHUNK_SIZE=800`，所以只会得到 1 个片段。把 `CHUNK_SIZE` 调小后，同一文件会切成多段。

## Phase 3 的设计取舍

向量化走 OpenAI 兼容的 `POST {base}/embeddings`。LLM 和 Embedding 仍然是两套配置。测试使用确定性的假向量，不访问外网，也不把假模型当成正式能力。

Chroma 的 `score` 是余弦距离，越小越近，字段 `score_kind` 固定为 `distance`。集合元数据记下模型名、维度和 `hnsw:space=cosine`。再次打开时这三项对不上，就拒绝写入，并提示换集合名或删掉集合后重建。健康检查发现模型名或维度不一致时返回 `error`，整体状态变成 `degraded`。

创建集合时显式传入 `embedding_function=None`，避免 Chroma 去下载默认的 ONNX 模型。导入 Chroma 之前关闭匿名遥测。集合名必须是 3 到 512 个字符。

更新顺序是：内容哈希和片段 ID 都没变就跳过；否则先算出新向量，upsert 成功后再删除不再使用的旧 ID。向量化或写入失败时，旧索引保持原样。如果进程在删除旧 ID 之前退出，同一文档会暂时留下两版片段，下一次导入会清掉旧 ID。

## Phase 4 的设计取舍

检索只做向量搜索。问题先用和索引相同的 Embedding 变成向量，再按余弦距离取 Top-K。`score` 是距离，越小越近。`RETRIEVAL_MAX_DISTANCE` 是这条距离的上限，0 表示方向相同；留空表示只按 Top-K 截断。

上下文按检索顺序编号为 `[C1]`、`[C2]`。长度按字符数限制，放不下的整段丢掉，不把截断的半段标成完整引用。完全相同的片段只留靠前的一条。资料里的 `[C2]` 和 `</documents>` 会被改写，避免文档伪造引用标记或提前结束资料区。

系统提示词只放回答规则。问题和资料放在用户消息里，并标明资料不可信。引用映射只接受回答里出现、且这次上下文真实存在的编号。模型写出的 `[C99]` 会被丢掉，文件名和原文只来自检索结果。

没有命中时仍然调用模型，提示词要求它说明知识库缺少依据。检索失败、上下文放不下和生成失败保持不同的异常。日志只记问题长度、命中数量、耗时和 Token。

问答编排可以通过 `POST /api/v1/chat` 调用。页面只访问这个 API，不直接打开向量库。

## Phase 5 的设计取舍

上传文件只在内存里进入导入流程，不按用户给出的文件名写到磁盘。文档身份是 `upload/{文件名}`。同一个文件名再次上传会更新同一份资料；文件名里不能带目录。

文档列表从向量库的片段元数据汇总，响应里没有正文。删除按 `document_id` 清掉该文档的全部片段；文档不存在时返回 404。导入时间写在片段元数据里，重复导入且内容没变时保留第一次的时间。

问答接口返回回答、引用、检索片段和分阶段耗时。`score` 仍是余弦距离。空问题返回 422。未打开外部模型时，健康检查仍然可用，上传、列表和问答返回配置错误。

日志记录文件名、文档 ID、片段数量和耗时。不记录正文和问题原文。

## Phase 6 的设计取舍

评测集是 `data/evaluation/questions.json`。文档 ID 由 `eval/{文件名}` 算出，和样例语料一一对应。运行评测不会改写预期文档或关键词。

Recall@K 和 MRR@K 按文档计算：前 K 条片段里出现过的 `document_id` 才算命中。同一文档的多个片段只算一次。没有标注文档的拒答题不进入这两个平均值。

关键词覆盖率是答案里命中预设词的比例，只用于发现答非所问，不是正确率。拒答检查只匹配提示词里要求的那几句“缺少相关信息”，不把任意否定句当成拒答。

单题的检索失败、生成失败和接口超时会记进失败列表并继续后面的题。超时和上游 HTTP 错误归到接口，检索逻辑错误和空回答仍归到各自阶段。

每次问答额外写一条 `query_observation`。里面有请求 ID、问题长度、各阶段耗时、Token 和错误类别，没有问题原文和资料正文。

比较报告时，差值是右侧减去左侧。配置不同会单独列出来，例如 Top-K 或模型名。

## 当前已知限制

- 健康检查不探测 Embedding 或 LLM 服务，密钥错误不会体现在 `/health` 里。
- 回答正文不会被自动事实核查。引用编号无法对上不存在的片段，但模型仍可能在没有 `[C1]` 的句子里发挥。
- 切分和上下文长度都按字符数，不是 Token 数。只识别行首 ATX 标题（`#`）。
- 更换 Embedding 模型或维度后，不能继续往旧集合里写。
- 页面没有多轮对话。同名文件会视为同一份资料。
- 评测会把样例文档写进当前集合。库里的其他文档也会参与检索，指标反映的是当前库。
- 平均耗时只统计成功返回的题目。失败题的阶段耗时留在问答日志里，不进平均值。
- 拒答短语匹配不能代替人工判断。
- 页面问答固定走向量检索。混合检索和重排都还不能在页面上切换。
- BM25 分数越大越相关，余弦距离越小越近。RRF 只使用名次。重排分数只来自重排服务。
- MockReranker 不是线上重排模型。它不访问服务，也不编造相关度。

## v0.2 Phase 2 的设计取舍

关键词检索自己实现 Okapi BM25，不把分数交给会在小语料上变成 0 的 IDF。输入是分词后的词序列，输出是每个片段的相关度。词在片段里出现得越多、在整个库里越少见，分数越高；片段长度按平均长度校正。参数是 `k1=1.5`、`b=0.75`。

IDF 用的是 Lucene 的公式 `ln(1 + (N - df + 0.5) / (df + 0.5))`，结果始终大于 0。另一版 `ln((N - df + 0.5) / (df + 0.5))` 在一个词刚好出现在一半片段里时等于 0。知识库只有两篇文档时，精确关键词也会被当成没有命中。`rank_bm25` 用的就是后一版，所以这里没有依赖它。

这个分数和 Chroma 的余弦距离方向相反，代码里用 `score_kind=bm25` 标明，禁止和 `distance` 相加。

中文分词使用 jieba。它把连续汉字切成词，例如「事务失效」会变成「事务」和「失效」。英文和技术标识符先整词保留，再拆开蛇形和驼峰，所以 `max_overflow` 既能整词命中，也能按 `max`、`overflow` 命中。分词器是独立接口，不写在 BM25 检索器里面。

BM25 索引不落盘。每次提问从 Chroma 读出全部片段，在内存里重建索引。上传或删除之后，下一次提问看到的是当前库。库还小，先不做单独的关键词索引文件。

问答接口的默认检索器仍然是向量检索。要使用 BM25，需要向检索器工厂传入 `bm25`。

## v0.2 Phase 3 的设计取舍

混合检索同时询问向量检索和 BM25。向量一路默认取 20 条，BM25 一路也取 20 条，再合成最终的 10 条。调用方如果要得更多，候选数量会跟着放大，避免最终条数超过某一路的候选。

合并用的是 RRF：`1 / (60 + 名次)`，名次从 1 开始，两路分数相加。一个片段若两路都出现，只把两路名次加进去；同一路里重复出现时，只保留更靠前的那一次。某一路没有结果，另一路照常排序。

RRF 不读取余弦距离，也不读取 BM25 分数。距离越小越好，BM25 越大越好，加在一起没有意义。融合后的 `score_kind` 是 `rrf`，`retriever` 是 `hybrid`。每条结果的 `sources` 记下原来的检索器和名次。调试日志同样写下这些名次，不写问题原文和片段正文。

向量检索失败时，混合检索直接失败，不会悄悄只剩 BM25。问答默认模式仍然是 `vector`。

## v0.2 Phase 4 的设计取舍

重排接在检索之后，用来把已经取出的候选重新排序，再留下更少的几条。默认先取 20 条，再留下 5 条。它不会自己再去库里找更多片段。

线上适配器是 `OpenAICompatibleReranker`。它请求 `POST {RERANKER_BASE_URL}/rerank`，正文包含 `model`、`query`、`documents` 和 `top_n`，并只接受响应里的 `results[].index` 与 `results[].relevance_score`。相关度缺失、序号越界或结果为空时直接失败，不补假分数，也不改用检索原来的顺序冒充重排成功。

`RERANKER_ENABLED` 默认是 `false`。Phase 7 用硅基流动的 `BAAI/bge-reranker-v2-m3` 跑过一次真实重排，数字见下面的评测摘要。

`MockReranker` 只在测试里使用。它按测试给出的 `chunk_id` 顺序重排，分数仍是检索阶段的原分数，`reranker` 标成 `mock`。

```text
Mock Reranker
≠
Production Reranker
```

工厂的 `hybrid_rerank` 模式如果没有配置真实重排服务，会拒绝创建，包括把 Mock 传进去的情况。问答默认模式仍然是 `vector`。

## v0.2 Phase 5 的设计取舍

评测集扩到 30 题，标注仍是文档 ID。只命中多个相关文档中的一个时，Recall 不到 1。Precision@K 用 K 做分母，检索不足 K 条时，空出来的位置不算相关。

回答评测会记下模型写出、但上下文里不存在的引用，例如 `[C99]`。页面问答仍只返回真实存在的引用，不会把无效编号展示成来源。

`python scripts/evaluate.py` 在临时集合 `kb_eval` 上比较 vector、bm25 和 hybrid，不写入当前知识库。未配置重排时跳过 hybrid_rerank，整次运行不因此失败。这次检索没有 Token 用量，`estimated_cost_usd` 记为 null。

外部模型关闭时，脚本使用哈希向量。哈希向量只保证同一段文本得到同一条向量，不能用来证明混合检索优于向量检索。BM25 不依赖向量，那一路是这次语料上的关键词结果。报告里的数字来自当次运行，不预填示例分数。

## v0.2 Phase 6 的设计取舍

问答请求可以带 `retrieval_mode`。不带时仍是 `vector`，原来的请求体不用改。页面上的四个选项对应 vector、bm25、hybrid、hybrid_rerank。

`hybrid_rerank` 没有配置重排服务时直接失败，并说明要设置 `RERANKER_ENABLED`。页面会显示这个错误，不会改用 Mock 重排。

勾选「展开检索过程」后，响应里带上 Vector、BM25、RRF、Reranker 各步的候选。没勾选时这些列表不返回。日志只增加模式、候选数量和分步耗时，不写问题原文和片段正文。

问答编排仍然只调用 `retrieve()`。模式由检索器工厂选择，流水线不按模式写分支。

## v0.2 Phase 7 的结果

2026-10-06 在同一份 30 题上比较了 vector、bm25、hybrid、hybrid_rerank。向量模型是 `BAAI/bge-m3`，重排模型是 `BAAI/bge-reranker-v2-m3`，`semantic_embedding` 为 true。4 道拒答题不进入质量均值，仍计入延迟。费用没有估算：价格字段为空，脚本也没有把接口返回的 Token 写入报告，所以 `estimated_cost_usd` 是 null。

完整表在 `evaluation_results/summary.md`。当次结果：

| 检索器 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR@5 | 平均延迟(ms) | 成本(USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | 0.8462 | 0.9423 | 1.0000 | 1.0000 | 0.2462 | 0.9712 | 194.7758 | null |
| bm25 | 0.8846 | 0.9423 | 0.9808 | 0.9808 | 0.2385 | 1.0000 | 45.8835 | null |
| hybrid | 0.8462 | 0.9808 | 0.9808 | 0.9808 | 0.2385 | 0.9808 | 183.7970 | null |
| hybrid_rerank | 0.8846 | 0.9808 | 1.0000 | 1.0000 | 0.2462 | 1.0000 | 469.1272 | null |

Precision@5 大约 0.24，是因为分母固定为 5，而多数题只有一篇相关文档。Recall@5 为 1 表示相关文档都出现在前 5 条里。Hybrid+Reranker 同时拿到了 BM25 的首位命中和向量检索的 Recall@5，延迟大约是向量检索的 2.4 倍。BM25 最快，但有一篇相关文档即使放到前 10 也没有召回。

## 下一阶段

任务书到 Phase 7 为止。
