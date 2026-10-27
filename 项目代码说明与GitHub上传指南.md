# 项目代码说明与 GitHub 上传指南

## 一、从哪里看代码

项目位置：`/Users/piaoliangjuesha1207/Documents/ChatGPT/ai知识库`。

```text
ai知识库/
├── app.py                         网页入口
├── rag.py                         文档处理、向量检索、Ollama 调用
├── evaluation.py                  题集校验、证据匹配和评测指标
├── evaluate.py                    命令行批量实验入口
├── examples/
│   ├── evaluation.json            10 道演示题及其证据标注
│   └── documents/                3 份自编演示资料
├── tests/
│   ├── test_evaluation.py         指标与检索自动化测试
│   └── test_reading.py            语义编码、解读与引文校验测试
├── requirements.txt               安装依赖清单
├── .gitignore                     不提交到 Git 的文件规则
└── README.md                      项目安装、使用与评测说明
```

问答流程：`app.py → rag.build_index → Index.retrieve → read_with_local_model（可选）→ validate_reading_response`。

网页评测流程：`app.py → evaluation.load_dataset → evaluation.evaluate → Index.retrieve → matches → 指标与报告`。

批量评测流程：`evaluate.py → 循环建立不同分块配置的索引 → evaluation.evaluate → 保存报告`。

## 二、各模块的作用

### app.py：让用户操作项目

负责上传资料、选择检索方式、设置分块参数、展示检索来源、运行评测和下载报告。

- `semantic_embedder()`：加载语义向量模型，使用 `st.cache_resource` 缓存，避免每次点击重复加载。
- `show_sources()`：展示文件、页码、相似度和原文。
- `st.session_state`：在当前网页会话中保存索引与报告；没有写入数据库，重启后需要重新建立。
- `current`：核对界面设置是否与已建立索引一致。文档或参数变化后要求重建。
- `report_key`：题集、Top-k 或设置变化后隐藏旧报告。

### rag.py：实际处理文档并寻找证据

| 函数或类 | 作用 |
|---|---|
| `normalize()` | 合并连续空白，让分块和证据匹配使用一致的文本形式。 |
| `split_text()` | 按字符长度分块，相邻片段保留重叠；校验重叠长度小于分块长度。 |
| `read_document()` | 提取 PDF、TXT、MD 文本；保留 PDF 页码，为片段生成标识。 |
| `LexicalEmbedder` | 把字符和相邻双字符转换为 TF-IDF 向量，作为不下载模型的演示基线。 |
| `SemanticEmbedder` | 使用中文 BGE 向量模型理解相近表达，只为问题添加中文检索指令。 |
| `unit_vectors()` | 把非零向量长度归一化到 1，零向量保留为零。 |
| `Index` | 保存片段、向量与对应的编码器，提供 `retrieve()`。 |
| `build_index()` | 把文件解析、分块、编码串起来，支持字符基线或 Sentence Transformers。 |
| `ask_local_model()` | 将问题与检索片段传给本机 Ollama，设置 180 秒超时。 |
| `read_with_local_model()` | 当前网页使用的阅读理解入口，要求生成答案、解释和可核对的原文引文。 |
| `validate_reading_response()` | 检查解读格式和引文来源，拦截不存在的来源及伪造引文；不自动证明推论正确。 |

核心检索代码：

```python
encode_query = getattr(self.embedder, "encode_query", None)
query_vectors = encode_query(question) if encode_query else self.embedder.encode([question])
query = unit_vectors(query_vectors)[0]
scores = self.vectors @ query
indices = np.argsort(-scores, kind="stable")[:top_k]
return [(self.chunks[int(i)], float(scores[int(i)])) for i in indices]
```

第一行把问题转换为单位向量。第二行计算每个片段与问题的点积；对非零单位向量，它就是余弦相似度。第三行按分数从高到低选前 k 个，分数相同时保持原片段顺序。最后返回片段及其分数。

当前使用 NumPy 对所有片段计算分数，是内存中的精确检索，适合小规模教学演示；尚未使用 FAISS、Chroma 或近似最近邻索引。

### evaluation.py：判断检索是否找到了正确证据

| 函数 | 作用 |
|---|---|
| `load_dataset()` | 检查 JSON 结构、题目 ID、证据 ID、原文、页码和无答案题标记。 |
| `matches()` | 文件名相同、可选页码符合、片段包含完整标注原文时记为命中。 |
| `evaluate()` | 检索题目，计算每题及总体指标，记录排序、漏召回与无法匹配的证据。 |
| `fingerprint()` | 对文件名与内容计算 SHA-256 指纹，帮助核对实验资料是否一致。 |
| `to_csv()` | 导出逐题 CSV，处理引号、换行及表格公式前缀。 |

指标口径：

- 单题 Recall@k = 前 k 个片段覆盖的不同标注证据数 / 该题标注证据总数。
- 总体 Recall@k = 所有可回答题的单题 Recall@k 的平均。
- Hit Rate@k = 至少找到一条证据的可回答题比例。
- MRR@k = 首个正确片段排名倒数的平均；前 k 条都不正确记 0。

例如一题标注 A、B 两条证据，前 3 个结果为“无关、A、重复 A”，Recall@3 为 1/2，而不是 2/2；首个相关片段排第 2，RR@3 为 1/2。

无答案题不加入这三个指标的分母。资料中不存在或被分块边界切断的已标注证据会被提示，但仍保留在分母，避免删掉困难样本使成绩虚高。

原文匹配不能自动识别同义改写，也不会给未标注的其他有效来源计分；它衡量的是标注证据的覆盖情况。拒答能力和生成答案正确性需要另外评估。

### evaluate.py：自动完成多组实验

通过 `argparse` 读取参数，循环不同分块长度，共用 `evaluation.py` 的指标逻辑。每种配置导出 JSON、CSV，并在 `comparison.json` 中汇总。

离线示例：

```bash
python evaluate.py --backend lexical --chunk-sizes 80 140 240 --overlap 20 --ks 1 3 5
```

语义模型实验，首次需要下载模型：

```bash
python evaluate.py --backend semantic --chunk-sizes 300 500 800 --overlap 80 --ks 1 3 5
```

字符基线与语义模型必须分别报告，不能把字符基线成绩当作语义模型成绩。

### examples：提供可复现的输入

`documents` 是项目自编资料；`evaluation.json` 包含 8 道可回答题和 2 道无答案题。

证据标注示例：

```json
{
  "id": "q01",
  "question": "Python 字典的键有什么要求？",
  "answerable": true,
  "relevant": [
    {
      "id": "keys",
      "file": "python_notes.md",
      "page": null,
      "quote": "字典中的键必须是唯一的，并且必须是可哈希的对象。"
    }
  ]
}
```

完整题集需要外层 `{"version": 1, "questions": [...]}`。PDF 页码从 1 开始，TXT、MD 一般为 null。文件名需与文档一致，quote 需来自文档原文。

### tests、依赖与配置

- `tests/test_evaluation.py`：使用 Python 自带 unittest，测试手算指标、重叠去重、页码、错误输入、无答案题及离线端到端流程，不下载模型。
- `requirements.txt`：列出 Streamlit、PyMuPDF、Sentence Transformers、NumPy。
- `.gitignore`：排除虚拟环境、缓存、密钥、自动生成报告等文件。
- `README.md`：让没有看过聊天记录的人也能安装并运行项目。

## 三、在当前电脑运行

当前项目已经有可用的 `.venv`，可体验中文语义检索，也可以切换字符基线：

```bash
cd "/Users/piaoliangjuesha1207/Documents/ChatGPT/ai知识库"
source .venv/bin/activate
python -m streamlit run app.py
```

如果提示 8501 端口已占用，可直接打开正在运行的页面，或用 `python -m streamlit run app.py --server.port 8502` 启动另一个端口。

打开网页后点击“建立 / 更新知识库”，进入“召回率评测”，点击“运行召回率评测”。

当前环境已经安装语义检索依赖并下载中文向量模型。在其他环境请先安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

在另一台电脑先安装 Python 3.11 / 3.12，在下载后的项目目录创建独立环境，再安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows 激活命令为 `.venv\Scripts\activate`（命令提示符），其余 Python 命令相同。

## 四、上传 GitHub

以下步骤由你在终端执行；尚未替你创建 GitHub 仓库或上传代码。

### 1. 创建空仓库

登录 https://github.com/new ，仓库名可填写 `ai-knowledge-base`，描述可填写 `个人 AI 知识库问答与证据召回评测系统`。

想用于作品集可选择 Public；还不想公开可先选 Private。为了直接推送本地代码，此处不要勾选初始化 README、.gitignore 或 License，因为本地已有文件。点击 Create repository。

### 2. 确认 Git 可用

```bash
git --version
```

macOS 若提示安装命令行工具，按系统提示安装后继续。

### 3. 初始化并选择提交文件

```bash
cd "/Users/piaoliangjuesha1207/Documents/ChatGPT/ai知识库"
git init -b main
git add app.py rag.py evaluation.py evaluate.py requirements.txt README.md .gitignore examples tests "项目代码说明与GitHub上传指南.md" "阅读理解功能说明.md"
git status
git diff --cached --stat
```

显式列出文件，确保只提交本项目源码、自编示例和文档；虚拟环境、模型、个人资料和临时报告不需要上传。源码压缩包也不用提交。

### 4. 保存第一次提交

```bash
git commit -m "Build RAG demo and evidence recall evaluation"
```

如果出现 `Author identity unknown`，将下列名字和邮箱替换为你自己的信息。邮箱可使用 GitHub Settings → Emails 中提供的 noreply 地址：

```bash
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub邮箱或noreply邮箱"
git commit -m "Build RAG demo and evidence recall evaluation"
```

此处不使用 `--global`，只配置本项目。

### 5. 连接远程仓库并推送

将 `YOUR_USERNAME` 替换为你的 GitHub 用户名；仓库名如果不同也相应修改：

```bash
git remote add origin https://github.com/YOUR_USERNAME/ai-knowledge-base.git
git push -u origin main
```

若提示登录，GitHub 的 HTTPS 推送不接受账户密码，使用 GitHub CLI 的浏览器登录，或合适权限的个人访问令牌。不要把令牌写到代码或远程仓库 URL 中。

如果已经安装 GitHub CLI，可先执行：

```bash
gh auth login
gh auth setup-git
git push -u origin main
```

`gh auth login` 选择 GitHub.com、HTTPS 和浏览器登录，按提示完成。

推送成功后刷新 GitHub 仓库，应看到 app.py、rag.py、evaluation.py、evaluate.py、examples、tests 与 README。

### 6. 后续修改怎样同步

例如修改了评测模块和测试：

```bash
git add evaluation.py tests/test_evaluation.py
git commit -m "Improve retrieval evaluation"
git push
```

每次先用 `git status` 查看更改，只提交本次有意义的修改。

### 常见问题

- `remote origin already exists`：先执行 `git remote -v` 核对地址；确实填错时再用 `git remote set-url origin https://github.com/YOUR_USERNAME/ai-knowledge-base.git`。
- `Repository not found`：检查用户名、仓库名，以及登录账号是否有访问权限。
- 推送提示远程已有提交：可能建库时初始化了 README。先检查两端内容并合并，保留已有文件；不要直接强制推送覆盖。

## 五、面试中怎样讲这个项目

可以这样解释实际实现：

> 我把系统拆成界面、检索、评测和实验入口。问答和评测使用同一个检索器，避免评测的是另一套逻辑。我按原文证据标注问题，在不同分块长度与 Top-k 下比较召回率，并处理重叠片段重复命中、无答案题和证据跨边界等情况。当前已验证中文语义检索及评测流程；阅读理解已实现引用校验与界面，生成答案的实际效果还需要安装 Ollama 后单独评估。

建议阅读顺序：先看 `rag.py` 的分块与检索，再看 `evaluation.py` 的匹配与计分，最后看 `app.py` 和 `evaluate.py` 如何调用它们。
