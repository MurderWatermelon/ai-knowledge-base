"""运行：streamlit run app.py"""
import hashlib
import json
from pathlib import Path

import streamlit as st

from evaluation import evaluate, fingerprint, load_dataset, to_csv
from rag import EMBEDDING_MODEL, SemanticEmbedder, read_with_local_model, build_index

st.set_page_config(page_title="个人 AI 知识库", page_icon="📚", layout="wide")
st.title("📚 个人 AI 知识库")
st.caption("上传资料 · 理解相近表述 · 原文解读 · 检索评测")


@st.cache_resource
def semantic_embedder():
    return SemanticEmbedder()


def show_sources(results):
    for rank, (chunk, score) in enumerate(results, 1):
        page = f" · 第 {chunk['page']} 页" if chunk["page"] else ""
        st.write(f"{rank}. {chunk['file']}{page} · 相似度 {score:.3f}")
        st.text(chunk["text"])


with st.sidebar:
    st.header("知识库设置")
    backend_label = st.selectbox("检索方式", ["中文语义理解（首次需下载模型）", "字符 TF-IDF（离线基线）"])
    backend = "lexical" if backend_label.startswith("字符") else "semantic"
    if backend == "semantic":
        st.caption("按含义查找相近表述，使用本地中文向量模型。首次下载后可在本机检索。")
    else:
        st.caption("字符基线主要比较用词。想查找换种说法的答案，请选择“中文语义理解”。")
    chunk_size = st.number_input("分块长度（字符）", min_value=20, max_value=10000, value=500, step=20)
    overlap = st.number_input("重叠长度（字符）", min_value=0, max_value=9999, value=80, step=10)
    uploads = st.file_uploader("上传 PDF / TXT / Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True)
    use_example = st.checkbox("使用项目自带的示例文档", value=True)
    use_reading_example = st.checkbox("加入阅读理解示例资料", value=False)
    files = [(u.name, u.getvalue()) for u in uploads]
    if use_example:
        files += [(p.name, p.read_bytes()) for p in sorted((Path(__file__).parent / "examples/documents").glob("*.md"))]
    if use_reading_example:
        files += [(p.name, p.read_bytes()) for p in sorted((Path(__file__).parent / "examples/reading_documents").glob("*.md"))]
    config = {"backend": backend, "embedding_model": EMBEDDING_MODEL if backend == "semantic" else "character-tfidf-v1",
              "chunk_size": int(chunk_size), "overlap": int(overlap), "corpus_sha256": fingerprint(files)}
    if st.button("建立 / 更新知识库", type="primary"):
        try:
            if not 0 <= overlap < chunk_size:
                raise ValueError("重叠长度必须小于分块长度。")
            with st.spinner("正在解析文档并建立索引……"):
                index = build_index(files, int(chunk_size), int(overlap), backend,
                                    semantic_embedder() if backend == "semantic" else None)
            st.session_state.index = index
            st.session_state.config = config
            st.session_state.pop("report", None)
            st.session_state.pop("report_key", None)
            st.success(f"已建立 {len(index.chunks)} 个片段。")
        except Exception as exc:
            st.error(f"建立索引失败：{exc}")

index = st.session_state.get("index")
active_config = st.session_state.get("config")
current = index is not None and active_config == config
if index is not None:
    st.info(f"当前索引：{len(index.chunks)} 个片段；分块 {active_config['chunk_size']}，重叠 {active_config['overlap']}。")
    if not current:
        st.warning("文档或设置已改变，请点击“建立 / 更新知识库”后继续。")
else:
    st.info("先点击左侧“建立 / 更新知识库”。中文语义模式首次需下载模型；可切换字符基线体验无需模型下载的检索。")

search_tab, eval_tab = st.tabs(["检索与问答", "召回率评测"])
with search_tab:
    with st.form("search"):
        question = st.text_input("你的问题", value="Python 字典的键有什么要求？")
        top_k = st.slider("检索 Top-k", 1, 20, 4)
        generate = st.checkbox("解读资料并回答（需已安装并启动 Ollama）")
        st.caption("例如：原文说“签收后七日内可申请退货”，你可以问“收到东西五天了，还能退吗？”")
        submitted = st.form_submit_button("检索", disabled=not current)
    if submitted and question.strip():
        try:
            results = index.retrieve(question, top_k)
            if generate:
                with st.spinner("正在阅读资料、组织答案并核对引用……"):
                    reading = read_with_local_model(question, results)
                if reading["status"] == "insufficient":
                    st.warning(reading["answer"])
                else:
                    st.subheader("答案")
                    st.write(reading["answer"])
                    st.subheader("解读依据")
                    st.write(reading["explanation"])
                    for citation in reading["citations"]:
                        page = f" · 第 {citation['page']} 页" if citation["page"] else ""
                        st.write(f"[资料{citation['source']}] {citation['file']}{page}")
                        st.text(citation["quote"])
                    st.caption("引文已核对来自检索原文；解读是否成立仍需结合条件与上下文检查。")
            show_sources(results)
        except Exception as exc:
            st.error(f"运行失败：{exc}")

with eval_tab:
    st.write("用标注原文核对检索结果，不调用生成模型。召回率衡量找到多少条正确证据，不代表回答正确率。")
    dataset_choice = st.selectbox("内置评测题集", ["基础检索", "阅读理解与同义改写"])
    dataset_name = "evaluation.json" if dataset_choice == "基础检索" else "paraphrase_evaluation.json"
    if dataset_choice != "基础检索":
        st.info("请在左侧勾选“加入阅读理解示例资料”并重建知识库。本题集评估同义问题能否找到正确段落，不自动评分生成答案。")
    sample = (Path(__file__).parent / "examples" / dataset_name).read_bytes()
    st.download_button("下载示例题集", sample, "evaluation.example.json", "application/json")
    uploaded_eval = st.file_uploader("上传评测题集（JSON）；不上传则使用示例", type=["json"])
    dataset_bytes = uploaded_eval.getvalue() if uploaded_eval else sample
    ks_text = st.text_input("对比 Top-k（用逗号分隔）", "1,3,5")
    st.caption("可在左侧修改分块长度并重建索引，再运行同一题集比较；批量分块实验见 README。")
    report_key = hashlib.sha256(dataset_bytes + ks_text.encode() + json.dumps(config, sort_keys=True).encode()).hexdigest()
    if st.button("运行召回率评测", disabled=not current):
        st.session_state.pop("report", None)
        try:
            questions = load_dataset(dataset_bytes)
            ks = [int(k.strip()) for k in ks_text.split(",")]
            with st.spinner("正在检索并核对标注……"):
                report = evaluate(questions, index.retrieve, index.chunks, ks,
                                  {**active_config, "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                                   "chunk_count": len(index.chunks)})
            st.session_state.report = report
            st.session_state.report_key = report_key
        except Exception as exc:
            st.error(f"评测失败：{exc}")
    report = st.session_state.get("report")
    if report and current and st.session_state.get("report_key") == report_key:
        st.write(f"题目 {report['question_count']} 道 · 可回答 {report['answerable_count']} 道 · 无答案 {report['unanswerable_count']} 道")
        if not report["answerable_count"]:
            st.warning("没有可回答题，召回率未定义；请添加有效证据标注。")
        if report["questions_with_unmatchable_evidence"]:
            st.warning(f"有 {report['questions_with_unmatchable_evidence']} 道题的部分标注无法在当前分块中找到，仍计入召回分母。请检查原文、来源、页码或分块边界。")
        st.table([{"Top-k": int(k), "Recall@k": m["recall"], "Hit Rate@k": m["hit_rate"], "MRR@k": m["mrr"]}
                  for k, m in report["summary"].items()])
        st.caption("Recall：每题命中证据数 / 标注证据数，再对可回答题取平均。Hit Rate：至少命中一条证据的题目比例。MRR：首个相关结果排名倒数的平均。")
        st.download_button("下载完整 JSON 报告", json.dumps(report, ensure_ascii=False, indent=2), "retrieval-report.json", "application/json")
        st.download_button("下载逐题 CSV", to_csv(report).encode("utf-8-sig"), "retrieval-details.csv", "text/csv")
        inspect_k = st.selectbox("逐题查看 Top-k", list(report["summary"]), format_func=lambda v: f"Top-{v}")
        only_misses = st.checkbox("只显示未完全召回的题目")
        for row in report["details"]:
            metric = row["metrics"][inspect_k]
            if only_misses and (not row["answerable"] or metric["recall"] == 1):
                continue
            label = f"{metric['recall']:.0%}" if row["answerable"] else "无答案题（不计分）"
            with st.expander(f"{row['id']} · {row['question']} · {label}"):
                if row["unmatchable_evidence"]:
                    st.warning("当前分块无法匹配的证据：" + ", ".join(row["unmatchable_evidence"]))
                for evidence in metric["missed_evidence"]:
                    st.write(f"未召回：{evidence['file']} · 页码 {evidence.get('page')} · {evidence['id']}")
                    st.text(evidence["quote"])
                for rank, result in enumerate(row["retrieved"][:int(inspect_k)], 1):
                    st.write(f"第 {rank} 名 · {result['file']} · 页码 {result['page']} · 分数 {result['score']:.3f} · 命中：{', '.join(result['matched_evidence']) or '无'}")
                    st.text(result["text"])
                if not row["answerable"]:
                    st.caption("这里只展示无答案题检索出的片段。当前检索总会返回 Top-k，不能据此评估拒答能力。")
