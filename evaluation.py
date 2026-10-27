"""独立于 LLM 的证据召回评测；相关证据是集合，不是片段数量。"""
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone

from rag import normalize


def load_dataset(data):
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"评测文件不是有效的 UTF-8 JSON：{exc}") from exc
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("评测集必须包含 version: 1 和 questions 列表。")
    questions = value.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions 必须是非空列表。")
    ids = set()
    for q in questions:
        if not isinstance(q, dict):
            raise ValueError("每道题必须是对象。")
        for field in ("id", "question"):
            if not isinstance(q.get(field), str) or not q[field].strip():
                raise ValueError(f"每道题需要非空字符串字段 {field}。")
        if q["id"] in ids:
            raise ValueError(f"题目 ID 重复：{q['id']}")
        ids.add(q["id"])
        if type(q.get("answerable")) is not bool:
            raise ValueError(f"{q['id']}：answerable 必须是 true 或 false。")
        relevant = q.get("relevant")
        if not isinstance(relevant, list) or bool(relevant) != q["answerable"]:
            raise ValueError(f"{q['id']}：可回答题必须标注证据，无答案题的 relevant 必须为空。")
        evidence_ids, signatures = set(), set()
        for e in relevant:
            if not isinstance(e, dict):
                raise ValueError(f"{q['id']}：证据必须是对象。")
            for field in ("id", "file", "quote"):
                if not isinstance(e.get(field), str) or not e[field].strip():
                    raise ValueError(f"{q['id']}：每条证据需要非空 {field}。")
            page = e.get("page")
            if page is not None and (type(page) is not int or page < 1):
                raise ValueError(f"{q['id']}：页码须为从 1 开始的整数或 null。")
            signature = (e["file"], page, normalize(e["quote"]))
            if e["id"] in evidence_ids or signature in signatures:
                raise ValueError(f"{q['id']}：重复证据不能重复计分。")
            evidence_ids.add(e["id"])
            signatures.add(signature)
    return questions


def matches(chunk, evidence):
    return (chunk["file"] == evidence["file"]
            and (evidence.get("page") is None or chunk.get("page") == evidence["page"])
            and normalize(evidence["quote"]) in normalize(chunk["text"]))


def evaluate(questions, retrieve, corpus, ks=(1, 3, 5), metadata=None):
    ks = list(ks)
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("k 列表必须由正整数组成。")
    ks = sorted(set(ks))
    # 公共函数调用同样校验，避免绕过文件加载时产生错误分母。
    questions = load_dataset(json.dumps({"version": 1, "questions": questions}))
    rows = []
    for q in questions:
        results = list(retrieve(q["question"], max(ks)))[:max(ks)]
        absent = [e["id"] for e in q["relevant"] if not any(matches(c, e) for c in corpus)]
        ranked = []
        for chunk, score in results:
            if not math.isfinite(float(score)):
                raise ValueError("检索分数必须为有限数值。")
            ranked.append({"chunk_id": chunk["id"], "file": chunk["file"],
                           "page": chunk.get("page"), "text": chunk["text"], "score": float(score),
                           "matched_evidence": [e["id"] for e in q["relevant"] if matches(chunk, e)]})
        metrics = {}
        for k in ks:
            prefix = ranked[:k]
            found = {eid for r in prefix for eid in r["matched_evidence"]}
            first = next((i for i, r in enumerate(prefix, 1) if r["matched_evidence"]), None)
            metrics[str(k)] = {
                "recall": len(found) / len(q["relevant"]) if q["answerable"] else None,
                "hit": int(bool(found)) if q["answerable"] else None,
                "reciprocal_rank": (1 / first if first else 0) if q["answerable"] else None,
                "matched_evidence": sorted(found),
                "missed_evidence": [e for e in q["relevant"] if e["id"] not in found],
                "returned_chunks": len(prefix),
            }
        rows.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                     "unmatchable_evidence": absent, "metrics": metrics, "retrieved": ranked})
    answerable = [r for r in rows if r["answerable"]]
    summary = {}
    for k in ks:
        summary[str(k)] = {
            field: sum(r["metrics"][str(k)][source] for r in answerable) / len(answerable) if answerable else None
            for field, source in (("recall", "recall"), ("hit_rate", "hit"), ("mrr", "reciprocal_rank"))
        }
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}, "question_count": len(rows),
            "answerable_count": len(answerable), "unanswerable_count": len(rows) - len(answerable),
            "questions_with_unmatchable_evidence": sum(bool(r["unmatchable_evidence"]) for r in rows),
            "summary": summary, "details": rows}


def fingerprint(files):
    # 带长度的字段消除文件拼接歧义，排序使上传顺序不影响内容指纹。
    digest = hashlib.sha256()
    for name, data in sorted(files):
        for field in (name.encode(), data):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def to_csv(report):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["question_id", "question", "answerable", "k", "recall", "hit", "reciprocal_rank",
                     "returned_chunks", "missed_evidence_ids", "unmatchable_evidence_ids"])
    def safe(value):
        # 防止用户题目被表格应用作为公式执行。
        return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
    for row in report["details"]:
        for k, metric in row["metrics"].items():
            writer.writerow([safe(row["id"]), safe(row["question"]), row["answerable"], k,
                             metric["recall"], metric["hit"], metric["reciprocal_rank"], metric["returned_chunks"],
                             safe(";".join(e["id"] for e in metric["missed_evidence"])),
                             safe(";".join(row["unmatchable_evidence"]))])
    return output.getvalue()
