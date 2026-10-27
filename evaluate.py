"""命令行批量比较分块大小与 Top-k，不调用 Ollama。"""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation import evaluate, fingerprint, load_dataset, to_csv
from rag import EMBEDDING_MODEL, SemanticEmbedder, build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=Path("examples/documents"))
    parser.add_argument("--dataset", type=Path, default=Path("examples/evaluation.json"))
    parser.add_argument("--backend", choices=["semantic", "lexical"], default="semantic")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 500, 800])
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    try:
        if not args.documents.is_dir():
            raise ValueError(f"文档目录不存在：{args.documents}")
        if any(k <= 0 for k in args.ks):
            raise ValueError("Top-k 必须为正整数。")
        if any(size <= args.overlap or args.overlap < 0 for size in args.chunk_sizes):
            raise ValueError("每个分块长度都必须大于重叠长度，重叠长度不能为负。")
        files = [(p.name, p.read_bytes()) for p in sorted(args.documents.iterdir())
                 if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}]
        if not files:
            raise ValueError("目录中没有 PDF、TXT 或 Markdown 文档。")
        data = args.dataset.read_bytes()
        questions = load_dataset(data)
        embedder = None
        if args.backend == "semantic":
            embedder = SemanticEmbedder()
        args.output.mkdir(parents=True, exist_ok=True)
        comparison = []
        for size in sorted(set(args.chunk_sizes)):
            index = build_index(files, size, args.overlap, args.backend, embedder)
            metadata = {"backend": args.backend, "embedding_model": EMBEDDING_MODEL if args.backend == "semantic" else "character-tfidf-v1",
                        "chunk_size": size, "overlap": args.overlap, "chunk_count": len(index.chunks),
                        "corpus_sha256": fingerprint(files), "dataset_sha256": hashlib.sha256(data).hexdigest()}
            report = evaluate(questions, index.retrieve, index.chunks, args.ks, metadata)
            stem = f"{args.backend}-chunk{size}-overlap{args.overlap}"
            (args.output / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (args.output / f"{stem}.csv").write_text(to_csv(report), encoding="utf-8-sig")
            for k, metrics in report["summary"].items():
                row = {**metadata, "k": int(k), **metrics,
                       "questions_with_unmatchable_evidence": report["questions_with_unmatchable_evidence"]}
                comparison.append(row)
                recall = "N/A" if metrics["recall"] is None else f"{metrics['recall']:.3f}"
                print(f"chunk={size} k={k} Recall={recall} unmatchable_questions={report['questions_with_unmatchable_evidence']}")
        (args.output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已保存到 {args.output.resolve()}")
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"评测失败：{exc}\n")


if __name__ == "__main__":
    main()
