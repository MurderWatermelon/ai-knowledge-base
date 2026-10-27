import csv
import io
import json
import math
from pathlib import Path
import unittest

from evaluation import evaluate, fingerprint, load_dataset, to_csv
from rag import Index, build_index, read_document, split_text


def chunk(cid, text, file="notes.md", page=None):
    return {"id": cid, "text": text, "file": file, "page": page}


def question(qid="q", relevant=None, answerable=True):
    if relevant is None:
        relevant = [{"id": "a", "file": "notes.md", "quote": "证据甲"},
                    {"id": "b", "file": "notes.md", "quote": "证据乙"}]
    return {"id": qid, "question": "测试问题", "answerable": answerable, "relevant": relevant}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [chunk("noise", "无关内容"), chunk("a", "证据甲"),
                       chunk("a-overlap", "重复的证据甲"), chunk("b", "证据乙")]

    def run_eval(self, questions=None, ks=(1, 2, 3, 4), results=None):
        ranked = [(c, 1.0 / (i + 1)) for i, c in enumerate(self.corpus)] if results is None else results
        return evaluate(questions or [question()], lambda q, k: ranked[:k], self.corpus, ks)

    def test_hand_computed_metrics_and_duplicate_overlap(self):
        report = self.run_eval()
        self.assertEqual(report["summary"]["1"], {"recall": 0, "hit_rate": 0, "mrr": 0})
        self.assertEqual(report["summary"]["2"], {"recall": .5, "hit_rate": 1, "mrr": .5})
        self.assertEqual(report["summary"]["3"]["recall"], .5)
        self.assertEqual(report["summary"]["4"], {"recall": 1, "hit_rate": 1, "mrr": .5})

    def test_macro_average_and_no_answer_exclusion(self):
        questions = [question(), question("miss", [{"id": "x", "file": "notes.md", "quote": "缺失证据"}]),
                     question("unknown", [], False)]
        report = self.run_eval(questions, ks=[2])
        self.assertEqual(report["summary"]["2"], {"recall": .25, "hit_rate": .5, "mrr": .25})
        self.assertEqual(report["answerable_count"], 2)
        self.assertEqual(report["unanswerable_count"], 1)
        self.assertIsNone(report["details"][2]["metrics"]["2"]["recall"])
        self.assertEqual(report["questions_with_unmatchable_evidence"], 1)
        self.assertEqual(report["details"][1]["unmatchable_evidence"], ["x"])

    def test_all_unanswerable_is_undefined_not_zero(self):
        report = self.run_eval([question("unknown", [], False)])
        self.assertEqual(report["summary"]["1"], {"recall": None, "hit_rate": None, "mrr": None})
        self.assertIn('"recall": null', json.dumps(report))

    def test_wrong_source_or_page_does_not_match(self):
        evidence = [{"id": "a", "file": "other.pdf", "page": 2, "quote": "证据甲"}]
        for c in [chunk("x", "证据甲", "notes.md", 2), chunk("x", "证据甲", "other.pdf", 1)]:
            report = self.run_eval([question(relevant=evidence)], [1], [(c, .9)])
            self.assertEqual(report["summary"]["1"]["recall"], 0)
        c = chunk("x", "证据甲", "other.pdf", 2)
        report = evaluate([question(relevant=evidence)], lambda q, k: [(c, .9)], [c], [1])
        self.assertEqual(report["summary"]["1"]["recall"], 1)

    def test_whitespace_normalization(self):
        c = chunk("x", "Python 字典\n 键是唯一的")
        e = [{"id": "x", "file": "notes.md", "quote": "Python 字典 键是唯一的"}]
        report = evaluate([question(relevant=e)], lambda q, k: [(c, 1)], [c], [1])
        self.assertEqual(report["summary"]["1"]["recall"], 1)

    def test_empty_results_and_k_larger_than_corpus(self):
        report = self.run_eval(results=[], ks=[10])
        self.assertEqual(report["summary"]["10"]["recall"], 0)
        self.assertEqual(report["details"][0]["metrics"]["10"]["returned_chunks"], 0)
        report = self.run_eval(ks=[10])
        self.assertEqual(report["details"][0]["metrics"]["10"]["returned_chunks"], 4)
        self.assertEqual(report["summary"]["10"]["recall"], 1)

    def test_invalid_ks(self):
        for ks in ([], [0], [-1], [True], [1.5], ["3"]):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                self.run_eval(ks=ks)

    def test_unmatchable_boundary_counts_as_miss(self):
        corpus = [chunk("x", "分块边"), chunk("y", "界测试")]
        q = question(relevant=[{"id": "boundary", "file": "notes.md", "quote": "分块边界测试"}])
        report = evaluate([q], lambda q, k: [(c, .8) for c in corpus], corpus, [2])
        self.assertEqual(report["summary"]["2"]["recall"], 0)
        self.assertEqual(report["details"][0]["unmatchable_evidence"], ["boundary"])

    def test_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError):
            self.run_eval(results=[(self.corpus[0], math.nan)])

    def test_loader_rejects_bad_annotations(self):
        good = {"version": 1, "questions": [question()]}
        mutations = [
            lambda d: d.update(version=2),
            lambda d: d.update(questions=[]),
            lambda d: d["questions"].append(d["questions"][0]),
            lambda d: d["questions"][0].update(answerable="true"),
            lambda d: d["questions"][0].update(relevant=[]),
            lambda d: d["questions"][0]["relevant"][0].update(quote=" "),
            lambda d: d["questions"][0]["relevant"][0].update(page=True),
            lambda d: d["questions"][0]["relevant"].append(d["questions"][0]["relevant"][0]),
            lambda d: d["questions"][0]["relevant"].append({"id": "another", "file": "notes.md", "quote": "证据甲"}),
        ]
        for mutate in mutations:
            data = json.loads(json.dumps(good))
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                load_dataset(json.dumps(data))
        with self.assertRaises(ValueError):
            load_dataset(b"not json")

    def test_csv_roundtrip_and_formula_escape(self):
        q = question("=danger")
        q["question"] = "多行问题\n带,逗号"
        report = self.run_eval([q], ks=[1])
        rows = list(csv.DictReader(io.StringIO(to_csv(report))))
        self.assertEqual(rows[0]["question_id"], "'=danger")
        self.assertEqual(rows[0]["question"], q["question"])
        self.assertEqual(rows[0]["missed_evidence_ids"], "a;b")

    def test_fingerprint_content_and_order(self):
        a, b = ("a", b"1"), ("b", b"2")
        self.assertEqual(fingerprint([a, b]), fingerprint([b, a]))
        self.assertNotEqual(fingerprint([a]), fingerprint([("a", b"2")]))
        self.assertNotEqual(fingerprint([("ab", b"c")]), fingerprint([("a", b"bc")]))


class RetrievalTests(unittest.TestCase):
    def test_chunking_overlap_and_validation(self):
        self.assertEqual(split_text("abcdefgh", 4, 1), ["abcd", "defg", "gh"])
        self.assertEqual(split_text("  \n", 4, 1), [])
        for size, overlap in [(0, 0), (4, 4), (4, -1), (4, 5)]:
            with self.assertRaises(ValueError):
                split_text("hello", size, overlap)

    def test_text_parser_and_ids(self):
        a = read_document("a.md", "\ufeff甲 乙\n丙".encode(), 4, 0)
        b = read_document("a.md", "\ufeff甲 乙\n丙".encode(), 4, 0)
        self.assertEqual(a, b)
        self.assertEqual(a[0]["text"], "甲 乙 ")
        self.assertIsNone(a[0]["page"])
        with self.assertRaises(ValueError):
            read_document("a.exe", b"abc")

    def test_shared_retriever_normalization_ranking_and_ties(self):
        class FakeEmbedder:
            def encode(self, texts):
                mapping = {"a": [3., 0.], "b": [0., 9.], "c": [6., 0.], "query": [8., 0.]}
                return [mapping[t] for t in texts]
        index = Index([chunk("a", "a"), chunk("b", "b"), chunk("c", "c")], FakeEmbedder())
        results = index.retrieve("query", 10)
        self.assertEqual([c["id"] for c, _ in results], ["a", "c", "b"])
        self.assertEqual([score for _, score in results], [1., 1., 0.])
        with self.assertRaises(ValueError):
            index.retrieve("query", 0)

    def test_duplicates_empty_and_invalid_utf8(self):
        for files in [[], [("a.txt", b"")], [("a.txt", b"a"), ("a.txt", b"b")], [("a.txt", b"\xff")]]:
            with self.subTest(files=files), self.assertRaises(ValueError):
                build_index(files, backend="lexical")

    def test_actual_offline_retrieval_and_sample_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        files = [(p.name, p.read_bytes()) for p in sorted((root / "examples/documents").glob("*.md"))]
        questions = load_dataset((root / "examples/evaluation.json").read_bytes())
        index = build_index(files, 140, 40, "lexical")
        results = index.retrieve("Python 字典的键有什么要求？", 3)
        self.assertEqual(results[0][0]["file"], "python_notes.md")
        report = evaluate(questions, index.retrieve, index.chunks, [1, 3, 100])
        self.assertEqual(report["answerable_count"], 8)
        self.assertEqual(report["unanswerable_count"], 2)
        self.assertEqual(report["summary"]["100"]["recall"], 1)
        self.assertEqual(report["questions_with_unmatchable_evidence"], 0)
        for smaller, larger in [("1", "3"), ("3", "100")]:
            self.assertLessEqual(report["summary"][smaller]["recall"], report["summary"][larger]["recall"])


if __name__ == "__main__":
    unittest.main()
