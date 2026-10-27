import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation import evaluate, load_dataset
from rag import (Index, QUERY_INSTRUCTION, SemanticEmbedder, build_index,
                 read_with_local_model, validate_reading_response)


class SemanticTests(unittest.TestCase):
    def test_chinese_instruction_is_only_added_to_question(self):
        class Model:
            def __init__(self):
                self.calls = []

            def encode(self, texts, **kwargs):
                self.calls.append(texts)
                return [[1., 0.] for _ in texts]

        model = Model()
        index = Index([{"id": "a", "file": "a.md", "page": None, "text": "在七日内可退货"}], SemanticEmbedder(model))
        results = index.retrieve("收到第五天了，还能退吗？", 1)
        self.assertEqual(model.calls, [["在七日内可退货"], [QUERY_INSTRUCTION + "收到第五天了，还能退吗？"]])
        self.assertEqual(results[0][0]["id"], "a")
        for question in ("", "  ", None):
            with self.assertRaises(ValueError):
                index.retrieve(question)

    def test_paraphrase_dataset_is_valid_and_all_evidence_exists(self):
        root = Path(__file__).resolve().parents[1]
        files = [(p.name, p.read_bytes()) for p in sorted((root / "examples/reading_documents").glob("*.md"))]
        questions = load_dataset((root / "examples/paraphrase_evaluation.json").read_bytes())
        index = build_index(files, backend="lexical")
        report = evaluate(questions, index.retrieve, index.chunks, [1, 3, 100])
        self.assertEqual(report["answerable_count"], 10)
        self.assertEqual(report["unanswerable_count"], 2)
        self.assertEqual(report["questions_with_unmatchable_evidence"], 0)
        self.assertEqual(report["summary"]["100"]["recall"], 1)


class ReadingTests(unittest.TestCase):
    def setUp(self):
        self.results = [({"id": "a", "file": "rules.pdf", "page": 2,
                          "text": "普通商品在签收后七日内可以退货。定制商品不适用。"}, .8)]
        self.valid = {"status": "answered", "answer": "普通款签收五天可以申请退货。",
                      "explanation": "五天在七日期限内。",
                      "citations": [{"source": 1, "quote": "普通商品在签收后七日内可以退货。"}]}

    def test_valid_reading_keeps_real_source_and_page(self):
        output = validate_reading_response(self.valid, self.results)
        self.assertEqual(output["citations"][0]["file"], "rules.pdf")
        self.assertEqual(output["citations"][0]["page"], 2)
        self.assertEqual(output["answer"], self.valid["answer"])

    def test_fabricated_or_out_of_range_citation_rejects_entire_answer(self):
        for citation in [{"source": 1, "quote": "所有商品无条件退货"},
                         {"source": 2, "quote": "定制商品不适用。"},
                         {"source": True, "quote": "定制商品不适用。"},
                         {"source": 1, "quote": " "}]:
            value = {**self.valid, "citations": [citation]}
            with self.subTest(citation=citation), self.assertRaises(ValueError):
                validate_reading_response(value, self.results)

    def test_missing_answer_explanation_or_citation_rejected(self):
        for field, replacement in [("answer", ""), ("explanation", None), ("citations", []), ("citations", [None])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reading_response({**self.valid, field: replacement}, self.results)

    def test_duplicate_citations_count_once(self):
        value = {**self.valid, "citations": self.valid["citations"] * 2}
        self.assertEqual(len(validate_reading_response(value, self.results)["citations"]), 1)

    def test_insufficient_cannot_leak_unsupported_answer(self):
        value = {"status": "insufficient", "answer": "编造答案", "citations": self.valid["citations"]}
        output = validate_reading_response(value, self.results)
        self.assertEqual(output["citations"], [])
        self.assertNotIn("编造", output["answer"])

    def test_malformed_status_rejected(self):
        for value in ([], None, {}, {"status": []}, {"status": "yes"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_reading_response(value, self.results)

    @patch("rag.urlopen")
    def test_no_results_does_not_call_model(self, mocked):
        self.assertEqual(read_with_local_model("问题", [])["status"], "insufficient")
        mocked.assert_not_called()

    @patch("rag.urlopen")
    def test_ollama_contract_and_validated_response(self, mocked):
        mocked.return_value = io.BytesIO(json.dumps({"message": {"content": json.dumps(self.valid)}}).encode())
        output = read_with_local_model("第五天还能退吗？", self.results)
        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["temperature"], 0)
        material = json.loads(payload["messages"][1]["content"])
        self.assertEqual(material["materials"][0]["source"], 1)
        self.assertEqual(material["question"], "第五天还能退吗？")
        self.assertEqual(output["citations"][0]["page"], 2)

    @patch("rag.urlopen")
    def test_model_non_json_response_is_clear_error(self, mocked):
        mocked.return_value = io.BytesIO(json.dumps({"message": {"content": "不是 JSON"}}).encode())
        with self.assertRaisesRegex(ValueError, "未返回有效"):
            read_with_local_model("问题", self.results)


if __name__ == "__main__":
    unittest.main()
