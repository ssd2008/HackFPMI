from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from benchmark import load_benchmark
from mailsearch.app import lifespan
from mailsearch.dates import parse_date_scope
from tools.import_mail_export import write_corpus


class ReceiptDates(unittest.TestCase):
    def test_calendar_periods_across_a_year_boundary(self):
        at = date(2025, 1, 3)  # Friday
        for expression, start, end in (
            ("за прошлую календарную неделю", date(2024, 12, 23), date(2024, 12, 29)),
            ("за последний календарный месяц", date(2024, 12, 1), date(2024, 12, 31)),
            ("в минувшем календарном месяце", date(2024, 12, 1), date(2024, 12, 31)),
            ("на текущей календарной неделе", date(2024, 12, 30), at),
            ("за календарный месяц", date(2025, 1, 1), at),
            ("за последнюю неделю", date(2024, 12, 28), at),
        ):
            with self.subTest(expression=expression):
                scope = parse_date_scope("Уведомления " + expression, at)
                self.assertEqual((scope.start, scope.end), (start, end))
                self.assertEqual(scope.text, "уведомления")

    def test_price_is_not_a_year(self):
        at = date(2028, 12, 22)
        self.assertFalse(parse_date_scope("Квитанция за 2300 рублей", at).bounded)
        self.assertFalse(parse_date_scope("Чек за 2300.15", at).bounded)
        self.assertEqual(parse_date_scope("Квитанции за 2028 год", at).start, date(2028, 1, 1))

    def test_deadline_and_receipt_day_in_one_request(self):
        at = date(2028, 12, 22)
        scope = parse_date_scope("Счета со сроком оплаты до 30.12.2028, пришедшие вчера", at)
        self.assertEqual((scope.start, scope.end), (date(2028, 12, 21), date(2028, 12, 21)))
        self.assertIn("30.12.2028", scope.text)
        self.assertFalse(parse_date_scope("Заказ с доставкой на 27.12.2028", at).bounded)

    def test_literal_dates_in_subject_or_body(self):
        at = date(2026, 4, 7)
        for text in (
            "Пришедшие в марте 2026, в теме дата 15.07.2023",
            "Письмо с датой 15.07.2023 в тексте, полученное в марте 2026",
        ):
            with self.subTest(text=text):
                scope = parse_date_scope(text, at)
                self.assertEqual((scope.start, scope.end), (date(2026, 3, 1), date(2026, 3, 31)))
                self.assertIn("15.07.2023", scope.text)
        self.assertFalse(parse_date_scope("В заголовке письма указана дата 15.07.2023", at).bounded)
        self.assertEqual(parse_date_scope("Письмо от 15.07.2023", at).start, date(2023, 7, 15))


class NewCorpora(unittest.TestCase):
    def test_native_export_preserves_bytes_and_lookup_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source.jsonl"
            record = {"doc_id": "archive:mail/1", "synthetic_user": "existing-box",
                      "doc": "Две строки\u2028внутри письма", "doc_dt": None}
            original = ("\r\n" + json.dumps(record, ensure_ascii=False) + "\r\n").encode()
            raw.write_bytes(original)
            for _ in range(2):
                with raw.open("rb") as source:
                    stats = write_corpus(source, root / "out", "other-name", ZoneInfo("Europe/Moscow"))
                self.assertEqual(stats["missing_receipt_date"], 1)
                self.assertEqual((root / "out/documents.jsonl").read_bytes(), original)

    def test_raw_export_dates_and_failed_reimport(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source.jsonl"
            record = {"id": "1", "date": "2026-01-01T22:15:00Z", "from": "sender",
                      "subject": "Тема", "body": "Первая строка\nВторая строка"}
            raw.write_text(json.dumps(record) + "\n")
            with raw.open("rb") as source:
                stats = write_corpus(source, root / "out", "box-a", ZoneInfo("Europe/Moscow"))
            self.assertEqual(stats["day_changed_by_timezone"], 1)
            output = root / "out/documents.jsonl"
            good = output.read_bytes()
            self.assertEqual(json.loads(good)["doc_dt"], "2026-01-02")
            self.assertTrue(json.loads(good)["doc"].endswith(record["body"]))
            with raw.open("ab") as source:
                source.write(b"{}\n")
            with raw.open("rb") as source, self.assertRaisesRegex(ValueError, "line 2"):
                write_corpus(source, root / "out", "box-a", ZoneInfo("Europe/Moscow"))
            self.assertEqual(output.read_bytes(), good)

    def test_unjudged_inputs_and_inconsistent_judgment(self):
        with TemporaryDirectory() as directory:
            data = Path(directory)
            doc = {"doc_id": "L-abc", "doc": "Тестовое письмо", "synthetic_user": "box-a", "doc_dt": None}
            query = {"query_id": "Q1", "query": "письмо", "synthetic_user": "box-a", "query_date": "2026-12-01"}
            (data / "documents.jsonl").write_text(json.dumps(doc), encoding="utf-8")
            (data / "queries_dev.jsonl").write_text(json.dumps(query), encoding="utf-8")
            docs, queries, labels, info = load_benchmark(data)
            self.assertEqual(len(docs), 1)
            self.assertEqual(queries, [query])
            self.assertFalse(labels)
            self.assertIsNone(info["qrels_file"])
            bad = {"query_id": "Q1", "doc_id": "missing", "relevance": 1}
            (data / "qrels_dev.jsonl").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "неизвестный"):
                load_benchmark(data)


class PublicPreview(unittest.IsolatedAsyncioTestCase):
    async def test_external_demo_rejects_other_corpora_before_creating_a_store(self):
        with TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "documents.jsonl").write_text('{"doc_id":"x","synthetic_user":"a","doc_dt":null,"doc":"Test"}\n')
            state = SimpleNamespace(data_dir=data)
            with patch("mailsearch.app.frontend_origin", "https://demo.invalid"):
                with self.assertRaisesRegex(ValueError, "синтетический корпус"):
                    async with lifespan(SimpleNamespace(state=state)):
                        self.fail("A different corpus was made available for the public demo")
            self.assertFalse(hasattr(state, "store"))


if __name__ == "__main__":
    unittest.main()
