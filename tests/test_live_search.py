from concurrent.futures import Future
from tempfile import TemporaryDirectory
from datetime import date
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from mailsearch.live import AttachmentSearch
from mailsearch.store import MailStore


class AttachmentScopeTests(TestCase):
    def test_only_visible_files_read_and_changed_files_reprocessed(self):
        rows = [
            dict(
                doc_id=did,
                synthetic_user=box,
                doc_dt=day,
                doc="От: Папа\nТема: Держи\nФайл",
                attachments=[{"path": did + ".png"}],
            )
            for did, box, day in (
                ("scan", "a", "2026-01-01"),
                ("foreign", "b", "2026-01-01"),
                ("future", "a", "2026-03-01"),
            )
        ]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "scan.png"
            file.write_bytes(b"first")
            engine = AttachmentSearch(MailStore(rows), root, root)
            self.addCleanup(engine.close)

            def submit(fn, source, files, tessdata):
                self.assertEqual(source["attachments"][0]["path"], "scan.png")
                result = Future()
                result.set_result(
                    [{"description": file.read_text(), "methods": ["ocr"]}]
                )
                return result

            with patch.object(engine.pool, "submit", side_effect=submit) as jobs:
                for query in ("first", "first"):
                    hits, _, partial = engine.search(
                        "a", query, date(2026, 2, 1), 10, []
                    )
                    self.assertEqual([did for did, _ in hits], ["scan"])
                    self.assertFalse(partial)
                self.assertEqual(jobs.call_count, 1)
                file.write_bytes(b"replacement")
                hits, _, _ = engine.search("a", "first", date(2026, 2, 1), 10, [])
                self.assertEqual(hits, [])
                self.assertEqual(jobs.call_count, 2)
