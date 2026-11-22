import tempfile
import unittest
import zipfile
from email.message import EmailMessage
from pathlib import Path

from mailsearch.attachments import prepare_row, searchable_documents


class AttachmentTests(unittest.TestCase):
    def test_mail_without_files_keeps_existing_index_text(self):
        row = {"doc": "От: Папа\nТема: Фотография\nПривет"}
        self.assertEqual(searchable_documents([row]), [row])

    def test_mime_file_searchable_without_changing_envelope_or_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mail = EmailMessage()
            mail.set_content("Привет")
            mail.add_attachment(
                "Опись книг".encode(),
                maintype="text",
                subtype="plain",
                filename="Каталог.txt",
            )
            (root / "message.eml").write_bytes(mail.as_bytes())
            body = "От: Библиотека\nТема: Архив\nПривет"
            row = prepare_row(
                {"doc": body, "eml_path": "message.eml"}, root, root, None
            )
            self.assertEqual(row["doc"], body)
            self.assertNotIn("eml_path", row)
            text = searchable_documents([row])[0]["doc"]
            self.assertTrue(text.startswith("От: Библиотека\nТема: Архив\n"))
            self.assertIn("Каталог.txt. Опись книг", text)

    def test_docx_extracts_text_from_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with zipfile.ZipFile(root / "file.docx", "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:t>Договор</w:t></w:document>',
                )
            row = prepare_row(
                {"attachments": [{"path": "file.docx"}]}, root, root, None
            )
            self.assertEqual(row["attachments"][0]["description"], "Договор")

    def test_path_cannot_escape_file_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ValueError):
                prepare_row({"attachments": [{"path": "../outside"}]}, root, root, None)
