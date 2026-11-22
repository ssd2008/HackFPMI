import io
from pathlib import Path
import unittest
from zipfile import ZipFile

from mailsearch.attachments import describe_file

TESS = Path(".cache/tessdata").resolve()


def archive(parts):
    buf = io.BytesIO()
    with ZipFile(buf, "w") as z:
        for name, content in parts.items():
            z.writestr(name, content)
    return buf.getvalue()


class FormatTests(unittest.TestCase):
    def test_office_content(self):
        for ext in ("odt", "ods", "odp"):
            raw = archive(
                {
                    "content.xml": '<r xmlns:t="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><t:p>Русский текст</t:p></r>'
                }
            )
            self.assertEqual(
                describe_file("a." + ext, None, raw, TESS, None)["description"],
                "Русский текст",
            )
        raw = archive(
            {
                "xl/sharedStrings.xml": "<sst><si><t>Билеты</t></si></sst>",
                "xl/worksheets/sheet1.xml": '<r><c t="s"><v>0</v></c><c t="inlineStr"><is><t>поезд</t></is></c><c><v>123</v></c></r>',
            }
        )
        self.assertEqual(
            describe_file("a.xlsx", None, raw, TESS, None)["description"],
            "Билеты поезд 123",
        )

    @unittest.skipUnless((TESS / "rus.traineddata").exists(), "optional OCR models")
    def test_raster_text_in_mixed_pdf_presentation_and_multipage_tiff(self):
        import pymupdf
        from PIL import Image

        def scan(word):
            with pymupdf.open() as doc:
                p = doc.new_page(width=400, height=90)
                p.insert_text((20, 55), word, fontsize=28)
                return p.get_pixmap().tobytes("png")

        pic = scan("CHECKPOINT")
        with pymupdf.open() as doc:
            p = doc.new_page(width=400, height=150)
            p.insert_text((10, 14), "Native heading")
            p.insert_image(pymupdf.Rect(0, 30, 400, 120), stream=pic)
            text = describe_file("a.pdf", None, doc.tobytes(), TESS, None)[
                "description"
            ]
            self.assertIn("Native heading", text)
            self.assertIn("CHECKPOINT", text)
        rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        for kind, part, media in (
            ("pptx", "ppt/slides/slide1.xml", "../media/a.png"),
            ("docx", "word/document.xml", "media/a.png"),
        ):
            directory, filename = part.rsplit("/", 1)
            raw = archive(
                {
                    part: f'<r xmlns:q="{rel}"><blip q:embed="rId1"/></r>',
                    directory
                    + "/_rels/"
                    + filename
                    + ".rels": f'<Relationships><Relationship Id="rId1" Type="{rel}/image" Target="{media}"/></Relationships>',
                    part.split("/")[0] + "/media/a.png": pic,
                }
            )
            self.assertIn(
                "CHECKPOINT",
                describe_file("a." + kind, None, raw, TESS, None)["description"],
            )
        buf = io.BytesIO()
        with (
            Image.open(io.BytesIO(pic)) as first,
            Image.open(io.BytesIO(scan("1234567890"))) as second,
        ):
            first.save(buf, format="TIFF", save_all=True, append_images=[second])
        text = describe_file("a.tif", None, buf.getvalue(), TESS, None)["description"]
        self.assertIn("CHECKPOINT", text)
        self.assertIn("1234567890", text)
