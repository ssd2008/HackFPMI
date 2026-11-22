"""Prepare local attachment descriptions before indexing a corpus.

Input rows may reference files under --files via attachments[].path, or an
RFC 822 message via eml_path. The server only sees the prepared JSONL.
"""

import argparse
import json
import mimetypes
import os
import posixpath
import re
import tempfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from xml.etree import ElementTree

MAX_BYTES = 24 * 1024 * 1024
MAX_PAGES = 5


def searchable_documents(rows):
    result = []
    for row in rows:
        evidence = []
        for item in row.get("attachments", []):
            name = item.get("filename", "")
            description = item.get("description", "")
            if name or description:
                evidence.append(f"Вложение {name}. {description}")
        if evidence:
            # Keep the envelope and attachment evidence inside the dense token budget.
            lines = row["doc"].splitlines(keepends=True)
            index = 0
            while index < len(lines) and re.match(
                r"^\s*(?:от|отправитель|тема)\s*:", lines[index], re.IGNORECASE
            ):
                index += 1
            body = "".join(lines[index:])
            text = (
                "".join(lines[:index]).rstrip()
                + "\n"
                + "\n".join(evidence)
                + "\n"
                + body
            )
            result.append({**row, "doc": text})
        else:
            result.append(row)
    return result


def read_local(root, relative, maximum):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"Путь за пределами каталога вложений: {relative}")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"Файл больше лимита: {relative}")
    return data


class VisualCaptioner:
    def __init__(self, folder):
        self.folder = folder
        self.model = None

    def __call__(self, image):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(
                self.folder, local_files_only=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.folder, local_files_only=True, dtype=torch.float32
            ).eval()
            torch.set_num_threads(min(4, torch.get_num_threads()))
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Describe the visible objects or type of document briefly in English. Do not transcribe personal details.",
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        batch = self.processor(
            text=prompt, images=[image.convert("RGB")], return_tensors="pt"
        )
        with torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=40, do_sample=False)
        return self.processor.decode(
            output[0, batch["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


OFFICE_MIME = {
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
}


def office_text(data, kind):
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(data)) as archive:
        remaining = MAX_BYTES

        def xml(name):
            nonlocal remaining
            info = archive.getinfo(name)
            remaining -= info.file_size
            if remaining < 0:
                raise ValueError("Слишком большой XML документа")
            return ElementTree.fromstring(archive.read(info))

        def numbered(pattern):
            found = [
                (int(m[1]), name)
                for name in archive.namelist()
                if (m := re.fullmatch(pattern, name))
            ]
            return [name for _, name in sorted(found)]

        if kind in (".odt", ".odp", ".ods"):
            return " ".join(
                "".join(p.itertext())
                for p in xml("content.xml").iter()
                if p.tag
                in (
                    "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p",
                    "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}h",
                )
            )
        if kind == ".pptx":
            return " ".join(
                t.text or ""
                for name in numbered(r"ppt/slides/slide(\d+)\.xml")
                for t in xml(name).findall(".//{*}t")
            )

        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(t.text or "" for t in s.findall(".//{*}t"))
                for s in xml("xl/sharedStrings.xml").findall("{*}si")
            ]
        values = []
        for name in numbered(r"xl/worksheets/sheet(\d+)\.xml"):
            for cell in xml(name).findall(".//{*}c"):
                value = cell.findtext("{*}v", "")
                if cell.get("t") == "s":
                    value = (
                        strings[int(value)]
                        if value.isdecimal() and int(value) < len(strings)
                        else ""
                    )
                elif cell.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//{*}t"))
                values.append(value)
        return " ".join(values)


def office_picture_text(data, kind, tessdata):
    """OCR the first embedded raster pictures actually referenced in the document."""
    from io import BytesIO

    embed = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    with zipfile.ZipFile(BytesIO(data)) as archive:
        names = set(archive.namelist())
        if not any(n.startswith(("word/media/", "ppt/media/")) for n in names):
            return ""
        if kind == ".docx":
            parts = ["word/document.xml"]
        else:
            parts = sorted(
                (n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                key=lambda n: int(re.search(r"(\d+)\.xml$", n)[1]),
            )
        remaining = MAX_BYTES

        def read(name):
            nonlocal remaining
            remaining -= archive.getinfo(name).file_size
            if remaining < 0:
                raise ValueError("Слишком большие встроенные данные")
            return archive.read(name)

        passages, seen = [], set()
        for part in parts:
            folder, filename = posixpath.split(part)
            rels = posixpath.join(folder, "_rels", filename + ".rels")
            if rels not in names:
                continue
            links = {
                r.get("Id"): r.get("Target", "")
                for r in ElementTree.fromstring(read(rels))
                if r.get("Type", "").endswith("/image")
                and r.get("TargetMode") != "External"
            }
            for blip in ElementTree.fromstring(read(part)).findall(".//{*}blip"):
                target = links.get(blip.get(embed))
                if not target:
                    continue
                name = posixpath.normpath(posixpath.join(folder, target)).lstrip("/")
                if (
                    name in seen
                    or name not in names
                    or Path(name).suffix.lower()
                    not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
                ):
                    continue
                seen.add(name)
                item = describe_file(
                    name, "image/" + Path(name).suffix[1:], read(name), tessdata, None
                )
                passages.append(item["description"])
                if len(seen) >= MAX_PAGES:
                    return " ".join(passages)
        return " ".join(passages)


def describe_file(name, mime, data, tessdata, captioner):
    from io import BytesIO

    if len(data) > MAX_BYTES:
        raise ValueError(f"Слишком большое вложение: {name}")
    mime = (mime or "application/octet-stream").split(";")[0].lower()
    ext = Path(name).suffix.lower()
    text, caption = "", ""
    methods = []
    partial = False
    if data.startswith(b"%PDF-") or mime == "application/pdf":
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            passages = []
            for page_index in range(min(MAX_PAGES, len(pdf))):
                page = pdf[page_index]
                value = page.get_text()
                if not value.strip():
                    value = ocr_page(page, tessdata)
                    methods.append("ocr")
                elif page.get_image_info():
                    # Preserve the native layer (including its reading order) and add
                    # text found in raster regions, even on pages with a footer.
                    try:
                        combined = ocr_page(page, tessdata, full=False)
                    except (RuntimeError, ValueError):
                        combined = ""
                        partial = True
                    existing = {line.strip() for line in value.splitlines()}
                    extra = [
                        line
                        for line in combined.splitlines()
                        if line.strip() not in existing
                    ]
                    if extra:
                        value += "\n" + "\n".join(extra)
                        methods.append("ocr")
                passages.append(value[:3000])
            text = " ".join(passages)
        methods.append("pdf")
    elif mime.startswith("image/") or ext in (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
    ):
        import pymupdf
        from PIL import Image, ImageOps

        passages, captions = [], []
        with Image.open(BytesIO(data)) as source:
            count = min(MAX_PAGES, source.n_frames) if source.format == "TIFF" else 1
            for index in range(count):
                if index:
                    source.seek(index)
                if source.width * source.height > 30_000_000:
                    raise ValueError(f"Слишком большое изображение: {name}")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((2200, 2200))
                buf = BytesIO()
                image.save(buf, format="PNG", compress_level=1)
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=image.width, height=image.height)
                    page.insert_image(page.rect, stream=buf.getvalue())
                    passages.append(ocr_page(page, tessdata, dpi=96))
                if captioner:
                    label = captioner(image)
                    if label and label not in captions:
                        captions.append(label)
        text, caption = " ".join(passages), " ".join(captions)
        methods.append("ocr")
        if caption:
            methods.append("vision")
    elif ext in (".pptx", ".xlsx", ".odt", ".odp", ".ods") or mime in OFFICE_MIME:
        kind = OFFICE_MIME.get(mime, ext)
        text = office_text(data, kind)
        methods.append(kind.lstrip("."))
    elif (
        ext == ".docx"
        or mime
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        with zipfile.ZipFile(BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_BYTES:
                raise ValueError(f"Слишком большой документ: {name}")
            root = ElementTree.fromstring(archive.read(info))
            tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
            text = " ".join(node.text or "" for node in root.iter(tag))
        methods.append("docx")
    elif mime.startswith("text/") or ext in (".txt", ".csv", ".md"):
        text = data.decode("utf-8-sig", errors="replace")
        methods.append("text")
    if "pptx" in methods or "docx" in methods:
        try:
            picture_text = office_picture_text(
                data, ".pptx" if "pptx" in methods else ".docx", tessdata
            )
        except (OSError, ValueError, RuntimeError, KeyError, ElementTree.ParseError):
            picture_text = ""
            partial = True
        if picture_text.strip():
            text += " " + picture_text
            methods.append("ocr")
    excerpt = " ".join(text.split())[:1400]
    description = " ".join(v for v in (caption, excerpt) if v)
    return {
        "filename": name,
        "mime_type": mime,
        "description": description,
        "methods": methods or ["filename"],
        **({"partial": True} if partial else {}),
    }


def ocr_page(page, tessdata, dpi=150, full=True):
    # Bound the raster size even for PDFs with an unusually large page box.
    dpi = min(dpi, int(72 * 2500 / max(1, page.rect.width, page.rect.height)))
    if dpi < 1:
        raise ValueError("Слишком большая страница для OCR")
    textpage = page.get_textpage_ocr(
        language="rus+eng", dpi=dpi, full=full, tessdata=str(tessdata)
    )
    return page.get_text(textpage=textpage)


def prepare_row(row, root, tessdata, captioner):
    result = []
    for item in row.get("attachments", []):
        if "path" not in item:
            result.append(
                {
                    k: item[k]
                    for k in (
                        "filename",
                        "mime_type",
                        "description",
                        "methods",
                        "partial",
                    )
                    if k in item
                }
            )
            continue
        name = item.get("filename") or Path(item["path"]).name
        data = read_local(root, item["path"], MAX_BYTES)
        result.append(
            describe_file(
                name,
                item.get("mime_type") or mimetypes.guess_type(name)[0],
                data,
                tessdata,
                captioner,
            )
        )
    if row.get("eml_path"):
        message = BytesParser(policy=policy.default).parsebytes(
            read_local(root, row["eml_path"], 2 * MAX_BYTES)
        )
        for part in message.walk():
            if part.is_multipart() or not (
                part.get_filename() or part.get_content_disposition() == "attachment"
            ):
                continue
            result.append(
                describe_file(
                    part.get_filename() or "attachment",
                    part.get_content_type(),
                    part.get_payload(decode=True) or b"",
                    tessdata,
                    captioner,
                )
            )
    out = {k: v for k, v in row.items() if k not in ("eml_path", "attachments")}
    if result:
        out["attachments"] = result
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents", required=True, type=Path)
    p.add_argument("--files", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tessdata", type=Path, default=Path(".cache/tessdata"))
    p.add_argument("--vision-model", type=Path, default=Path(".cache/attachment-vlm"))
    p.add_argument(
        "--ocr-only", action="store_true", help="Skip visual captions for photographs"
    )
    args = p.parse_args()
    if args.documents.resolve() == args.output.resolve():
        p.error("Выбери отдельный выходной файл")
    captioner = None if args.ocr_only else VisualCaptioner(args.vision_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with args.documents.open() as src, temp.open("w") as dst:
            for line in src:
                if line.strip():
                    row = prepare_row(
                        json.loads(line), args.files.resolve(), args.tessdata, captioner
                    )
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
