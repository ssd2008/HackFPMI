"""Each worker owns its MuPDF context and lazily loads the local visual encoder."""

from functools import lru_cache

from .attachments import describe_file, prepare_row
from .live_vision import FOLDER, LiveCaptioner


@lru_cache(maxsize=1)
def captioner():
    return LiveCaptioner(FOLDER) if (FOLDER / "labels.npz").is_file() else None


def extract(source, files, tessdata):
    return prepare_row(source, files, tessdata, captioner()).get("attachments", [])


def extract_bytes(name, mime, payload, tessdata):
    try:
        item = describe_file(name, mime, payload, tessdata, captioner())
    except Exception:
        # Preserve the filename when an individual visitor file cannot be read.
        item = {"filename": name, "mime_type": mime, "description": "", "partial": True}
    return [item]
