"""Convert a JSONL mail export to a separate project data directory.

Example: python3 tools/import_mail_export.py export.jsonl .cache/other-mail --timezone Europe/Moscow
Use - as the input path for a pipe. Raw input has id, date, from, subject and body.
Only the received calendar day, sender, subject and original body enter the search
corpus. For native project records (doc_id, synthetic_user, doc, doc_dt), validate
and copy the input bytes unchanged so that existing queries and relevance labels
continue to refer to the same mailboxes and documents.
"""

import argparse
from contextlib import nullcontext
from datetime import date, datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def receipt_day(value, timezone):
    if value is None or value == "":
        return None, False
    if not isinstance(value, str):
        raise ValueError("date must be a string or null")
    value = value.strip()
    if not value:
        return None, False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return date.fromisoformat(value).isoformat(), False
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        stamp = parsedate_to_datetime(value)
    if stamp.tzinfo is None:
        # A wall-clock timestamp already names a local day; there is no UTC instant
        # to convert. Do not let the execution host's timezone alter that date.
        return stamp.date().isoformat(), False
    local_day = stamp.astimezone(timezone).date()
    return local_day.isoformat(), local_day != stamp.date()


def text_field(row, name):
    value = row[name]
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def header(value):
    # Unfold multiline envelope fields, while leaving the message body untouched.
    return re.sub(r"[\r\n]+[ \t]*", " ", value)


def normalize(row, mailbox_key, box, timezone):
    raw_id, timestamp = row["id"], row["date"]
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
        raise ValueError("missing message ID")
    seed = json.dumps([mailbox_key, str(raw_id)], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    doc_id = "L-" + hashlib.sha256(seed).hexdigest()[:20]
    sender = header(text_field(row, "from"))
    subject = header(text_field(row, "subject"))
    text = f"От: {sender}\nТема: {subject}\nТекст письма: {text_field(row, 'body')}"
    day, shifted = receipt_day(timestamp, timezone)
    return {"doc_id": doc_id, "synthetic_user": box, "doc": text, "doc_dt": day}, shifted


def validate_native(row):
    if not all(isinstance(row[k], str) and row[k].strip() for k in ("doc_id", "synthetic_user")):
        raise ValueError("empty lookup ID")
    if not isinstance(row["doc"], str):
        raise ValueError("doc must be a string")
    day = row["doc_dt"]
    if day is not None and (not isinstance(day, str) or date.fromisoformat(day).isoformat() != day):
        raise ValueError("doc_dt must be YYYY-MM-DD or null")


def write_corpus(source, dest, mailbox_key, timezone):
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "documents.jsonl"
    if target.is_file() and os.path.samestat(os.fstat(source.fileno()), target.stat()):
        raise ValueError("Input and output must be different files.")
    raw_box = "box-" + hashlib.sha256(mailbox_key.encode("utf-8")).hexdigest()[:12]
    ids, boxes = set(), set()
    unknown = shifted = 0
    corpus_format = None
    staging = NamedTemporaryFile(dir=dest, prefix=".mail-import-", delete=False)
    temporary = Path(staging.name)
    try:
        with staging:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    staging.write(line)
                    continue
                try:
                    row = json.loads(line)
                    row_format = "native" if "doc_id" in row else "raw"
                    if corpus_format is not None and corpus_format != row_format:
                        raise ValueError("mixed input formats")
                    corpus_format = row_format
                    if row_format == "native":
                        validate_native(row)
                        encoded, day_shift = line, False
                    else:
                        row, day_shift = normalize(row, mailbox_key, raw_box, timezone)
                        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                    if row["doc_id"] in ids:
                        raise ValueError("duplicate message ID")
                except (ValueError, KeyError, TypeError, OverflowError):
                    # Exceptions from JSON, datetime or email parsers may contain private input.
                    raise ValueError(f"Invalid, missing or duplicate field on input line {line_number}.") from None
                ids.add(row["doc_id"])
                boxes.add(row["synthetic_user"])
                unknown += row["doc_dt"] is None
                shifted += day_shift
                staging.write(encoded)
        if not ids:
            raise ValueError("Input contains no messages.")
        # Publish the complete file atomically. Concurrent imports can never clobber
        # a corpus; re-importing exactly the same bytes is idempotent.
        try:
            os.link(temporary, target)
        except FileExistsError:
            with temporary.open("rb") as new, target.open("rb") as old:
                if hashlib.file_digest(new, "sha256").digest() != hashlib.file_digest(old, "sha256").digest():
                    raise ValueError("This directory already contains a different corpus. Choose a separate data directory.") from None
    finally:
        temporary.unlink(missing_ok=True)
    return {"format": corpus_format, "documents": len(ids), "mailboxes": len(boxes), "missing_receipt_date": unknown,
            "day_changed_by_timezone": shifted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to UTF-8 JSONL, or - for standard input.")
    parser.add_argument("destination", type=Path, help="Separate directory for documents.jsonl.")
    parser.add_argument("--mailbox-key", default="single-mailbox", help="Stable namespace for raw records. Native records retain the supplied IDs.")
    parser.add_argument("--timezone", default="UTC", help="IANA timezone used to convert offset-aware timestamps. Default: UTC. Date-only and naive timestamps keep their calendar day.")
    args = parser.parse_args()
    if not args.mailbox_key.strip():
        parser.error("Mailbox key must be nonempty.")
    try:
        timezone = ZoneInfo(args.timezone)
        source = nullcontext(sys.stdin.buffer) if args.source == "-" else Path(args.source).open("rb")
        with source as stream:
            stats = write_corpus(stream, args.destination, args.mailbox_key, timezone)
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        parser.exit(1, f"Import failed: {error}\n")
    if stats["format"] == "raw":
        stats["timezone"] = timezone.key
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
