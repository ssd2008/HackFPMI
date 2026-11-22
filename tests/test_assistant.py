from datetime import date
import unittest

from mailsearch.assistant import assist, invitation_inbox, watch_matches
from mailsearch.store import MailStore


def mail(id_, day, text, box="main"):
    return {"doc_id": id_, "doc_dt": day, "doc": text, "synthetic_user": box}


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.store = MailStore([
            mail("lecture", "2026-11-20", "Тема: Лекция\nПриглашаем вас на лекцию 26 ноября 2026 в 19:00."),
            mail("registration", "2026-11-20", "Приглашаем на концерт. Регистрация до 22 ноября 2026."),
            mail("past", "2026-11-20", "Приглашаем на встречу 20 ноября 2026."),
            mail("old", "2026-11-21", "Приглашение на ЕГЭ в личном кабинете."),
            mail("new", "2026-11-22", "Приглашение на ЕГЭ готово."),
            mail("foreign", "2026-11-21", "Приглашение на ЕГЭ готово.", "other"),
            mail("future-mark", "2026-11-22", "Важное письмо."),
            mail("task", "2026-11-20", "Примите решение до 22.11.2026."),
        ])
        self.at = date(2026, 11, 21)

    def test_upcoming_invitation_uses_event_date_not_receipt_or_registration(self):
        reply = assist(self.store, "main", self.at, "На какие мероприятия меня пригласили?")
        self.assertEqual(reply["kind"], "events")
        self.assertEqual([item["id"] for item in reply["items"]], ["lecture"])
        item = reply["items"][0]
        self.assertEqual((item["event_date"], item["event_time"]), ("2026-11-26", "19:00"))
        self.assertIn(item["evidence"], self.store.by_id["lecture"]["doc"])

    def test_watch_watermark_and_scope(self):
        query = "Найди когда придет приглашение на ЕГЭ"
        self.assertEqual(assist(self.store, "main", self.at, query)["kind"], "watch")
        baseline = watch_matches(self.store, "main", self.at, self.at, query, [])
        self.assertEqual(baseline["matched_ids"], ["old"])
        at = date(2026, 11, 22)
        watch = watch_matches(self.store, "main", at, self.at, query, baseline["matched_ids"])
        self.assertEqual(watch["matched_ids"], ["new"])

    def test_invitation_folder_filters_mailbox_and_receipt_dates_before_paging(self):
        bounds = {"date_from": "2026-11-20", "date_to": "2026-11-21"}
        page = invitation_inbox(self.store, "main", "2026-11-22", 0, 2, **bounds)
        self.assertEqual(page["total"], 4)
        self.assertEqual([item["id"] for item in page["items"]], ["old", "registration"])
        page = invitation_inbox(self.store, "main", "2026-11-22", 2, 2, **bounds)
        self.assertEqual([item["id"] for item in page["items"]], ["past", "lecture"])

    def test_watch_accepts_subject_without_assistant_command(self):
        query = "Жду приглашение на ЕГЭ"
        baseline = watch_matches(self.store, "main", self.at, self.at, query, [])
        self.assertTrue(baseline["query_valid"])
        self.assertEqual(baseline["matched_ids"], ["old"])
        matches = watch_matches(self.store, "main", date(2026, 11, 22), self.at, query, baseline["matched_ids"])
        self.assertEqual(matches["matched_ids"], ["new"])
        self.assertFalse(watch_matches(self.store, "main", self.at, self.at, "жду письмо", [])["query_valid"])

    def test_manual_marks_merge_before_pagination(self):
        reply = self.store.action_inbox("main", self.at, 0, 1, ["lecture", "foreign", "future-mark", "task"])
        self.assertEqual(reply["total"], 2)
        self.assertEqual(reply["items"][0]["id"], "task")
        self.assertTrue(reply["items"][0]["manual"])
        self.assertEqual([item["id"] for item in reply["urgent"]], ["task"])
        reply = self.store.action_inbox("main", self.at, 1, 1, ["lecture"])
        self.assertEqual(reply["items"][0]["id"], "lecture")
        self.assertIsNone(reply["items"][0]["evidence"])

    def test_automatic_candidates_use_same_scope(self):
        self.assertEqual([item["id"] for item in self.store.reminder_candidates("main", self.at.isoformat())], ["task"])
