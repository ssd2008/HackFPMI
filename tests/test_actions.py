from datetime import date
import unittest

from mailsearch.actions import extract_action
from mailsearch.store import MailStore


def mail(id_, text, received="2027-12-19", mailbox="first"):
    return {"doc_id": id_, "doc": text, "doc_dt": received, "synthetic_user": mailbox}


class ActionExtractionTests(unittest.TestCase):
    def test_deadlines_in_both_time_positions(self):
        for text, day, clock in (
            ("Отгрузите заказ до 19.12.2027 18:00.", "2027-12-19", "18:00"),
            ("Примите решение по заявке до 9:15 мск 20.12.2027.", "2027-12-20", "09:15"),
            ("Пожалуйста, пришлите анкету до 22 декабря 2027 в 17:00.", "2027-12-22", "17:00"),
            ("Подпишите акт до завтра.", "2027-12-20", None),
            ("Подпишите акт до 2028-01-02.", "2028-01-02", None),
        ):
            with self.subTest(text=text):
                result = extract_action(text, "2027-12-19")
                self.assertEqual((result.deadline, result.deadline_time), (day, clock))
                self.assertIn(result.evidence, text)

    def test_ignores_notifications_negations_and_optional_instructions(self):
        for text in (
            "Заявка на возврат подтверждена.",
            "Если хотите продолжить, оплатите заказ до 19.12.2027.",
            "Отправлять документы не нужно.",
            "Не нужно оплачивать заказ.",
            "Не нужно отправить счёт повторно.",
            "При необходимости проверьте статус доставки.",
            "Рекомендуем отгрузить его до 19.12.2027.",
            "Условия возврата изменились. Можно подтвердить заявку в кабинете.",
            "> Оплатите счёт до завтра.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_action(text, "2027-12-19"))

    def test_request_survives_missing_or_unrelated_deadline(self):
        for text, received in (
            ("Подпишите акт. Обновление доступно до 20 декабря.", "2027-12-19"),
            ("Нужно обработать заказ. Рекомендуем отгрузить его до 20.12.2027.", "2027-12-19"),
            ("Отправьте документы до 31.02.2027.", "2027-12-19"),
            ("Пришлите бумаги до завтра.", None),
            ("Доставьте заказ до 11:00.", "2027-12-19"),
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_action(text, received).deadline)


class ActionInboxTests(unittest.TestCase):
    def test_scope_paging_and_upcoming_deadlines(self):
        store = MailStore([
            mail("today", "Отгрузите заказ до 19.12.2027 18:00."),
            mail("tomorrow", "Оплатите счёт до 20.12.2027 09:00."),
            mail("future", "Оплатите счёт до 20.12.2027.", "2027-12-20"),
            mail("later", "Обработайте заявку до 23.12.2027."),
            mail("past", "Подпишите документы до 18.12.2027.", "2027-12-17"),
            mail("undated", "Пришлите смету.", None),
            mail("complete", "Отгрузка завершена."),
            mail("other", "Оплатите счёт до 19.12.2027.", mailbox="second"),
        ])
        result = store.action_inbox("first", date(2027, 12, 19), offset=1, limit=2)
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["offset"], 1)
        self.assertEqual([item["id"] for item in result["items"]], ["today", "later"])
        self.assertEqual([item["id"] for item in result["urgent"]], ["today", "tomorrow"])
        next_day = store.action_inbox("first", date(2027, 12, 20), offset=0, limit=25)
        self.assertEqual([item["id"] for item in next_day["urgent"]], ["tomorrow", "future"])
        self.assertIsNone(next_day["items"][-1]["deadline"])
