"""Questions about a specific message keep their constraints through answer generation."""
import unittest
from unittest.mock import patch

from mailsearch import generation as rag


class QuestionIntentTests(unittest.TestCase):
    def test_followup_question_is_not_a_name_or_a_request_for_a_summary(self):
        query = "Найди квитанцию Андрею Кедрову. Сколько он заплатил?"
        self.assertTrue(rag._is_related(query, "Квитанция Андрея Кедрова. Итого 345 рублей."))
        self.assertFalse(rag._is_related(query, "Квитанция Андрея Липова. Итого 345 рублей."))
        self.assertFalse(rag._is_related(query, "Квитанция Игоря Кедрова. Итого 345 рублей."))
        self.assertFalse(rag._list_request(query))
        self.assertEqual(rag._question_parts(query), ["Сколько он заплатил?"])
        self.assertTrue(rag._list_request("Покажи квитанции за подписку."))

    def test_explicit_names_titles_and_unit_values_remain_anchors(self):
        request = "Найди договор. Алиса Морозова его подписала?"
        self.assertFalse(rag._is_related(request, "Договор подписала Вера Морозова."))
        self.assertFalse(rag._is_related("Когда откроется «Дальний мыс»?", "Открытие кафе «Южный ветер» уже скоро."))
        query = "Найди насос 1,5 кВт. Какая цена?"
        self.assertTrue(rag._is_related(query, "Приобретен насос 1.5 кВт стоимостью 8400 руб."))
        self.assertFalse(rag._is_related(query, "Приобретен насос 11.5 кВт стоимостью 8400 руб."))
        self.assertFalse(rag._supported("Стоимость 910 EUR", "Стоимость 910, валюта не указана."))
        self.assertFalse(rag._supported("Стоимость 999 RUB", "Стоимость 910 RUB."))
        self.assertFalse(rag._supported("Договор с поставщиком подписала Алиса Морозова.",
                                        "Договор с поставщиком подписала Вера Морозова."))

    def test_separated_instructions_and_total_fit_one_cited_answer(self):
        how = "Чтобы получить пропуск, предъявите фотографию."
        total = "Всего осталось 23 заявки."
        body = how + " " + ("Список сверили с архивом. " * 100) + total
        doc = {"doc_id": "note", "doc": body, "doc_dt": None}
        query = "Как получить пропуск и сколько всего осталось заявок?"
        passage = rag._passage(doc, query)
        self.assertLessEqual(len(passage), rag.MAX_PASSAGE_CHARS)
        self.assertIn(how, passage)
        self.assertIn(total, passage)
        with patch.object(rag, "_generate", return_value=None):
            reply = rag.answer_question(query, [doc])
        self.assertEqual(reply["sources"], ["note"])
        self.assertIn(how, reply["answer"])
        self.assertIn(total, reply["answer"])
        self.assertIn("[1]", reply["answer"])
        other = {"doc_id": "older", "doc": how + " Всего осталось 16 заявок.", "doc_dt": None}
        with patch.object(rag, "_generate", return_value=[how, "Всего осталось 16 заявок."]):
            partial = rag.answer_question(query, [doc, other])
        self.assertEqual(partial["mode"], "extractive")
        self.assertEqual(partial["sources"], ["note"])
        self.assertIn(total, partial["answer"])


if __name__ == "__main__":
    unittest.main()
