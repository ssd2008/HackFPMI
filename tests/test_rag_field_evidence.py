"""Concrete facts and fallback citations must describe the same item or event."""
import unittest
from unittest.mock import patch

from mailsearch import generation as rag
from mailsearch.quantities import has_counted_quantity


def letter(key, text):
    return {"doc_id": key, "doc": text, "doc_dt": None}


class FieldEvidenceTests(unittest.TestCase):
    def test_count_binds_to_the_requested_noun_in_its_event(self):
        red = letter("r", "Для мастерской собрано 14 коробок. Перевозка потребует 3 рейса.")
        blue = letter("b", "Для ярмарки собрано 27 коробок. Для мастерской выделена машина.")
        with patch.object(rag, "_generate", return_value=None):
            reply = rag.answer_question("Сколько коробок собрано для мастерской?", [blue, red])
        self.assertEqual(reply["sources"], ["r"])
        self.assertIn("14 коробок", reply["answer"])
        self.assertNotIn("27", reply["answer"])

    def test_quantity_crosses_alphabets_without_crossing_unrelated_nouns(self):
        q = "Сколько модемов подготовили?"
        supply = letter("m", "Reserved 18 new USB modems for deployment.")
        furniture = letter("f", "Reserved 12 desks with modems installed.")
        self.assertTrue(has_counted_quantity(q, supply["doc"], across_scripts=True))
        self.assertFalse(has_counted_quantity(q, furniture["doc"]))
        self.assertFalse(has_counted_quantity(q, "В описи 7 модемов.", across_scripts=True))
        with patch.object(rag, "_generate", return_value=None):
            reply = rag.answer_question(q, [furniture, supply])
            self.assertEqual(reply["sources"], ["m"])
            self.assertIn("18 new USB modems", reply["answer"])
            conflicting = letter("x", "Reserved 21 new USB modems for deployment.")
            self.assertEqual(rag.answer_question(q, [supply, conflicting])["mode"], "no_evidence")

    def test_a_named_measurement_and_a_postal_address_cite_the_right_letter(self):
        reading = letter("s", "Измерения в доке. Давление: 1.73. Телефон диспетчера 555-0123.")
        site = letter("l", "Склад декораций находится по адресу улица Лесная 18.")
        with patch.object(rag, "_generate", return_value=None):
            value = rag.answer_question("Какое давление указано в измерениях из дока?", [site, reading])
            address = rag.answer_question("По какому адресу находится склад декораций?", [reading, site])
        self.assertEqual((value["sources"], address["sources"]), (["s"], ["l"]))
        self.assertIn("1.73", value["answer"])
        self.assertNotIn("555", value["answer"])
        self.assertIn("Лесная 18", address["answer"])

    def test_words_locating_a_table_do_not_select_the_neighboring_row(self):
        inventory = letter("t", "Перечень: фарфор — 17; гипс — 40; шамот — 64.")
        with patch.object(rag, "_generate", return_value=None):
            reply = rag.answer_question("Какой номер присвоен шамоту в письме с перечнем фарфора и гипса?", [inventory])
        self.assertEqual(reply["sources"], ["t"])
        self.assertIn("64", reply["answer"])
        self.assertNotIn("17", reply["answer"])

    def test_calendar_value_uses_the_event_and_preserves_a_second_question(self):
        a = letter("a", "Визит орнитолога назначен на 17 июня. Подача заявок открыта.")
        b = letter("b", "Дата отправки: 20 июня. Встреча фотографов назначена на 22 июня.")
        with patch.object(rag, "_generate", return_value=None):
            reply = rag.answer_question("На какое число назначен визит орнитолога?", [b, a])
        self.assertEqual(reply["sources"], ["a"])
        self.assertIn("17 июня", reply["answer"])
        self.assertIsNone(rag._anchored_detail("Где состоится выставка и сколько залов открыто?",
                                              [letter("c", "Для выставки открыто 4 зала.")]))
        self.assertIsNone(rag._anchored_detail("Когда построен ангар с рекордом вместимости?",
                                              [letter("d", "Вместимость: 42.")]))

    def test_tracking_number_requires_its_own_label_and_value(self):
        message = letter("a", "Открыт заказ 9335-71. Сумма заказа 109 руб.")
        with patch.object(rag, "_generate", return_value=["Трек-номер 9335-71."]) as model:
            self.assertEqual(rag.answer_question("Какой трек-номер заказа 9335-71?", [message])["mode"], "no_evidence")
            model.assert_not_called()
        message["doc"] += " Код отслеживания посылки: RK765432109NL."
        with patch.object(rag, "_generate", return_value=["9335-71"]):
            reply = rag.answer_question("Какой трек-номер заказа 9335-71?", [message])
        self.assertIn("RK765432109NL", reply["answer"])
        self.assertEqual(reply["sources"], ["a"])

    def test_price_is_bound_to_requested_item_and_currency(self):
        kettle = letter("k", "Новый заказ. В заказе походный чайник из титана. Общая сумма 85.40 EUR.")
        mower = letter("m", "Новый заказ. В заказе газонокосилка. Общая сумма 750.00 EUR.")
        q = "Сколько стоил заказ на походный чайник из титана?"
        with patch.object(rag, "_generate", return_value=["85.40 руб."]):
            answer = rag.answer_question(q, [mower, kettle])
        self.assertEqual(answer["sources"], ["k"])
        self.assertIn("85.40 EUR", answer["answer"])
        self.assertNotIn("750", answer["answer"])
        second = letter("k2", "Новый заказ. В заказе походный чайник из титана. Общая сумма 39.00 EUR.")
        with patch.object(rag, "_generate", return_value=None):
            self.assertEqual(rag.answer_question(q, [kettle, second])["mode"], "no_evidence")

    def test_other_charges_in_the_same_letter_are_not_alternative_prices(self):
        receipt = letter("r", "Стоимость абонемента 47.10 EUR. Доставка стоит 8.30 EUR.")
        with patch.object(rag, "_generate", return_value=["Абонемент стоит 47.10 EUR или 8.30 EUR."]):
            answer = rag.answer_question("Сколько стоил абонемент?", [receipt])
        self.assertEqual(answer["mode"], "extractive")
        self.assertIn("47.10 EUR", answer["answer"])
        self.assertNotIn("8.30", answer["answer"])

    def test_clock_requires_the_requested_event_and_survives_fallback(self):
        plan = letter("p", "Конференция начнётся в 10:15. Завершение в 17:00. Отправлено в 08:00.")
        q = "Во сколько начинается конференция?"
        with patch.object(rag, "_generate", return_value=["Конференция начинается в 17:00."]):
            answer = rag.answer_question(q, [plan])
        self.assertIn("10:15", answer["answer"])
        self.assertNotIn("17:00", answer["answer"])
        self.assertEqual(answer["sources"], ["p"])
        changes = letter("c", "Конференция. Завершение в 17:00.")
        self.assertEqual(rag.answer_question(q, [changes])["mode"], "no_evidence")

    def test_measurement_cannot_come_from_another_product(self):
        a = letter("a", "В заказе телескопические ручки: 70 см и 90 см.")
        b = letter("b", "В заказе лист алюминия: 80 см.")
        with patch.object(rag, "_generate", return_value=None):
            answer = rag.answer_question("Какие размеры у телескопических ручек из заказа?", [b, a])
        self.assertEqual(answer["sources"], ["a"])
        self.assertIn("70 см", answer["answer"])
        self.assertIn("90 см", answer["answer"])
        board = letter("board", "В заказе две детали серии Гранит. Длины 250 мм и 420 мм.")
        with patch.object(rag, "_generate", return_value=["250 и 420"]):
            answer = rag.answer_question("Какие длины указаны для серии Гранит?", [board])
        self.assertEqual(answer["mode"], "extractive")
        self.assertIn("250 мм и 420 мм", answer["answer"])


if __name__ == "__main__":
    unittest.main()
