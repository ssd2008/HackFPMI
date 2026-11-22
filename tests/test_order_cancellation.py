"""An exact order reference can identify a warning or a confirmed cancellation."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mailsearch.retrieval import SearchEngine


class OrderCancellationTest(unittest.TestCase):
    day = date(2027, 6, 1)
    target = 'SYN-83f131b394'
    other = 'SYN-75fcab8812'

    def search(self, request, letters):
        docs = [dict(doc_id=f'letter-{i}', synthetic_user='example', doc_dt='2027-06-01',
                     doc=letter.format(target=self.target, other=self.other))
                for i, letter in enumerate(letters)]
        with patch("mailsearch.retrieval.DenseIndex") as encoder:
            encoder.return_value.vectors = np.zeros((len(docs), 2), dtype=np.float32)
            encoder.return_value.query.return_value = np.zeros(2, dtype=np.float32)
            engine = SearchEngine(docs, Path('.'))
        return {key for key, _ in engine.search('example', request.format(target=self.target), self.day, limit=20)}

    def test_warning_alone_is_not_confirmation(self):
        warnings = [
            'Вы получили заказ {target}. Если не успеете его обработать, нам придётся его отменить.',
            'Заказ {target} будет отменён, если не поступит оплата.',
            'Создан заказ {target}. Необработанный заказ отменят.',
        ]
        for request in ['Найди подтверждение отмены заказа {target}',
                        'Покажи сообщение, что заказ {target} уже отменили',
                        'Где письмо о том, что отменён заказ {target}?']:
            with self.subTest(request=request):
                self.assertFalse(self.search(request, warnings))

    def test_plain_reference_warning_quote_and_yes_no_question(self):
        letters = ['Заказ {target}. Если не обработать заказ, его отменят.']
        for request in ['{target}', 'Заказ {target} уже отменили?', 'Отменили ли заказ {target}',
                        'Покажи предупреждение о возможной отмене заказа {target}',
                        'По заказу {target} писали «его отменят»',
                        'Найди отмену доставки по заказу {target}',
                        'Найди заявку на отмену заказа {target}', 'Покажи правила отмены заказа {target}']:
            with self.subTest(request=request):
                self.assertEqual(self.search(request, letters), {'letter-0'})

    def test_confirmed_unknown_and_mixed_statuses_survive(self):
        letters = [
            'Заказ {target}. Заказ отменён по просьбе покупателя.',
            'Заказ {target} будет отменён, если не оплатить. Потом его действительно отменили.',
            'Заказ {target}. Если не оплатить, его отменят. Затем поддержку запросили ещё раз, отмена была подтверждена.',
            'Заказ {target}. Статус выясняем.',
            'Заказ {target} отменён, чтобы исправить двойное списание.',
            'Заказ {target} уже отменили, если хотите оформить новый, перейдите в каталог.',
            'Заказ {target} готов к выдаче, если доставку отменят, товар можно забрать лично.',
            'Заказ {target} будет отменён, если не оплатить, но его уже отменили по просьбе покупателя.',
        ]
        self.assertEqual(self.search('Найди письмо об отмене заказа {target}', letters),
                         {f'letter-{i}' for i in range(len(letters))})

    def test_cancellation_of_unrelated_event_does_not_confirm_order(self):
        letters = [
            'Заказ {target}. Если не оплатить заказ, его отменят. Доставка сегодня отменена.',
            'Заказ {target}. Если не оплатить заказ, его отменят. Заказ {other} отменён.',
            'Заказ {other} отменён. Заказ {target}. Если не оплатить, его отменят.',
            'Заказ {target}. Была отменена встреча. Сам заказ {target} уже отменили.',
        ]
        self.assertEqual(self.search('Покажи письмо об отмене заказа {target}', letters), {'letter-3'})

    def test_unknown_or_unrelated_conditions_do_not_create_warning(self):
        letters = ['Заказ {target}. Если потребуется, мы отменим встречу с курьером.',
                   'Заказ {target}. Кто сказал, что заказ отменён?',
                   'Проверили заказ {target} и заказ {other}: если сроки истекут, его отменят.']
        self.assertEqual(self.search('Найди подтверждение отмены заказа {target}', letters),
                         {'letter-0','letter-1','letter-2'})


if __name__ == '__main__':
    unittest.main()
