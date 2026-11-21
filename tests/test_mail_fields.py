"""Mail headers have independent semantics from quoted and forwarded message text."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mailsearch.retrieval import SearchEngine


class MailFieldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def message(doc_id, box, day, text, **kwargs):
            return dict(doc_id=doc_id, synthetic_user=box, doc_dt=day, doc=text, **kwargs)
        rows = [
            message('one', 'a', '2028-03-01', 'От: Анна Ромашкина <anna@team.test>\nТема: Итоги «Весна»\nТекст письма: Отчёт.'),
            message('other', 'a', '2028-03-02', 'From: Tom <tom@other.test>\nSubject: Re: Итоги «Весна»\n\nFrom: Анна <anna@team.test>\nпроекты от anna@team.test'),
            message('two', 'a', '2028-03-02', 'Презентация бюджета', sender='Anna <anna@team.test>', subject='Report "A" due today'),
            message('one-future', 'a', '2028-03-04', 'From: Ann <anna@team.test>\nSubject: Итоги «Весна»'),
            message('private', 'b', '2028-01-01', 'From: Ann <anna@team.test>\nSubject: Итоги «Весна»'),
        ]
        with patch('mailsearch.retrieval.DenseIndex') as dense:
            dense.return_value.vectors = np.zeros((len(rows), 2), dtype=np.float32)
            dense.return_value.query.return_value = np.zeros(2, dtype=np.float32)
            cls.engine = SearchEngine(rows, Path('.'))

    def ids(self, query):
        return {did for did, _ in self.engine.search('a', query, date(2028, 3, 2))}

    def test_domain_address_and_name_match_header_only(self):
        for q in ('Покажи письма с адресов на @team.test', 'Find mail from anna@team.test', 'from:@team.test'):
            self.assertEqual(self.ids(q), {'one', 'two'})
        self.assertEqual(self.ids('от отправителя «Анна Ромашкина»'), {'one'})
        self.assertFalse(self.ids('from:missing@team.test'))

    def test_whole_title_and_multilingual_nested_quotes(self):
        self.assertEqual(self.ids('Найди письмо с темой «ИТОГИ «Весна»».'), {'one'})
        self.assertFalse(self.ids('with the exact subject "Итоги"'))
        self.assertEqual(self.ids('with the exact subject "Report "A" due today"'), {'two'})
        self.assertFalse(self.ids('with the exact subject "Отчёт"'))

    def test_combined_scope_and_residual_query(self):
        self.assertEqual(self.ids('from:anna@team.test 1 марта 2028 года'), {'one'})
        self.assertEqual(self.ids('from:anna@team.test subject:"Итоги «Весна»"'), {'one'})
        self.assertEqual(self.ids('from:anna@team.test "Презентация бюджета"'), {'two'})
        self.assertEqual(self.ids('«проекты от anna@team.test»'), {'other'})


if __name__ == '__main__':
    unittest.main()
