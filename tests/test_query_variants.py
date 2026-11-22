import unittest

from mailsearch.query_variants import expand_query


class QueryVariantTests(unittest.TestCase):
    def test_supported_variants_and_inflections(self):
        visible = {"общежитие", "spark", "договор"}
        self.assertEqual(expand_query("В общаге про СПАРК и дог-р", visible.__contains__),
                         "В общежитие про spark и договор")

    def test_original_form_and_unknown_target(self):
        visible = {"кафка", "kafka"}
        self.assertEqual(expand_query("Кафка, общага", visible.__contains__), "Кафка, общага")

    def test_protected_identifiers_and_quotes(self):
        query = '"спарк" «общага» me@спарк.ru https://example.test/общага SYN-спарк'
        self.assertEqual(expand_query(query, {"spark", "общежитие"}.__contains__), query)

    def test_mixed_spellings_share_lexical_evidence(self):
        from mailsearch.retrieval import _features, _words
        for query, spellings in [
            ("спарк", ("Spark", "Спарком")), ("общага", ("общежитие", "общаге")),
            ("дог-р", ("договор", "дог-р")),
        ]:
            for spelling in spellings:
                with self.subTest(query=query, spelling=spelling):
                    self.assertEqual(_features(_words(query), query=True), _features(_words(spelling)))

    def test_unlisted_names_in_both_directions_and_visible_candidate_collision(self):
        from mailsearch.query_variants import transliteration_index, VARIANTS
        vocabulary = {"gitlab", "grafana", "clickhouse", "redis", "falcon", "phalcon", "общежитие"}
        names = transliteration_index(vocabulary)
        for query, expected in [("гитлаб", "gitlab"), ("графана", "grafana"), ("кликхаус", "clickhouse"),
                                ("obshchezhitie", "общежитие"), ("фалкон", "фалкон")]:
            self.assertNotIn(query, VARIANTS)
            self.assertEqual(expand_query(query, vocabulary.__contains__, names), expected)
        self.assertEqual(expand_query("фалкон", {"phalcon"}.__contains__, names), "phalcon")
        self.assertEqual(expand_query("графана", set().__contains__, names), "графана")
        self.assertEqual(expand_query("редис", {"редис", "redis"}.__contains__, names), "редис")
        protected = 'SYN-гитлаб "гитлаб" https://test.invalid/гитлаб графана@test.example'
        self.assertEqual(expand_query(protected, vocabulary.__contains__, names), protected)

    def test_phrase_abbreviations_work_in_both_directions(self):
        from mailsearch.retrieval import _features, _words
        for short, full in [("ТЗ", "технического задания"), ("КП", "коммерческое предложение"),
                            ("Р/С", "расчетный счет"), ("ВКР", "выпускная квалификационная работа")]:
            self.assertEqual(_features(_words(short)), _features(_words(full), query=True))
        self.assertNotIn("abbr:техническое задание", _features(_words("техническое сложное задание")))
