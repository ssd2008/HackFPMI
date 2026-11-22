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
