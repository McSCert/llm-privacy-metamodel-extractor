import math
import unittest

from evaluation.evaluate_retrieval import (
    Chunk,
    evaluate_ranking,
    get_candidate_grade,
    is_in_principle,
    normalize_reference,
    parse_gold_references,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_parses_clause_grade_prefix_and_letter_item(self):
        warnings = []
        grades = parse_gold_references(
            "S1-4.1.3:2;S1-4.1.4:2(a);S14.1:1",
            "q-test",
            warnings,
        )
        self.assertEqual(
            grades,
            {"4.1.3": 2, "4.1.4(a)": 2, "4.1": 1},
        )
        self.assertEqual(warnings, [])

    def test_ungraded_reference_is_kept_with_warning(self):
        warnings = []
        grades = parse_gold_references("S1-4.5.3", "q-test", warnings)
        self.assertEqual(grades, {"4.5.3": 2})
        self.assertEqual(len(warnings), 1)

    def test_normalizes_chunk_reference_and_scopes_principle_boundary(self):
        self.assertEqual(
            normalize_reference("4.1 Principle 1 — Accountability"),
            "4.1",
        )
        self.assertEqual(normalize_reference("4.1.4(a)"), "4.1.4(a)")
        self.assertTrue(is_in_principle("4.1.4(a)", "4.1"))
        self.assertFalse(is_in_principle("4.10.1", "4.1"))

    def test_parent_chunk_matches_gold_letter_item(self):
        self.assertEqual(
            get_candidate_grade("4.1.4", {"4.1.4(a)": 2}),
            2,
        )

    def test_calculates_binary_retrieval_and_graded_ranking_metrics(self):
        def chunk(reference):
            return Chunk(
                chunk_id=reference,
                law="PIPEDA",
                article_ref=reference,
                reference=reference,
                text="",
                concept_tags=(),
                vector=[1.0],
            )

        ranked = [
            (chunk("4.1.1"), 0.9),
            (chunk("4.1.2"), 0.8),
            (chunk("4.1.9"), 0.1),
        ]
        metrics = evaluate_ranking(
            ranked,
            {"4.1.1": 2, "4.1.2": 1, "4.1.3": 2},
            k=3,
        )
        self.assertAlmostEqual(metrics["precision_at_k"], 2 / 3)
        self.assertAlmostEqual(metrics["recall_at_k"], 2 / 3)
        self.assertAlmostEqual(metrics["grade2_recall_at_k"], 1 / 2)
        self.assertEqual(metrics["mrr_at_k"], 1.0)
        expected_dcg = 3 + 1 / math.log2(3)
        expected_idcg = 3 + 3 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(metrics["ndcg_at_k"], expected_dcg / expected_idcg)

    def test_empty_gold_has_zero_metrics_without_division_error(self):
        chunk = Chunk("id", "PIPEDA", "4.1.1", "4.1.1", "", (), [1.0])
        metrics = evaluate_ranking([(chunk, 1.0)], {}, k=3)
        self.assertEqual(metrics["precision_at_k"], 0.0)
        self.assertEqual(metrics["recall_at_k"], 0.0)
        self.assertEqual(metrics["mrr_at_k"], 0.0)
        self.assertEqual(metrics["ndcg_at_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
