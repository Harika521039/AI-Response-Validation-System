"""
test_verdict_agent.py
----------------------
Tests for the M3.2 Verdict Agent: weight validation, normalization,
the weighted-score formula, the Pass/Needs Improvement/Fail thresholds,
and every critical rule.
"""
import unittest
from agents import verdict_agent
from agents.schemas import (
    AccuracyOutput,
    CompletenessOutput,
    FlaggedClaim,
    HallucinationOutput,
    RelevanceOutput,
    SchemaValidationError,
    VerdictOutput,
)
def _relevance(score=5):
    return RelevanceOutput(score=score, label="l", reasoning="r")
def _accuracy(score=5):
    return AccuracyOutput(score=score, label="l", reasoning="r")
def _completeness(score=5):
    return CompletenessOutput(score=score, label="l", reasoning="r")
def _hallucination(status="No hallucination", claims=None):
    return HallucinationOutput(hallucination_status=status, reasoning="r", flagged_claims=claims or [])
def _claim(status):
    return FlaggedClaim(claim="c", claim_status=status, reasoning="r")
class TestWeights(unittest.TestCase):
    def test_weights_total_exactly_one_hundred_percent(self):
        self.assertEqual(round(sum(verdict_agent.WEIGHTS.values()), 4), 1.0)
    def test_weights_are_not_all_equal(self):
        """The brief explicitly forbids the lazy 25%-each split."""
        self.assertNotEqual(len(set(verdict_agent.WEIGHTS.values())), 1)
    def test_accuracy_and_hallucination_are_weighted_highest(self):
        ordered = sorted(verdict_agent.WEIGHTS.items(), key=lambda kv: kv[1], reverse=True)
        top_two = {ordered[0][0], ordered[1][0]}
        self.assertEqual(top_two, {"accuracy", "hallucination"})
class TestNormalization(unittest.TestCase):
    def test_score_of_one_normalizes_to_zero(self):
        self.assertEqual(verdict_agent._normalize_1_to_5(1), 0.0)
    def test_score_of_five_normalizes_to_one_hundred(self):
        self.assertEqual(verdict_agent._normalize_1_to_5(5), 100.0)
    def test_score_of_three_normalizes_to_fifty(self):
        self.assertEqual(verdict_agent._normalize_1_to_5(3), 50.0)
    def test_hallucination_uses_claim_fraction_when_claims_exist(self):
        """Supported = full credit, Unsupported = half, Contradicted = none."""
        half_unsupported = _hallucination(
            "Partially hallucinated", [_claim("Supported"), _claim("Unsupported")]
        )
        self.assertEqual(verdict_agent._normalize_hallucination(half_unsupported), 75.0)
        half_contradicted = _hallucination(
            "Partially hallucinated", [_claim("Supported"), _claim("Contradicted")]
        )
        self.assertEqual(verdict_agent._normalize_hallucination(half_contradicted), 50.0)
        all_unsupported = _hallucination(
            "Partially hallucinated", [_claim("Unsupported"), _claim("Unsupported")]
        )
        self.assertEqual(verdict_agent._normalize_hallucination(all_unsupported), 50.0)
        all_contradicted = _hallucination("Hallucinated", [_claim("Contradicted")])
        self.assertEqual(verdict_agent._normalize_hallucination(all_contradicted), 0.0)
    def test_hallucination_falls_back_to_status_map_with_no_claims(self):
        self.assertEqual(
            verdict_agent._normalize_hallucination(_hallucination("No hallucination", [])), 100.0
        )
        self.assertEqual(
            verdict_agent._normalize_hallucination(_hallucination("Hallucinated", [])), 0.0
        )
class TestVerdictOutputSchema(unittest.TestCase):
    def test_weights_must_total_one_hundred_percent(self):
        with self.assertRaises(SchemaValidationError):
            VerdictOutput(
                relevance_score=5,
                accuracy_score=5,
                completeness_score=5,
                hallucination_status="No hallucination",
                normalized_scores={"accuracy": 100.0},
                weights={"accuracy": 0.5, "relevance": 0.2},
                weighted_overall_score=90.0,
                verdict="Pass",
                major_issues=[],
                reasoning="r",
            )
    def test_rejects_unknown_verdict_label(self):
        with self.assertRaises(SchemaValidationError):
            VerdictOutput(
                relevance_score=5,
                accuracy_score=5,
                completeness_score=5,
                hallucination_status="No hallucination",
                normalized_scores={"accuracy": 100.0},
                weights={"accuracy": 1.0},
                weighted_overall_score=90.0,
                verdict="Excellent",
                major_issues=[],
                reasoning="r",
            )
class TestComputeVerdict(unittest.TestCase):
    def test_all_perfect_scores_pass_with_no_issues(self):
        result = verdict_agent.compute_verdict(
            _relevance(5), _accuracy(5), _hallucination("No hallucination", [_claim("Supported")]), _completeness(5)
        )
        self.assertEqual(result.verdict, "Pass")
        self.assertEqual(result.weighted_overall_score, 100.0)
        self.assertEqual(result.major_issues, [])
    def test_preserves_each_judges_own_score(self):
        result = verdict_agent.compute_verdict(
            _relevance(4), _accuracy(3), _hallucination("Partially hallucinated", [_claim("Supported")]), _completeness(2)
        )
        self.assertEqual(result.relevance_score, 4)
        self.assertEqual(result.accuracy_score, 3)
        self.assertEqual(result.completeness_score, 2)
        self.assertEqual(result.hallucination_status, "Partially hallucinated")
    def test_mediocre_combination_lands_in_needs_improvement(self):
        result = verdict_agent.compute_verdict(
            _relevance(3),
            _accuracy(3),
            _hallucination("Partially hallucinated", [_claim("Supported"), _claim("Unsupported")]),
            _completeness(3),
        )
        self.assertEqual(result.verdict, "Needs Improvement")
    def test_low_combination_lands_in_fail(self):
        result = verdict_agent.compute_verdict(
            _relevance(1), _accuracy(1), _hallucination("Hallucinated", [_claim("Unsupported")]), _completeness(1)
        )
        self.assertEqual(result.verdict, "Fail")
    def test_critical_rule_hallucinated_status_forces_fail(self):
        """Perfect on every other dimension, but fully hallucinated -> Fail."""
        result = verdict_agent.compute_verdict(
            _relevance(5), _accuracy(5), _hallucination("Hallucinated", [_claim("Unsupported")]), _completeness(5)
        )
        self.assertEqual(result.verdict, "Fail")
        self.assertTrue(any("Hallucinated" in issue for issue in result.major_issues))
    def test_critical_rule_accuracy_one_forces_fail(self):
        result = verdict_agent.compute_verdict(
            _relevance(5), _accuracy(1), _hallucination("No hallucination", [_claim("Supported")]), _completeness(5)
        )
        self.assertEqual(result.verdict, "Fail")
    def test_critical_rule_severe_contradiction_forces_fail(self):
        claims = [_claim("Contradicted"), _claim("Contradicted"), _claim("Supported")]
        result = verdict_agent.compute_verdict(
            _relevance(5), _accuracy(3), _hallucination("Partially hallucinated", claims), _completeness(5)
        )
        self.assertEqual(result.verdict, "Fail")
        self.assertTrue(any("contradict" in issue.lower() for issue in result.major_issues))
    def test_critical_rule_low_accuracy_never_allows_a_pass(self):
        """
        Under the current weights a mostly-incorrect response can never earn
        a high enough weighted score to reach Pass on its own (accuracy=2
        contributes at most 8.75 of the 35 accuracy points), so the
        threshold lookup alone already keeps this out of Pass. The
        dedicated critical rule (CR4) is still exercised directly here as
        an explicit, weight-independent safety net: even if a hypothetical
        base verdict of "Pass" were reached with accuracy=2, the rule must
        still pull it down to "Needs Improvement".
        """
        result = verdict_agent.compute_verdict(
            _relevance(5), _accuracy(2), _hallucination("No hallucination", [_claim("Supported")]), _completeness(5)
        )
        self.assertNotEqual(result.verdict, "Pass")
        downgraded, issues = verdict_agent._apply_critical_rules(
            "Pass", _accuracy(2), _hallucination("No hallucination", [_claim("Supported")])
        )
        self.assertEqual(downgraded, "Needs Improvement")
        self.assertTrue(issues)
    def test_critical_rules_never_upgrade_a_verdict(self):
        """A rule can only make things worse, never rescue a low score."""
        result = verdict_agent.compute_verdict(
            _relevance(1), _accuracy(1), _hallucination("Hallucinated", [_claim("Unsupported")]), _completeness(1)
        )
        self.assertEqual(result.verdict, "Fail")
    def test_major_issues_mentions_missing_completeness_aspects(self):
        incomplete = CompletenessOutput(
            score=2, label="Mostly incomplete", reasoning="r", missing_aspects=["what currency is used"]
        )
        result = verdict_agent.compute_verdict(
            _relevance(5), _accuracy(5), _hallucination("No hallucination", [_claim("Supported")]), incomplete
        )
        self.assertTrue(any("what currency is used" in issue for issue in result.major_issues))
    def test_reasoning_mentions_the_weighted_score_and_verdict(self):
        result = verdict_agent.compute_verdict(
            _relevance(5), _accuracy(5), _hallucination("No hallucination", [_claim("Supported")]), _completeness(5)
        )
        self.assertIn(str(result.weighted_overall_score), result.reasoning)
        self.assertIn(result.verdict, result.reasoning)
if __name__ == "__main__":
    unittest.main()
