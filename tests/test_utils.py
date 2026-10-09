
import unittest
from tests.support import OfflineTestCase
from agents import utils
class TestClaimExtraction(unittest.TestCase):
    def test_splits_a_multi_sentence_response_into_claims(self):
        claims = utils.split_into_claims(
            "Jupiter is the largest planet. It has exactly 100 confirmed moons."
        )
        self.assertEqual(len(claims), 2)
    def test_merges_a_stance_fragment_into_the_next_sentence(self):
        """'No, that's a myth.' states no checkable fact on its own."""
        claims = utils.split_into_claims(
            "No, that's a myth. Goldfish can remember things for several months."
        )
        self.assertEqual(len(claims), 1)
        self.assertIn("Goldfish", claims[0])
    def test_single_sentence_returns_one_claim(self):
        self.assertEqual(len(utils.split_into_claims("Paris is the capital of France.")), 1)
    def test_empty_text_returns_no_claims(self):
        self.assertEqual(utils.split_into_claims(""), [])
class TestContentCoverage(unittest.TestCase):
    def test_identical_text_is_fully_covered(self):
        text = "Paris is the capital of France."
        self.assertEqual(utils.content_coverage(text, text), 1.0)
    def test_unrelated_text_has_no_coverage(self):
        self.assertEqual(
            utils.content_coverage("Bananas are rich in potassium.", "Paris is the capital of France."),
            0.0,
        )
    def test_numbers_count_as_content(self):
        """The figure in a factual claim is content, not decoration."""
        self.assertIn("384400", utils.content_words("The Moon is 384,400 kilometers away."))
    def test_word_forms_match_after_stemming(self):
        self.assertGreater(
            utils.content_coverage("The carrot improves vision.", "Carrots improved vision."),
            0.5,
        )
class TestConflictDetection(unittest.TestCase):
    def test_detects_a_swapped_subject(self):
        self.assertTrue(
            utils.subject_conflict(
                "Berlin is the capital of France.",
                "Paris is the capital and most populous city of France.",
            )
        )
    def test_does_not_flag_a_conflict_when_the_subject_is_in_the_evidence(self):
        """Guards the false positive where an adverbial opening looked like a subject."""
        self.assertFalse(
            utils.subject_conflict(
                "The Grotto at Notre Dame is a Marian place of prayer and reflection.",
                "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.",
            )
        )
    def test_detects_a_different_number(self):
        self.assertTrue(
            utils.numbers_conflict(
                "The Moon is about 1,000,000 kilometers from Earth.",
                "The Moon is about 384,400 kilometers from Earth.",
            )
        )
    def test_no_numeric_conflict_when_only_one_side_has_numbers(self):
        self.assertFalse(utils.numbers_conflict("The Moon is far away.", "The Moon is 384,400 km away."))
    def test_detects_evidence_that_refutes_the_claim(self):
        self.assertTrue(
            utils.refutation_conflict(
                "Eating carrots dramatically improves your night vision.",
                "The popular claim that eating carrots gives you dramatically improved night vision is exaggerated.",
            )
        )
    def test_no_refutation_conflict_when_the_claim_agrees_with_the_evidence(self):
        self.assertFalse(
            utils.refutation_conflict(
                "Carrots do not dramatically improve night vision.",
                "The claim that carrots dramatically improve night vision is a myth.",
            )
        )
    def test_a_harmless_aside_is_not_treated_as_refutation(self):
        """'not all regions' must not make a correct claim look contradicted."""
        self.assertFalse(
            utils.refutation_conflict(
                "Humans use virtually all of their brain.",
                "Humans use virtually all of their brain, although not all regions are active at once.",
            )
        )
class TestQuestionShape(unittest.TestCase):
    def test_recognises_a_yes_no_question(self):
        self.assertTrue(utils.is_yes_no_question("Do goldfish have a 3 second memory?"))
        self.assertFalse(utils.is_yes_no_question("What is the capital of France?"))
    def test_recognises_a_direct_answer(self):
        self.assertTrue(utils.starts_with_direct_answer("No, that is a myth."))
        self.assertFalse(utils.starts_with_direct_answer("Paris is the capital."))
class TestEvidenceMatching(unittest.TestCase):
    def test_picks_the_best_matching_sentence_from_a_chunk(self):
        chunk = [
            "Atop the Main Building is a golden statue. "
            "Immediately behind the basilica is the Grotto, a Marian place of prayer."
        ]
        sentences = utils.evidence_sentences(chunk)
        self.assertEqual(len(sentences), 2)
        match = utils.match_claim_to_evidence("The Grotto is a Marian place of prayer.", sentences)
        self.assertIn("Grotto", match["evidence"])
    def test_returns_empty_match_when_there_is_no_evidence(self):
        match = utils.match_claim_to_evidence("Any claim.", [])
        self.assertIsNone(match["evidence"])
        self.assertEqual(match["coverage"], 0.0)
if __name__ == "__main__":
    unittest.main()
