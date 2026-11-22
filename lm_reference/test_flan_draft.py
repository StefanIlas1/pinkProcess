"""The target model alone decides which proposed tokens reach the answer."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .flan_draft import TokenDraft


class DraftTests(unittest.TestCase):
    def test_context_eos_unknown_and_budget(self):
        draft = TokenDraft({(0, 3):4, (3, 4):5, (5,):1})
        self.assertEqual(draft.propose([3], 7), [4, 5])
        self.assertEqual(draft.propose([3], 1), [4])
        self.assertEqual(draft.propose([7], 7), [])
        self.assertEqual(draft.propose([3], 0), [])

    def test_history_table_requires_the_pinned_bytes(self):
        with TemporaryDirectory() as work:
            path = Path(work) / "draft.gz"
            path.write_bytes(b"wrong table")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                TokenDraft.load_history(path)

    @unittest.skipUnless(os.getenv("PINKPROCESS_FLAN_MODEL"), "A quantized FLAN checkpoint is required.")
    def test_rejected_proposals_calibration_limits_and_active_weight_edit(self):
        from .flan import Flan
        model = Flan(os.environ["PINKPROCESS_FLAN_MODEL"], None, reference=True)
        draft = TokenDraft.load()
        for prompt, limit, edit in (("What does the pool forecast mean?", 3, False),
                                    ("Translate to German: The stars are bright.", 32, False),
                                    ("Calculate 5 plus 8.", 32, True)):
            with self.subTest(prompt=prompt):
                if edit:
                    model.select_head_row(852, bytes(model.t["lm_head.weight"].shape[1]))
                model.token_draft = None
                expected = model.generate(prompt, max_tokens=limit)
                activation, score = model.first_output_activation, model.first_output_max
                model.token_draft = draft
                self.assertEqual(model.generate(prompt, max_tokens=limit), expected)
                self.assertEqual(model.first_output_activation, activation)
                self.assertEqual(model.first_output_max, score)
                phases = model.projection_profile()
                if edit:
                    self.assertLess(phases["accepted_proposals"], phases["proposed_tokens"])
                    model.restore_head()

    @unittest.skipUnless(os.getenv("PINKPROCESS_FLAN_MODEL"), "A quantized FLAN checkpoint is required.")
    def test_frozen_heldout_questions_match_scalar_greedy_decoding(self):
        from .flan import Flan
        model = Flan(os.environ["PINKPROCESS_FLAN_MODEL"], None, reference=True)
        draft = TokenDraft.load()
        evidence = Path(__file__).resolve().parents[1] / "kit/evidence/blind-diagnosis-replacement-w8a8.json"
        for case in json.loads(evidence.read_bytes())["cases"]:
            with self.subTest(question=case["prompt"]):
                model.token_draft = None
                expected = model.generate(case["prompt"])
                calibration = (model.first_output_activation, model.first_output_max)
                model.token_draft = draft
                self.assertEqual(model.generate(case["prompt"]), expected)
                self.assertEqual((model.first_output_activation, model.first_output_max), calibration)
