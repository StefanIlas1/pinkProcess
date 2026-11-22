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

    def test_prompt_copy_is_local_limited_and_ends_at_sentence_boundaries(self):
        class Tokenizer:
            def encode(self, sentence):
                return [int(token) for token in sentence.rstrip(".").split()] + [9, 1]

        base = TokenDraft({(4, 5): 8, (9,): 7})
        first = base.for_prompt(Tokenizer(), "3 4 5 6. 2 3 8")
        second = base.for_prompt(Tokenizer(), "3 4 5 7.")
        self.assertEqual(first.propose([4, 5], 3), [6, 9, 7])
        self.assertEqual(second.propose([4, 5], 1), [7])
        self.assertEqual(base.propose([4, 5], 1), [8])

    @unittest.skipUnless(os.getenv("PINKPROCESS_HISTORY_MODEL"), "A quantized history checkpoint is required.")
    def test_wide_prompt_copy_matches_token_ids_and_row_calibration(self):
        import gzip
        from unittest.mock import patch
        from .flan import Flan
        from .flan_prompt import telemetry_prompt

        model = Flan(os.environ["PINKPROCESS_HISTORY_MODEL"], None, reference=True)
        evidence = Path(__file__).resolve().parents[1] / "kit/evidence/frozen-queue-history-18.json.gz"
        cases = json.loads(gzip.decompress(evidence.read_bytes()))["cases"]
        draft = TokenDraft.load()
        for i in (0, 3, 6, 12, 15):
            case = cases[i]
            prompt = telemetry_prompt(case["question"], case["evidence"], history=True)
            for edited in (False, True):
                if edited:
                    model.select_head_row(852, bytes([13]) * 512)
                with self.subTest(i=i, edited=edited):
                    limit = 46 if not edited else 23
                    model.token_draft = None
                    with patch.object(model.tokenizer, "decode", wraps=model.tokenizer.decode) as decode:
                        expected = model.generate(prompt, limit)
                        tokens = tuple(decode.call_args.args[0])
                    calibration = (model.first_output_activation, model.first_output_max)
                    model.token_draft = draft
                    model.prompt_draft_positions = 16
                    with patch.object(model.tokenizer, "decode", wraps=model.tokenizer.decode) as decode:
                        self.assertEqual(model.generate(prompt, limit), expected)
                        self.assertEqual(tuple(decode.call_args.args[0]), tokens)
                    self.assertEqual((model.first_output_activation, model.first_output_max), calibration)
                if edited:
                    model.restore_head()

    @unittest.skipUnless(os.getenv("PINKPROCESS_HISTORY_MODEL") and os.getenv("PINKPROCESS_HISTORY_TOKEN_DRAFT"),
                         "The released history checkpoint and proposal table are required.")
    def test_frozen_prompt_copy_counts_and_calibration(self):
        import gzip
        from unittest.mock import patch
        from .flan import Flan
        from .flan_prompt import telemetry_prompt

        evidence = Path(__file__).resolve().parents[1] / "kit/evidence"
        study = json.loads((evidence / "history-prompt-copy-parity.json").read_bytes())
        cases = json.loads(gzip.decompress((evidence / study["inputs"]).read_bytes()))["cases"]
        model = Flan(os.environ["PINKPROCESS_HISTORY_MODEL"], None, reference=True)
        model.token_draft = TokenDraft.load_history(os.environ["PINKPROCESS_HISTORY_TOKEN_DRAFT"])
        self.assertEqual(len(cases), len(study["cases"]))
        for case, observed in zip(cases, study["cases"]):
            with self.subTest(case=case["case_id"]):
                self.assertEqual(case["case_id"], observed["id"])
                prompt = telemetry_prompt(case["question"], case["evidence"], history=True)
                outputs = []
                for width, key in ((None, "existing_width_8"), (16, "prompt_copy_width_16")):
                    model.prompt_draft_positions = width
                    with patch.object(model.tokenizer, "decode", wraps=model.tokenizer.decode) as decode:
                        answer = model.generate(prompt)
                        tokens = tuple(decode.call_args.args[0])
                    outputs.append((answer, tokens, tuple(model.first_output_activation), model.first_output_max))
                    self.assertEqual(model.generation_timing["verification_blocks"], observed[key])
                self.assertEqual(*outputs)
