from contextlib import closing
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
import unittest

from lm_reference.weight_versions import ORIGINAL, WeightLayout, WeightVersionStore
from .board_language import BoardLanguage, GENERAL_MODEL, HISTORY_MODEL
from .kit_dataset import POOL_SAMPLE_CAPACITY, pool_sample_csv


QUESTION = "Has this connection pool recovered?"


class FakeRuntime:
    def __init__(self, block=False):
        self.block = block
        self.started = Event()
        self.release = Event()
        self.fabric_projections = 3
        self.first_output_activation = [1, -1, 0, 2]
        self.tokenizer = self
        self.installed = None
        self.install_attempts = []
        self.fail_installs = set()
        self.prompts = []
        self.generation_error = None
        self.fabric = None

    def decode(self, ids):
        return str(ids[0])

    def generate(self, prompt):
        self.prompts.append(prompt)
        self.started.set()
        if self.block and not self.release.wait(3):
            raise RuntimeError("Test generation timed out.")
        if self.generation_error is not None:
            raise self.generation_error
        return f"{self.installed.name}: {prompt}"

    def install_weight_version(self, version):
        self.installed = version
        self.install_attempts.append(version.name)
        if version.name in self.fail_installs:
            raise RuntimeError(f"Injected readback error for {version.name}")
        return version.classifier_sha256

    @property
    def t(self):
        return {"lm_head.weight": MockHead(self.installed.checkpoint[len(b"weights:"):])}


class MockHead:
    def __init__(self, matrix):
        self.matrix = matrix

    def __getitem__(self, row):
        return memoryview(self.matrix)[row * 4:(row + 1) * 4]


class MockFabric:
    def __init__(self):
        self.checked_rows = []
        self.rejected_row = None

    def verify_row_bytes(self, payload):
        self.checked_rows.append(payload)
        if payload == self.rejected_row:
            raise RuntimeError("Injected physical row mismatch")


class BoardLanguageTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.state_dir = Path(directory.name)
        self.base = b"weights:" + bytes(range(12))
        self.layout = WeightLayout(3, 4, len(b"weights:"), sha256(self.base).hexdigest(), "b" * 64)
        self.observation = {
            "fresh": True, "state": "ready", "origin": "live", "capacity": 10,
            "latest": {"time_ms": 6000, "busy": 5, "waiting": 0},
            "observations": {"peak_waiting": {"time_ms": 1000, "busy": 10, "waiting": 12}},
        }

    def store(self):
        return WeightVersionStore(self.state_dir / "versions.sqlite", self.base, layout=self.layout)

    def coordinator(self, runtime, snapshot=None):
        return BoardLanguage(runtime, self.store(), self.state_dir,
                             snapshot or (lambda: self.observation))

    def start(self, coordinator, **kwargs):
        return coordinator.handle({"op": "lm_start", "question": QUESTION, **kwargs})

    def finish(self, coordinator, job):
        deadline = monotonic() + 3
        while monotonic() < deadline:
            state = coordinator.handle({"op": "lm_result", "job_id": job["job_id"]})
            if state["state"] != "running":
                return state
            sleep(.005)
        self.fail("Language-model job did not finish.")

    def save(self, coordinator, name="Astra attempt"):
        return coordinator.handle({"op": "lm_save", "name": name, "target_row": 1,
                                   "response": {"row": [-127, 0, 127, 22]}})

    def test_full_checkpoint_switch_replay_and_dataset_isolation(self):
        from dataclasses import replace
        from types import SimpleNamespace

        self.observation["observations"]["first"] = {"time_ms": 0, "busy": 10, "waiting": 3}
        base, history = FakeRuntime(), FakeRuntime()
        alternate_bytes = b"weights:" + bytes(reversed(range(12)))
        layout = replace(self.layout, checkpoint_sha256=sha256(alternate_bytes).hexdigest())
        alternate = WeightVersionStore(self.state_dir / "alt.sqlite", alternate_bytes, layout=layout)
        history.installed = alternate.original
        bound = []
        base.query_projection = history.query_projection = SimpleNamespace(bind_model=bound.append)

        def open_service():
            versions = self.store()
            versions.register_checkpoint(HISTORY_MODEL, alternate)
            return BoardLanguage(base, versions, self.state_dir, lambda: self.observation,
                                 dataset_score=lambda *_: None, history_runtime=history)
        service = open_service()
        self.save(service)
        original_job = self.finish(service, self.start(service))
        original_prompt = base.prompts[-1]
        selected = service.handle(dict(op="lm_select", name=HISTORY_MODEL))
        self.assertEqual(selected["active"]["checkpoint_sha256"], layout.checkpoint_sha256)
        self.assertIs(service.runtime, history)
        service = open_service()
        self.assertIs(service.runtime, history)
        self.assertEqual(service.active.name, HISTORY_MODEL)
        old = self.finish(service, self.start(service, capture_id=original_job["capture_id"]))
        self.assertEqual(old["prompt_variant"], "original")
        self.assertEqual(history.prompts[-1], original_prompt)
        fresh = self.finish(service, self.start(service))
        self.assertEqual(fresh["prompt_variant"], "queue_history")
        self.assertNotEqual(history.prompts[-1], original_prompt)
        # The same row bytes define distinct classifiers under the two base models.
        context = service.handle(dict(op="lm_row", target_row=1))
        self.assertEqual(context["base_sha256"], layout.checkpoint_sha256)
        self.assertEqual(context["row"], list(alternate_bytes[-8:-4]))
        named = "History row test"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.save(service, named)
        service.handle(dict(op="lm_save", name=named, target_row=1, base_sha256=context["base_sha256"],
                            response={"row": [-127, 0, 127, 22]}))
        result = service.handle(dict(op="lm_select", name=named))
        self.assertEqual(result["active"]["base_name"], HISTORY_MODEL)
        self.assertNotEqual(result["active"]["classifier_sha256"], selected["active"]["classifier_sha256"])
        service = open_service()
        self.assertEqual(service.active.name, named)
        self.assertIs(service.runtime, history)
        self.assertEqual(self.finish(service, self.start(service))["prompt_variant"], "queue_history")
        with self.assertRaisesRegex(ValueError, "different base|another base"):
            service.handle(dict(op="lm_save", name="wrong source", target_row=1,
                                base_sha256=self.layout.checkpoint_sha256, response={"row": [1]*4}))
        service.handle(dict(op="lm_select", name="Astra attempt"))
        dataset_id = service.handle(dict(op="lm_dataset_upload", csv=pool_sample_csv(), capacity=POOL_SAMPLE_CAPACITY))["dataset_id"]
        service.handle(dict(op="lm_dataset_select", dataset_id=dataset_id, name=HISTORY_MODEL))
        result = self.finish(service, service.handle(dict(op="lm_dataset_ask", dataset_id=dataset_id, question=QUESTION)))
        self.assertEqual((result["active"]["checkpoint_sha256"], result["prompt_variant"]),
                         (layout.checkpoint_sha256, "queue_history"))
        self.assertIs(service.runtime, base)
        self.assertEqual(service.active.name, "Astra attempt")
        self.assertTrue(bound)
        context = service.handle(dict(op="lm_dataset_calibration", dataset_id=dataset_id, target_row=1))
        self.assertEqual((context["ready"], context["base_sha256"]), (True, layout.checkpoint_sha256))
        with self.assertRaisesRegex(ValueError, "another base"):
            service.handle(dict(op="lm_dataset_save", dataset_id=dataset_id, selected_model=HISTORY_MODEL,
                                name="wrong", target_row=1, response={"row": [1]*4},
                                base_sha256=self.layout.checkpoint_sha256))
        dataset_saved = service.handle(dict(op="lm_dataset_save", dataset_id=dataset_id, selected_model=HISTORY_MODEL,
                                            name="History dataset edit", target_row=1,
                                            response={"row": [6, 7, 8, 9]}, base_sha256=context["base_sha256"]))
        self.assertEqual(dataset_saved["selected_model"], "History dataset edit")
        rerun = self.finish(service, service.handle(dict(op="lm_dataset_ask", dataset_id=dataset_id, question=QUESTION)))
        self.assertEqual((rerun["active"]["name"], rerun["active"]["base_name"], rerun["prompt_variant"]),
                         ("History dataset edit", HISTORY_MODEL, "queue_history"))
        self.assertEqual(service.runtime.installed.name, "Astra attempt")
        service.jobs.pop(service.activation_job_id)
        with self.assertRaisesRegex(ValueError, "selected base checkpoint"):
            service.handle(dict(op="lm_row", target_row=1))
        service.handle(dict(op="lm_restore"))
        replay = self.finish(service, self.start(service, capture_id=fresh["capture_id"]))
        self.assertEqual(replay["prompt_variant"], "original")
        self.assertEqual(base.prompts[-1], original_prompt)
        with self.assertRaisesRegex(ValueError, "full checkpoint"):
            self.save(service, HISTORY_MODEL)
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            self.store().register_checkpoint(HISTORY_MODEL, self.store())

    def test_general_csv_answer_uses_distinct_checkpoint_and_restores_live_selection(self):
        base, general = FakeRuntime(), FakeRuntime()
        store = self.store()
        store.register_checkpoint(GENERAL_MODEL, store)
        service = BoardLanguage(base, store, self.state_dir, lambda: self.observation,
                                general_runtime=general)
        self.assertEqual(service.status()["data_model"], GENERAL_MODEL)
        result = self.finish(service, service.handle(dict(op="lm_data_start", question="Which endpoint slowed?",
                                                           findings="Latency rose on /checkout.")))
        self.assertEqual(result["active"]["name"], GENERAL_MODEL)
        self.assertIn("Latency rose on /checkout.", result["answer"])
        self.assertIn("Question: Which endpoint slowed?", result["answer"])
        self.assertEqual(service.status()["active"]["name"], ORIGINAL)
        self.assertIs(service.runtime, base)

    def test_dataset_selection_is_scoped_and_followup_uses_prior_answer(self):
        runtime = FakeRuntime()
        score_calls = []

        def score(window, capacity):
            score_calls.append((len(window), capacity))
            return {"model": "pool_wait_logistic", "alert": True, "version": "test",
                    "time_ms": window[-1].time_ms, "horizon_ms": 10000}

        service = BoardLanguage(runtime, self.store(), self.state_dir,
                                lambda: self.fail("Dataset inference touched the live stream."), dataset_score=score)
        dataset = service.handle(dict(op="lm_dataset_upload", csv=pool_sample_csv(), capacity=POOL_SAMPLE_CAPACITY, name="Raw CSV"))
        dataset_id = dataset["dataset_id"]
        self.assertEqual(score_calls, [(50, POOL_SAMPLE_CAPACITY)])
        self.assertEqual(dataset["snapshot"]["origin"], "dataset")
        self.assertEqual(dataset["summary"]["sample_count"], dataset["snapshot"]["sample_count"])
        self.assertNotIn("csv", dataset)
        self.save(service)
        service.handle(dict(op="lm_dataset_select", dataset_id=dataset_id, name="Astra attempt"))
        question = "What is the latest forecast?"
        first = self.finish(service, service.handle(dict(op="lm_dataset_ask", dataset_id=dataset_id,
                                                         question=question, stage="final")))
        self.assertEqual(first["active"]["name"], "Astra attempt")
        self.assertIn("CSV measurements", first["answer"])
        self.assertIn("The FPGA warns", first["answer"])
        self.assertEqual(runtime.installed.name, ORIGINAL)
        next_job = service.handle(dict(op="lm_dataset_ask", dataset_id=dataset_id, question=QUESTION,
                                       stage="followup"))
        self.assertIn("Previous question: " + question, self.finish(service, next_job)["answer"])
        reopened = BoardLanguage(FakeRuntime(), self.store(), self.state_dir, lambda: self.observation,
                                 dataset_score=score)
        reply = reopened.handle(dict(op="lm_dataset_models", dataset_id=dataset_id))
        self.assertEqual(reply["selected_model"], "Astra attempt")
        self.assertEqual(reopened.status()["active"]["name"], ORIGINAL)

    def test_dataset_calibration_saves_a_real_scoped_weight_version(self):
        service = BoardLanguage(FakeRuntime(), self.store(), self.state_dir, lambda: self.observation,
                                dataset_score=lambda window, capacity: None)
        dataset_id = service.handle(dict(op="lm_dataset_upload", csv=pool_sample_csv(),
                                         capacity=POOL_SAMPLE_CAPACITY))["dataset_id"]
        action = dict(op="lm_dataset_calibration", dataset_id=dataset_id, target_row=1)
        self.assertFalse(service.handle(action)["ready"])
        calibration = service.handle(dict(op="lm_dataset_calibrate", dataset_id=dataset_id))
        self.assertEqual(self.finish(service, calibration)["state"], "complete")
        context = service.handle(action)
        self.assertTrue(context["ready"])
        response = service.handle(dict(op="lm_dataset_save", dataset_id=dataset_id, name="CSV synthesis",
                                       target_row=1, response={"row": [127, 0, -127, 2]},
                                       base_sha256=context["base_sha256"], selected_model=context["selected_model"]))
        self.assertEqual(response["selected_model"], "CSV synthesis")
        self.assertEqual(self.store().get("CSV synthesis").patch.payload, bytes([127, 0, 129, 2]))
        self.assertEqual(self.store().active().name, ORIGINAL)

    def test_capture_freezes_evidence_while_inference_and_telemetry_continue(self):
        runtime = FakeRuntime(block=True)
        self.addCleanup(runtime.release.set)
        coordinator = self.coordinator(runtime)
        recording = self.start(coordinator)
        self.assertTrue(runtime.started.wait(1))
        self.observation["latest"]["waiting"] = 9
        self.assertEqual(recording["snapshot"]["latest"]["waiting"], 0)
        self.assertEqual(self.save(coordinator)["active"]["name"], ORIGINAL)
        with self.assertRaisesRegex(ValueError, "in progress"):
            coordinator.handle({"op": "lm_select", "name": "Astra attempt"})
        with self.assertRaisesRegex(ValueError, "in progress"):
            self.start(coordinator)
        runtime.release.set()
        first = self.finish(coordinator, recording)
        self.assertIn("Current queue: 0 waiting requests.", first["answer"])
        self.assertEqual(first["active"]["name"], ORIGINAL)
        self.assertEqual(first["fabric_projections"], 3)
        second = self.finish(coordinator, self.start(coordinator))
        self.assertNotEqual(second["capture_id"], first["capture_id"])
        self.assertIn("Current queue: 9 waiting requests.", second["answer"])

    def test_verified_selection_reloads_before_offline_capture_replay(self):
        runtime = FakeRuntime()
        coordinator = self.coordinator(runtime)
        first = self.finish(coordinator, self.start(coordinator))
        row = coordinator.handle({"op": "lm_row", "target_row": 1})
        self.assertEqual(row["row"], [4, 5, 6, 7])
        self.assertEqual(row["activation"], runtime.first_output_activation)
        self.save(coordinator)
        chosen = coordinator.handle({"op": "lm_select", "name": "Astra attempt"})["active"]
        self.assertEqual(chosen, runtime.installed.metadata())
        self.assertEqual(chosen, self.store().active().metadata())
        self.assertEqual(runtime.installed.patch.payload, bytes([129, 0, 127, 22]))

        def offline():
            self.fail("Replay read live telemetry.")
        rebooted_runtime = FakeRuntime()
        rebooted = self.coordinator(rebooted_runtime, snapshot=offline)
        self.assertEqual(rebooted_runtime.installed.metadata(), chosen)
        listed = rebooted.handle({"op": "lm_captures"})
        expected_capture = dict(id=first["capture_id"], question=QUESTION, captured_at_ms=first["captured_at_ms"],
                                reading_time_ms=6000, baseline={"answer": first["answer"], "version": first["active"]})
        self.assertGreater(first["captured_at_ms"], 0)
        self.assertEqual(listed, {"ok": True, "captures": [expected_capture], "before": None})
        replay_job = self.start(rebooted, capture_id=first["capture_id"])
        self.assertEqual(replay_job["baseline"], expected_capture["baseline"])
        replay = self.finish(rebooted, replay_job)
        self.assertEqual(replay["active"], chosen)
        self.assertEqual(replay["baseline"], expected_capture["baseline"])
        self.assertEqual(rebooted_runtime.prompts, runtime.prompts)
        self.assertNotEqual(replay["answer"], first["answer"])
        self.assertEqual(rebooted.handle({"op": "lm_captures"}), listed)
        restored = rebooted.handle({"op": "lm_restore"})
        self.assertEqual(restored["active"]["name"], ORIGINAL)
        self.assertEqual(rebooted_runtime.installed.checkpoint, self.base)
        self.assertEqual([version["name"] for version in restored["versions"]], [ORIGINAL, chosen["name"]])
        self.assertEqual(self.store().active().checkpoint, self.base)
        self.assertEqual(self.finish(rebooted, self.start(rebooted, capture_id=first["capture_id"]))["answer"],
                         first["answer"])

    def test_replay_rejects_changed_question_and_corrupted_capture(self):
        coordinator = self.coordinator(FakeRuntime())
        captured = self.finish(coordinator, self.start(coordinator))["capture_id"]
        with self.assertRaisesRegex(ValueError, "original question"):
            coordinator.handle({"op": "lm_start", "question": "What is the latency?", "capture_id": captured})
        with closing(sqlite3.connect(coordinator.capture_db)) as connection, connection:
            connection.execute("UPDATE captures SET body = ? WHERE id = ?", (b"{}", captured))
        with self.assertRaisesRegex(ValueError, "unavailable or damaged"):
            self.start(coordinator, capture_id=captured)
        self.assertEqual(coordinator.handle({"op": "lm_captures"})["captures"], [])

    def test_capture_listing_paginates_legacy_records_without_inventing_baselines(self):
        import json

        coordinator = self.coordinator(FakeRuntime())
        records = []
        for index in range(22):
            body = json.dumps(dict(question=QUESTION, prompt=f"Legacy prompt {index}",
                                   snapshot=self.observation)).encode()
            records.append((sha256(body).hexdigest(), body))
        with coordinator.open_captures() as db:
            db.executemany('INSERT INTO captures VALUES (?,?)', records)
        recent = coordinator.handle({"op": "lm_captures"})
        older = coordinator.handle({"op": "lm_captures", "before": recent["before"]})
        self.assertEqual(len(recent["captures"]), 20)
        self.assertEqual([item["id"] for item in older["captures"]], [records[1][0], records[0][0]])
        self.assertIsNone(older["before"])
        self.assertIsNone(older["captures"][0]["captured_at_ms"])
        legacy = self.finish(coordinator, self.start(coordinator, capture_id=records[0][0]))
        self.assertIn("Legacy prompt 0", legacy["answer"])
        self.assertIsNone(legacy["baseline"])
        with self.assertRaisesRegex(ValueError, "Invalid saved-capture page"):
            coordinator.handle({"op": "lm_captures", "before": True})

    def test_failed_version_selection_keeps_runtime_and_store_in_sync(self):
        runtime = FakeRuntime()
        coordinator = self.coordinator(runtime)
        name = "Astra attempt"
        self.save(coordinator, name)
        runtime.fail_installs.add(name)
        with self.assertRaisesRegex(RuntimeError, "Injected readback error"):
            coordinator.handle({"op": "lm_select", "name": name})
        self.assertEqual(runtime.install_attempts[-2:], [name, ORIGINAL])
        self.assertEqual(runtime.installed.name, ORIGINAL)
        self.assertEqual(self.store().active().name, ORIGINAL)
        self.assertEqual(coordinator.handle({"op": "lm_versions"})["active"]["name"], ORIGINAL)

        runtime.fail_installs.clear()
        with closing(sqlite3.connect(coordinator.versions.database)) as connection, connection:
            connection.execute("""CREATE TRIGGER interrupt_selection BEFORE UPDATE ON model_state
                                  BEGIN SELECT RAISE(ABORT, 'interrupted'); END""")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            coordinator.handle({"op": "lm_select", "name": name})
        self.assertEqual(runtime.installed.name, ORIGINAL)
        self.assertEqual(self.store().active().name, ORIGINAL)

    def test_select_updates_persistence_only_after_fabric_readback(self):
        runtime = FakeRuntime()
        runtime.fabric = MockFabric()
        coordinator = self.coordinator(runtime)
        self.save(coordinator)
        edit = bytes([129, 0, 127, 22])
        runtime.fabric.rejected_row = edit
        with self.assertRaisesRegex(RuntimeError, "physical row mismatch"):
            coordinator.handle({"op": "lm_select", "name": "Astra attempt"})
        self.assertIn(edit, runtime.fabric.checked_rows)
        self.assertEqual(runtime.installed.name, ORIGINAL)
        self.assertEqual(self.store().active().name, ORIGINAL)

        runtime.fabric.rejected_row = None
        coordinator.handle({"op": "lm_select", "name": "Astra attempt"})
        self.assertEqual(runtime.fabric.checked_rows[-1], edit)
        self.assertEqual(self.store().active().name, "Astra attempt")
        coordinator.handle({"op": "lm_restore"})
        self.assertEqual(runtime.fabric.checked_rows[-1], bytes([4, 5, 6, 7]))
        self.assertEqual(self.store().active().name, ORIGINAL)

    def test_failed_restoration_fences_inference_until_service_restart(self):
        runtime = FakeRuntime()
        coordinator = self.coordinator(runtime)
        self.save(coordinator)
        runtime.fail_installs.update(["Astra attempt", ORIGINAL])
        with self.assertRaisesRegex(RuntimeError, "Could not restore"):
            coordinator.handle({"op": "lm_select", "name": "Astra attempt"})
        with self.assertRaisesRegex(RuntimeError, "require recovery"):
            self.start(coordinator)
        with self.assertRaisesRegex(RuntimeError, "require recovery"):
            coordinator.handle({"op": "lm_versions"})
        rebooted = self.coordinator(FakeRuntime())
        self.assertEqual(rebooted.handle({"op": "lm_versions"})["active"]["name"], ORIGINAL)

    def test_generation_failure_is_reported_and_releases_job_slot(self):
        runtime = FakeRuntime()
        coordinator = self.coordinator(runtime)
        runtime.generation_error = OSError("Injected generation failure")
        result = self.finish(coordinator, self.start(coordinator))
        self.assertEqual((result["state"], result["error"]), ("failed", "Injected generation failure"))
        runtime.generation_error = None
        self.assertEqual(self.finish(coordinator, self.start(coordinator))["state"], "complete")


if __name__ == "__main__":
    unittest.main()
