"""Real-weight replay for optional full-checkpoint and row version switching."""
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest


@unittest.skipUnless(os.getenv("PINKPROCESS_FLAN_MODEL") and os.getenv("PINKPROCESS_HISTORY_MODEL"),
                     "Set both diagnostic checkpoint paths for the full replay.")
class CheckpointRoutingTest(unittest.TestCase):
    def test_named_history_row_restarts_and_restores_exact_inference(self):
        from kit.test_benchmark_projection_overlay import FakeOverlay
        from lm_reference.flan import Flan
        from lm_reference.flan_draft import TokenDraft
        from lm_reference.flan_prompt import telemetry_prompt
        from lm_reference.projection_overlay import OverlayProjection
        from lm_reference.weight_versions import WeightVersionStore
        from pinkprocess.astra_edit import materialize
        from pinkprocess.board_language import BoardLanguage, HISTORY_MODEL
        from pinkprocess.kit_evidence import analyst_evidence

        paths = [Path(os.environ["PINKPROCESS_FLAN_MODEL"]), Path(os.environ["PINKPROCESS_HISTORY_MODEL"])]
        question = "Has this system recovered, or was it quiet throughout?"
        reading = dict(time_ms=5900, busy=2, waiting=0)
        snapshot = dict(state="ready", fresh=True, capacity=4, latest=reading,
                        observations=dict(first={**reading, "time_ms": 0}, peak_waiting=reading))
        heads = []
        class RowProbe:
            def verify_row_bytes(self, row):
                heads.append(row)

        with TemporaryDirectory() as temp:
            def open_service():
                stores = [WeightVersionStore.from_flan(
                    Path(temp) / f"{index}.sqlite", path,
                    manifest_sha256=sha256((path / "model.json").read_bytes()).hexdigest())
                          for index, path in enumerate(paths)]
                stores[0].register_checkpoint(HISTORY_MODEL, stores[1])
                fake = FakeOverlay(queue=True)
                runner = OverlayProjection(fake, fake.allocate, scope="all", queue=True, resident_head_mib=16)
                self.addCleanup(runner.close)
                base = Flan(paths[0], RowProbe(), query_projection=runner)
                alternate = Flan(paths[1], base.fabric)
                alternate.query_projection = runner
                base.token_draft = alternate.token_draft = TokenDraft.load()
                return BoardLanguage(base, stores[0], temp, lambda: snapshot, history_runtime=alternate)

            def ask(service, *, capture=None):
                request = dict(op="lm_start", question=question, capture_id=capture)
                job = service.handle(request)
                deadline = monotonic() + 120
                while job["state"] == "running" and monotonic() < deadline:
                    sleep(.01)
                    job = service.handle(dict(op="lm_result", job_id=job["job_id"]))
                self.assertEqual(job["state"], "complete", job.get("error"))
                self.assertEqual(job["fabric_projections"], job["accelerated_projections"])
                self.assertIs(service.runtime.query_projection.model_weights, service.runtime.t)
                return job

            service = open_service()
            original = ask(service)
            service.handle(dict(op="lm_select", name=HISTORY_MODEL))
            baseline = ask(service)
            ctx = service.handle(dict(op="lm_row", target_row=852))
            row = materialize(dict(direction="increase", strength=110), ctx["row"], ctx["activation"])
            name = "Test history token"
            service.handle(dict(op="lm_save", name=name, target_row=852, base_sha256=ctx["base_sha256"],
                                response=dict(row=row, target_row=852, base_sha256=ctx["base_sha256"])))
            service.handle(dict(op="lm_select", name=name))
            self.assertEqual(heads[-1], bytes(x % 256 for x in row))
            edited = ask(service, capture=baseline["capture_id"])
            version = service.versions.get(name)
            self.assertEqual((edited["active"]["base_name"], edited["prompt_variant"]),
                             (HISTORY_MODEL, "queue_history"))
            self.assertNotEqual(version.checkpoint_sha256, ctx["base_sha256"])
            self.assertNotEqual(baseline["answer"], edited["answer"])
            ref = Flan(paths[1], None, reference=True)
            ref.install_weight_version(version)
            self.assertEqual(edited["answer"], ref.generate(
                telemetry_prompt(question, analyst_evidence(snapshot), history=True)))
            # Reopen the persistent selection with fresh model mappings and DMA caches.
            service.runtime.query_projection.close()
            service = open_service()
            self.assertEqual(service.active.metadata(), edited["active"])
            self.assertEqual(ask(service, capture=baseline["capture_id"])["answer"], edited["answer"])
            service.handle(dict(op="lm_select", name=HISTORY_MODEL))
            self.assertEqual(ask(service, capture=baseline["capture_id"])["answer"], baseline["answer"])
            service.handle(dict(op="lm_restore"))
            self.assertEqual(ask(service, capture=original["capture_id"])["answer"], original["answer"])
            self.assertIn(name, [v["name"] for v in service.status()["versions"]])
