import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import sleep
from unittest.mock import patch
from playwright.sync_api import sync_playwright, expect
from lm_reference.weight_versions import WeightLayout, WeightVersionStore
from pinkprocess.board_language import BoardLanguage
from pinkprocess.kit_app import Kit, KitHandler, KitServer
from pinkprocess.test_board_language import FakeRuntime

QUESTION = "What is the measured queue doing?"
FOLLOWUP = "What occurred before the last sample?"
ANSWER = "Three requests are waiting. The recorded queue persisted through the final sample. Check the producer."
CHAT = "The uploaded CSV shows the same queue count over the last ten readings."
CALIBRATION = "Calibration on the uploaded queue."
CSV = ("time_ms,busy,waiting\n" + "".join(f"{i*100},4,3\n" for i in range(60))).encode()


class ProjectionRuntime(FakeRuntime):
    def generate(self, prompt):
        self.prompts.append((self.installed.name, prompt))
        self.partial_answer = "Readings are being reviewed."
        sleep(.12)
        if "Summarize the observed connection pool readings." in prompt:
            return CALIBRATION
        if FOLLOWUP in prompt:
            assert "Previous question: " + QUESTION in prompt
            assert "Previous answer: " + ANSWER in prompt
            return CHAT
        assert QUESTION in prompt
        return ANSWER


class BoardTransport:
    def __init__(self, board):
        self.board = board
        self.calls = []

    def discover(self):
        return [{"host":"test-board.invalid", "name":"Isolated Board"}]

    def connect(self, host):
        assert host == "test-board.invalid", host
        return self.request("status")

    def close(self):
        pass

    def request(self, action, **params):
        self.calls.append((action, params))
        if action == "status":
            return {"ok":True,"backend":"PYNQ-Z2","version":"original","versions":["original"],
                    "lm_supported":True,"stream_supported":False, "datasets_supported":True}
        try:
            return self.board.handle({"op":action, **params})
        except (ValueError, RuntimeError) as err:
            return {"ok":False, "error":str(err)}


with TemporaryDirectory(prefix="pinkprocess-dataset-ci-") as tmp, sync_playwright() as playwright:
    tmp = Path(tmp)
    base = b"weights:" + bytes(range(12))
    layout = WeightLayout(3, 4, len(b"weights:"), hashlib.sha256(base).hexdigest(), "b"*64)
    weights = WeightVersionStore(tmp/"versions.sqlite", base, layout=layout)
    runtime = ProjectionRuntime()
    counts = []

    def score(window, capacity):
        counts.append((len(window), capacity))
        return {"model":"pool_wait_logistic", "alert":True, "version":"qa",
                "time_ms":window[-1].time_ms, "horizon_ms":10000}

    def no_live_stream():
        raise AssertionError("Uploaded dataset requested a live telemetry snapshot")

    board = BoardLanguage(runtime, weights, tmp, no_live_stream, dataset_score=score)
    transport = BoardTransport(board)
    cli = tmp/"codex"
    cli.write_text("""#!/usr/bin/env python3
import json, pathlib, sys
if (pathlib.Path(__file__).parent / "fail").exists():
    sys.exit(7)
args = sys.argv
assert args[args.index("--model") + 1] == "gpt-6.2-astra"
assert args[args.index("--sandbox") + 1] == "read-only"
assert '"target_row": 1' in sys.stdin.read()
pathlib.Path(args[args.index("-o") + 1]).write_text(json.dumps({
    "direction":"increase","strength":5,"reason":"isolated deterministic CLI result"}))
""")
    cli.chmod(0o755)
    with patch.dict(os.environ, {"PATH":str(tmp) + os.pathsep + os.environ["PATH"]}), KitServer(("127.0.0.1",0),KitHandler) as server:
        server.kit = Kit(transport)
        host = f"127.0.0.1:{server.server_port}"
        server.allowed_hosts = {host}
        worker = Thread(target=server.serve_forever,daemon=True)
        worker.start()
        chrome = playwright.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",headless=True)
        try:
            page = chrome.new_page(viewport={"width":1240,"height":840})
            errors = []
            page.on("pageerror",lambda exc:errors.append(str(exc)))
            page.goto("http://" + host)
            page.locator("#host").fill("test-board.invalid")
            page.locator("#connect").click()
            expect(page.locator("#connection")).to_contain_text("PYNQ-Z2")
            page.locator("#workflow-capacity").fill("12")
            upload = page.locator("#workflow-upload")
            expect(upload).to_be_enabled()
            upload.set_input_files({"name":"bad.csv","mimeType":"text/csv","buffer":b"wrong,fields\n1,2\n"})
            expect(page.locator("#workflow-status")).to_have_attribute("data-error", "true")
            assert not runtime.prompts
            expect(upload).to_be_enabled()
            upload.set_input_files({"name":"board-dataset.csv","mimeType":"text/csv","buffer":CSV})
            expect(page.locator("#workflow-filename")).to_contain_text("60 readings")
            expect(page.locator("#workflow-rows tr")).to_have_count(30)
            expect(page.locator("#workflow-model")).to_have_value("original")
            assert counts == [(50,12)], counts
            page.locator("#workflow-create").click()
            page.locator("#workflow-name").fill("Scoped QA")
            page.locator("#workflow-target").fill("1")
            page.locator("#workflow-instruction").fill("Increase the target token modestly.")
            page.locator("#workflow-synthesize").click()
            expect(page.locator("#workflow-model")).to_have_value("Scoped QA", timeout=8000)
            assert weights.get("Scoped QA").patch.payload == bytes([9,0,6,12])
            assert list(n for n,_ in runtime.prompts) == ["original"]
            assert weights.active().name == "original"
            stages = [
                ("transcript", QUESTION), ("generalization","60 readings"), ("short_answer","Three requests are waiting."),
                ("summary","Three requests are waiting.\nCheck the producer."), ("final",ANSWER)]
            page.locator("#workflow-question").fill(QUESTION)
            page.locator("#workflow-ask").click()
            page.locator("#tab-table").click()
            # Tab choice persists while asynchronous FPGA answer resolves.
            expect(page.locator("#workflow-ask")).to_be_enabled(timeout=8000)
            expect(page.locator("#tab-table")).to_have_attribute("aria-selected","true")
            assert len(runtime.prompts) == 2, runtime.prompts
            assert runtime.prompts[1][0] == "Scoped QA"
            for name,text in stages:
                page.locator("#tab-"+name).click()
                output = page.locator("#panel-"+name+" .workflow-output")
                if name in ("transcript","short_answer","summary","final"):
                    expect(output).to_have_text(text)
                else:
                    expect(output).to_contain_text(text, ignore_case=True)
                print(f"Panel {name}: {output.inner_text()}")
            page.locator("#tab-chat").click()
            expect(page.locator("#workflow-chat-log")).to_contain_text(ANSWER)
            page.locator("#workflow-followup").fill(FOLLOWUP)
            page.locator("#workflow-send").click()
            expect(page.locator("#workflow-chat-log")).to_contain_text(CHAT, timeout=8000)
            assert len(runtime.prompts) == 3, runtime.prompts
            assert [payload.get("stage") for action,payload in transport.calls if action == "lm_dataset_ask"] == ["final","followup"]
            assert weights.active().name == "original"
            page.locator("#workflow-create").click()
            page.locator("#workflow-name").fill("Must not save")
            page.locator("#workflow-target").fill("1")
            page.locator("#workflow-instruction").fill("Trigger a local command failure.")
            (tmp/"fail").touch()
            page.locator("#workflow-synthesize").click()
            expect(page.locator("#workflow-status")).to_have_attribute("data-error", "true", timeout=8000)
            expect(page.locator("#workflow-status")).to_contain_text("Astra did not return an edit")
            expect(page.locator("#workflow-model")).to_have_value("Scoped QA")
            assert all(version["name"] != "Must not save" for version in weights.list())
            assert not errors, errors
            page.screenshot(path="/tmp/pinkprocess-real-kit-integration.png")
            print("PASS: invalid/valid upload, calibrated creation before Ask, four attributed views, one Ask/one follow-up generation, CLI failure recovery")
        finally:
            chrome.close()
            server.shutdown()
            worker.join()
