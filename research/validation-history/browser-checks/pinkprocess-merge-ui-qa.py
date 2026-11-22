from pathlib import Path
from threading import Thread
from playwright.sync_api import sync_playwright, expect
from pinkprocess.kit_app import Kit, KitServer, KitHandler
from pinkprocess.test_kit_app import MockBoard

BASE = {"name":"original", "checkpoint_sha256":"a"*64, "classifier_sha256":"b"*64}
EDIT = {"name":"Saved edit", "checkpoint_sha256":"d"*64, "classifier_sha256":"e"*64}
QUESTION = "What should the next on-call engineer check?"
READING = {"time_ms":10000,"busy":4,"waiting":3}
SNAPSHOT = dict(state="ready",fresh=True,capacity=4,latest=READING,origin="demo",
    source="QA telemetry", observations={"sample_count":50, "first":READING, "latest":READING,
    "start_ms":5000,"end_ms":10000,"peak_waiting":READING},
    forecast={"model":"queue_outlook","scores":[1,-1],"horizon_ms":10000})
CAPTURE = "c"*64

class Fake(MockBoard):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.active = BASE
        self.stream = SNAPSHOT

    def request(self, action, **params):
        self.calls.append((action, params))
        if action == "status":
            return {**super().request(action, **params), "lm_supported":True, "stream_supported":True, "stream_evidence_supported":True}
        if action == "lm_select":
            self.active = EDIT
        if action == "lm_restore":
            self.active = BASE
        if action in ("lm_versions","lm_select","lm_restore"):
            return {"ok":True, "active":self.active, "versions":[BASE, EDIT], "capture_listing_supported":True}
        baseline = {"answer":"Saved original diagnosis.", "version":BASE}
        if action == "lm_captures":
            return {"ok":True,"captures":[{"id":CAPTURE,"question":QUESTION,
                     "captured_at_ms":10000,"reading_time_ms":10000,"baseline":baseline}], "before":None}
        if action == "lm_start":
            return {"ok":True,"job_id":"qa","state":"running","snapshot":SNAPSHOT,"active":self.active,
                    "capture_id":CAPTURE,"baseline":baseline}
        if action == "lm_result":
            return {"ok":True,"job_id":"qa","state":"complete","answer":"Inspect arrival rates and hold times.",
                    "active":self.active,"capture_id":CAPTURE,"baseline":baseline,
                    "model":"flan-t5-small","accelerated_projections":3722,"fabric_projections":3722}
        return super().request(action, **params)

with KitServer(("127.0.0.1", 0), KitHandler) as server, sync_playwright() as p:
    fake = Fake()
    server.kit = Kit(fake)
    host = f"127.0.0.1:{server.server_port}"
    server.allowed_hosts = {host}
    worker = Thread(target=server.serve_forever,daemon=True)
    worker.start()
    chrome = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
    try:
        page = chrome.new_page(viewport={"width":1100,"height":720})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto("http://" + host)
        expect(page.locator("#ask-live")).to_be_disabled()
        expect(page.locator("#connection")).to_contain_text("Disconnected")
        expect(page.locator("#live-examples")).to_contain_text("How many requests")
        page.screenshot(path="/tmp/pinkprocess-merge-ui-disconnected.png")
        page.locator("#host").fill("qa.invalid")
        page.locator("#connect").click()
        expect(page.locator("#connection")).to_contain_text("PYNQ-Z2")
        expect(page.locator("#lm-versions")).to_be_visible()
        expect(page.locator("#saved-incident")).to_be_visible()
        page.locator("#live-question").fill("How many requests are waiting now?")
        page.locator("#ask-live").click()
        expect(page.locator("#live-answer")).to_have_text("Waiting requests: 3.")
        page.locator("#live-question").fill(QUESTION)
        page.locator("#ask-live").click()
        expect(page.locator("#live-answer")).to_have_text("Inspect arrival rates and hold times.", timeout=8000)
        assert any(a == "lm_start" and args.get("question") == QUESTION for a, args in fake.calls)
        assert any(a == "lm_result" for a, _ in fake.calls)
        page.locator("#lm-selection").select_option(label="Saved edit")
        page.locator("#lm-select").click()
        expect(page.locator("#lm-version")).to_contain_text("Saved edit")
        with page.expect_response(lambda response: response.url.endswith("/api/action") and response.request.post_data_json.get("action") == "ask_begin"):
            page.locator("#replay-saved").click()
        expect(page.locator("#live-answer")).to_have_text("Inspect arrival rates and hold times.", timeout=8000)
        assert any(a == "lm_start" and args.get("capture_id") == CAPTURE for a, args in fake.calls)
        page.locator("#lm-restore").click()
        expect(page.locator("#lm-version")).to_contain_text("original")
        page.locator("footer").get_by_role("link",name="CSV analysis").click()
        assert page.locator("#csv-workspace").evaluate("e => e.open")
        page.locator("#query").click()
        expect(page.locator("#result")).to_be_visible()
        assert any(a == "query_batch" for a, _ in fake.calls)
        page.locator("footer").get_by_role("link",name="Board").click()
        expect(page.locator("#board-panel")).to_be_visible()
        page.screenshot(path="/tmp/pinkprocess-merge-ui-connected.png")
        mobile = chrome.new_page(viewport={"width":375,"height":667})
        mobile.on("pageerror", lambda exc: errors.append(str(exc)))
        mobile.goto("http://" + host)
        expect(mobile.locator("#connection")).to_contain_text("PYNQ-Z2")
        nav = mobile.locator("footer")
        nav.get_by_role("link", name="Board").click()
        expect(mobile.locator("#board-panel")).to_be_in_viewport()
        nav.get_by_role("link", name="Ask").click()
        expect(mobile.locator("#live-question")).to_be_in_viewport()
        mobile.screenshot(path="/tmp/pinkprocess-merge-ui-mobile.png")
        assert mobile.evaluate("document.documentElement.scrollWidth === window.innerWidth")
        print("rendered layout", mobile.evaluate("[...document.querySelectorAll('header,main,footer')].map(e=>[e.tagName, e.offsetTop, e.offsetHeight])"))
        assert not errors, errors
        print("Rendered telemetry Ask, FLAN job/poll, saved capture replay, version switch/restore, CSV analysis and mobile layout passed.")
    finally:
        chrome.close()
        server.shutdown()
        worker.join()
