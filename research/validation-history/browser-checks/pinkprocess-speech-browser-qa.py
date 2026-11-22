from threading import Thread
from playwright.sync_api import sync_playwright, expect
from pinkprocess.kit_app import Kit, KitServer, KitHandler
from pinkprocess.test_kit_app import MockBoard


class IsolatedKit(Kit):
    calls = []

    def action(self, data):
        action = data["action"]
        if not action.startswith("dataset_"):
            return super().action(data)
        self.calls.append(data.copy())
        if action == "dataset_upload":
            assert data["csv"].startswith("time_ms,busy,waiting\n")
            assert data["name"] == "dictation-test.csv"
            return {"dataset_id": "qa", "selected_model": "original", "models": [{"name": "original"}]}
        if action == "dataset_ask":
            assert data["dataset_id"] == "qa"
            return {"state": "complete", "answer": {"generalization":"Generalization evidence",
                    "short_answer":"Short answer evidence", "summary":"Summary evidence",
                    "final":"Final FPGA result"}[data["stage"]]}
        raise AssertionError(action)


with KitServer(("127.0.0.1", 0), KitHandler) as server, sync_playwright() as p:
    server.kit = IsolatedKit(MockBoard())
    host = f"127.0.0.1:{server.server_port}"
    server.allowed_hosts = {host}
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    chrome = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                               headless=True)
    try:
        page = chrome.new_page(viewport={"width": 1100, "height": 750})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.add_init_script('''
            window.voiceCommands = [];
            window.webkit = {messageHandlers: {speechInput: {postMessage(command) {
              voiceCommands.push(command);
              const state = {ready:"ready", start:"requesting", stop:"stopping"}[command];
              window.pinkprocessSpeechInput({state});
            }}}};''')
        page.goto("http://" + host)
        assert page.evaluate("voiceCommands") == ["ready"]
        button = page.locator("#workflow-dictate")
        composer = page.locator("#workflow-question")
        ask = page.locator("#workflow-ask")
        expect(button).to_be_visible()
        expect(button).to_be_enabled()
        composer.fill("Check")
        button.click()
        assert page.evaluate("voiceCommands")[-1] == "start"
        expect(ask).to_be_disabled()
        page.evaluate("pinkprocessSpeechInput({state:'listening', transcript:'the queue'})")
        expect(composer).to_have_value("Check the queue")
        expect(button).to_have_text("Stop dictation")
        button.click()
        assert page.evaluate("voiceCommands")[-1] == "stop"
        page.evaluate("pinkprocessSpeechInput({state:'idle', transcript:'the queue depth', final:true})")
        expect(composer).to_have_value("Check the queue depth")
        assert not IsolatedKit.calls
        page.evaluate("pinkprocessSpeechInput({state:'error', message:'No usable microphone'})")
        expect(page.locator("#workflow-speech-status")).to_have_text("No usable microphone")
        expect(composer).to_have_value("Check the queue depth")
        page.evaluate("pinkprocessSpeechInput({state:'unavailable', message:'Permission denied'})")
        expect(button).to_be_hidden()
        composer.fill("Check the queue depth now")
        csv = ("time_ms,busy,waiting\n" + "".join(f"{i*100},4,3\n" for i in range(60))).encode()
        page.locator("#workflow-upload").set_input_files({"name":"dictation-test.csv","mimeType":"text/csv","buffer":csv})
        expect(page.locator("#workflow-filename")).to_contain_text("60 readings")
        expect(page.locator("#workflow-rows tr")).to_have_count(30)
        expect(page.locator("#tab-table")).to_have_attribute("aria-selected", "true")
        expect(ask).to_be_enabled()
        ask.click()
        expect(page.locator("#panel-final .workflow-output")).to_have_text("Final FPGA result")
        assert [call["stage"] for call in IsolatedKit.calls if call["action"] == "dataset_ask"] == [
            "generalization", "short_answer", "summary", "final"]
        assert all(call.get("question","Check the queue depth now") == "Check the queue depth now"
                   for call in IsolatedKit.calls)
        browser = chrome.new_page()
        browser.goto("http://" + host)
        expect(browser.locator("#workflow-dictate")).to_be_hidden()
        expect(browser.locator("#workflow-question")).to_be_editable()
        assert not errors, errors
        page.screenshot(path="/tmp/pinkprocess-speech-csv-qa.png")
        print("Voice bridge ready/start/partial/stop/final/error/manual and CSV four-stage Ask passed in isolated rendering")
    finally:
        chrome.close()
        server.shutdown()
        worker.join()
