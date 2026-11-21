import AppKit
import WebKit

// Replay the native app's live-file callbacks into its shipping WebKit page.
// Measure when an updated answer has had a frame in which to be displayed.
final class Measurement: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    let view = WKWebView()
    var window: NSWindow!
    var failed = true
    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 800),
                          styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "pinkProcess forecast test"
        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        view.navigationDelegate = self
        view.load(URLRequest(url: URL(string: CommandLine.arguments[1])!))
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        do {
            let path = URL(fileURLWithPath: CommandLine.arguments[2])
            let csv = try String(contentsOf: path, encoding: .utf8)
            let script = """
            const sleep = ms => new Promise(r => setTimeout(r, ms));
            const frame = () => new Promise(r => requestAnimationFrame(r));
            for(let i=0;busy && i<500;i++) await sleep(20);
            if (!status || busy) throw Error("Kit did not connect");
            const lines = csv.trim().split(/\\r?\\n/);
            el("capacity").value = String(capacity);
            window.pinkProcessLiveStatus("measured pool");
            const out = [];
            const start = performance.now();
            for(let t=5000; t<=50000; t+=1000) {
                const due = start + (t-5000) + 50;
                await sleep(Math.max(0, due - performance.now()));
                window.pinkProcessLiveCSV(lines.slice(0,t/100+2).join("\\n")+"\\n","measured pool");
                const deadline = performance.now()+4000;
                while (result?.dataset?.end_ms !== t) {
                    if (performance.now() > deadline) throw Error(el("notice").textContent || "Missing result");
                    await sleep(5);
                }
                el("answer").scrollIntoView({block:"nearest"});
                await frame(); await frame();
                const text = el("answer").textContent;
                const late = performance.now() - (start+t-5000);
                if (late > 1000) throw Error("Presentation overran next sample at "+t);
                if (!text || el("result").hidden || !text.includes(seconds(t))) throw Error("Stale result");
                out.push({time_ms:t, display_delay_ms:late, text, version:result.version,
                          hash:result.weights_sha256, score:result.rows.at(-1).score});
            }
            return JSON.stringify(out);
            """
            view.callAsyncJavaScript(script, arguments: ["csv":csv, "capacity":Int(CommandLine.arguments[4])!],
                                     in: nil, in: .page) { result in
                do {
                    guard let text = try result.get() as? String else { throw CocoaError(.fileReadCorruptFile) }
                    try text.write(toFile: CommandLine.arguments[3], atomically: true, encoding: .utf8)
                    self.failed = false
                } catch { FileHandle.standardError.write(Data((String(describing:error)+"\n").utf8)) }
                exit(self.failed ? 1 : 0)
            }
        } catch { fputs("\(error)\n", stderr); exit(1) }
    }
}
let app = NSApplication.shared
let measurement = Measurement()
app.delegate = measurement
app.setActivationPolicy(.accessory)
app.run()
exit(measurement.failed ? 1 : 0)
