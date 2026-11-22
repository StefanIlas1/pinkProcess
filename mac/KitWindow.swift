import AppKit
import WebKit
import Security
import CryptoKit
import UniformTypeIdentifiers

final class KitWindow: NSObject, NSApplicationDelegate, WKUIDelegate, WKNavigationDelegate, WKScriptMessageHandler {
    private let service = Process()
    private var window: NSWindow!
    private var origin: URL?
    private var closing = false
    private var timer: Timer?
    private var stream: LiveCSV?
    private var streamName = ""
    private var view: WKWebView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let config = WKWebViewConfiguration()
        config.userContentController.add(self, name: "watchCSV")
        config.userContentController.add(self, name: "toggleFullscreen")
        view = WKWebView(frame: .zero, configuration: config)
        view.uiDelegate = self
        view.navigationDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1024, height: 760),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "pinkProcess"
        window.minSize = NSSize(width: 640, height: 480)
        window.contentView = view
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        guard let resources = Bundle.main.resourceURL else { return }
        service.executableURL = resources.appendingPathComponent("backend/kit-backend")
        let pipe = Pipe()
        service.standardOutput = pipe
        service.terminationHandler = { _ in
            DispatchQueue.main.async {
                if !self.closing { self.fail("The local service stopped. Reopen pinkProcess to reconnect.") }
            }
        }
        do {
            try service.run()
        } catch {
            fail(error.localizedDescription)
            return
        }
        DispatchQueue.global().async {
            let data = pipe.fileHandleForReading.availableData
            let address = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let url = URL(string: address), url.scheme == "http", url.host == "127.0.0.1", url.port != nil else { return }
            DispatchQueue.main.async {
                self.origin = url
                self.view.load(URLRequest(url: url))
            }
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url, let origin = origin,
              url.scheme == origin.scheme, url.host == origin.host, url.port == origin.port else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { result in completionHandler(result == .OK ? panel.urls : nil) }
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let source = message.frameInfo.request.url, message.frameInfo.isMainFrame,
              source.scheme == origin?.scheme, source.host == origin?.host, source.port == origin?.port,
              let action = message.body as? String else { return }
        if message.name == "toggleFullscreen", action == "toggle" {
            window.toggleFullScreen(nil)
            return
        }
        guard message.name == "watchCSV" else { return }
        if action == "stop" {
            timer?.invalidate()
            stream = nil
            return
        }
        guard action == "pick" else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.commaSeparatedText]
        panel.allowsMultipleSelection = false
        panel.beginSheetModal(for: window) { response in
            guard response == .OK, let url = panel.url else { return }
            self.timer?.invalidate()
            self.stream = LiveCSV(url: url)
            self.streamName = url.lastPathComponent
            self.notifyPage("window.pinkProcessLiveStatus?.(name)", ["name": self.streamName])
            self.timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in self.poll() }
            self.poll()
        }
    }

    private func notifyPage(_ code: String, _ arguments: [String: Any]) {
        view.callAsyncJavaScript(code, arguments: arguments, in: nil, in: .page, completionHandler: nil)
    }

    private func poll() {
        do {
            if let csv = try stream?.poll() {
                notifyPage("window.pinkProcessLiveCSV?.(csv, name)", ["csv": csv, "name": streamName])
            }
        } catch {
            timer?.invalidate()
            stream = nil
            notifyPage("window.pinkProcessLiveError?.(message)", ["message": error.localizedDescription])
        }
    }

    @objc func pairKit(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= 65536 else {
                    throw CocoaError(.fileReadCorruptFile)
                }
                let data = try Data(contentsOf: url)
                guard let fields = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let host = fields["host"] as? String, !host.isEmpty,
                      let token = fields["token"] as? String,
                      token.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil,
                      let cert = fields["certificate"] as? String else { throw CocoaError(.fileReadCorruptFile) }
                let pem = cert.components(separatedBy: .newlines).filter { !$0.hasPrefix("-----") }.joined()
                guard let der = Data(base64Encoded: pem),
                      SecCertificateCreateWithData(nil, der as CFData) != nil else { throw CocoaError(.fileReadCorruptFile) }
                let storage = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                          appropriateFor: nil, create: true).appendingPathComponent("pinkProcess/kits")
                try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                let id = SHA256.hash(data: der).map { String(format: "%02x", $0) }.joined()
                let target = storage.appendingPathComponent(id + ".json")
                // Stage in a private directory. The credential is private before publication.
                let staging = storage.appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false,
                                                        attributes: [.posixPermissions: 0o700])
                defer { try? FileManager.default.removeItem(at: staging) }
                let temp = staging.appendingPathComponent("pairing")
                try data.write(to: temp)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: temp.path)
                if FileManager.default.fileExists(atPath: target.path) {
                    _ = try FileManager.default.replaceItemAt(target, withItemAt: temp)
                } else { try FileManager.default.moveItem(at: temp, to: target) }
                let alert = NSAlert()
                alert.messageText = "Kit paired"
                alert.informativeText = "Scan for boards to connect to the kit."
                alert.beginSheetModal(for: self.window)
            } catch {
                let alert = NSAlert()
                alert.messageText = "Could not import the kit pairing file"
                alert.informativeText = error.localizedDescription
                alert.beginSheetModal(for: self.window)
            }
        }
    }

    private func fail(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "pinkProcess could not start"
        alert.informativeText = message
        alert.runModal()
        NSApp.terminate(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) {
        closing = true
        if service.isRunning { service.terminate() }
    }
}

let app = NSApplication.shared
let delegate = KitWindow()
app.delegate = delegate
app.setActivationPolicy(.regular)
let menu = NSMenu()
let item = NSMenuItem()
menu.addItem(item)
let actions = NSMenu()
let pairing = actions.addItem(withTitle: "Import Kit Pairing…", action: #selector(KitWindow.pairKit(_:)), keyEquivalent: "i")
pairing.target = delegate
actions.addItem(.separator())
actions.addItem(withTitle: "Quit pinkProcess", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
item.submenu = actions
let editItem = NSMenuItem()
menu.addItem(editItem)
let edit = NSMenu(title: "Edit")
for (title, selector, key) in [("Cut", "cut:", "x"), ("Copy", "copy:", "c"), ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")] {
    edit.addItem(withTitle: title, action: Selector(selector), keyEquivalent: key)
}
editItem.submenu = edit
app.mainMenu = menu
app.run()
