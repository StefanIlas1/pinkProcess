import AVFoundation
import Foundation
import Speech

/// Owns one microphone session. Entry points and callbacks are serialized on the main queue.
final class SpeechInput {
    private enum State { case idle, requesting, listening, stopping }

    private let publish: ([String: Any]) -> Void
    private var recognizer: SFSpeechRecognizer?
    private var session: UUID?
    private var state = State.idle
    private var audio: AVAudioEngine?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var timer: Timer?
    private var configurationObserver: NSObjectProtocol?
    private var transcript = ""

    init(publish: @escaping ([String: Any]) -> Void) {
        self.publish = publish
    }

    func handle(_ command: String) {
        switch command {
        case "ready":
            guard state == .idle else { return }
            if let message = unavailableReason() {
                send("unavailable", message: message)
            } else {
                send("ready")
            }
        case "start":
            start()
        case "stop":
            stop()
        default:
            break
        }
    }

    private func unavailableReason() -> String? {
        guard #available(macOS 10.15, *) else {
            return "On-device dictation requires macOS 10.15 or later."
        }
        if AVCaptureDevice.authorizationStatus(for: .audio) == .denied ||
            AVCaptureDevice.authorizationStatus(for: .audio) == .restricted {
            return "Allow pinkProcess to use the microphone in System Settings to dictate."
        }
        let status = SFSpeechRecognizer.authorizationStatus()
        if status == .denied || status == .restricted {
            return "Allow pinkProcess to use Speech Recognition in System Settings to dictate."
        }
        if recognizer == nil { recognizer = SFSpeechRecognizer() }
        guard let recognizer else {
            return "On-device dictation is unavailable for the current speech language."
        }
        if status == .authorized {
            guard recognizer.supportsOnDeviceRecognition else {
                return "On-device dictation is unavailable for the current speech language."
            }
            guard recognizer.isAvailable else {
                return "Speech Recognition is currently unavailable. Try again or type your question."
            }
        }
        return nil
    }

    private func start() {
        guard state == .idle else { return }
        if let message = unavailableReason() {
            send("unavailable", message: message)
            return
        }
        let id = UUID()
        session = id
        state = .requesting
        transcript = ""
        send("requesting")
        switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized:
            requestMicrophone(for: id)
        case .notDetermined:
            SFSpeechRecognizer.requestAuthorization { [weak self] status in
                DispatchQueue.main.async {
                    guard let self, self.session == id else { return }
                    if status == .authorized {
                        self.requestMicrophone(for: id)
                    } else {
                        self.finish("unavailable", message: "Speech Recognition permission was not granted.")
                    }
                }
            }
        default:
            finish("unavailable", message: "Speech Recognition permission is unavailable.")
        }
    }

    private func requestMicrophone(for id: UUID) {
        if let message = unavailableReason() {
            finish("unavailable", message: message)
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            beginRecording(id)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self, self.session == id else { return }
                    if granted {
                        self.beginRecording(id)
                    } else {
                        self.finish("unavailable", message: "Microphone permission was not granted.")
                    }
                }
            }
        default:
            finish("unavailable", message: "Microphone permission is unavailable.")
        }
    }

    private func beginRecording(_ id: UUID) {
        guard session == id else { return }
        guard #available(macOS 10.15, *), let recognizer,
              recognizer.isAvailable, recognizer.supportsOnDeviceRecognition else {
            finish("unavailable", message: "On-device dictation is currently unavailable.")
            return
        }
        let engine = AVAudioEngine()
        let node = engine.inputNode
        let format = node.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            finish("error", message: "No usable microphone input was found.")
            return
        }

        let recognition = SFSpeechAudioBufferRecognitionRequest()
        recognition.requiresOnDeviceRecognition = true
        recognition.shouldReportPartialResults = true
        request = recognition
        task = recognizer.recognitionTask(with: recognition) { [weak self] result, error in
            let text = result?.bestTranscription.formattedString
            let final = result?.isFinal == true
            let failed = error != nil
            DispatchQueue.main.async {
                guard let self, self.session == id else { return }
                if let text { self.transcript = text }
                if final || failed {
                    if final || self.state == .stopping {
                        self.finish("idle", final: true)
                    } else {
                        self.finish("error", message: "Dictation ended unexpectedly. You can edit or finish your question by typing.",
                                    final: true)
                    }
                } else if let text {
                    self.send(self.state == .stopping ? "stopping" : "listening", transcript: text)
                }
            }
        }
        audio = engine
        node.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            recognition.append(buffer)
        }
        do {
            engine.prepare()
            try engine.start()
        } catch {
            finish("error", message: "The microphone could not be started. Check the selected audio input.")
            return
        }
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            guard let self, self.session == id, self.state == .listening else { return }
            self.finish("error", message: "The audio input changed. Start dictation again or continue typing.",
                        final: true)
        }
        state = .listening
        send("listening")
        schedule(after: 60, for: id) { [weak self] in self?.stop() }
    }

    private func stop() {
        if state == .requesting {
            finish("idle")
            return
        }
        guard state == .listening, let id = session else { return }
        state = .stopping
        send("stopping")
        stopAudio()
        schedule(after: 3, for: id) { [weak self] in self?.finish("idle", final: true) }
    }

    private func schedule(after seconds: TimeInterval, for id: UUID, _ perform: @escaping () -> Void) {
        timer?.invalidate()
        let deadline = Timer(timeInterval: seconds, repeats: false) { [weak self] _ in
            guard self?.session == id else { return }
            perform()
        }
        timer = deadline
        RunLoop.main.add(deadline, forMode: .common)
    }

    private func stopAudio() {
        if let observer = configurationObserver {
            NotificationCenter.default.removeObserver(observer)
            configurationObserver = nil
        }
        if let audio {
            audio.stop()
            audio.inputNode.removeTap(onBus: 0)
            self.audio = nil
            request?.endAudio()
        }
    }

    func cancel() {
        session = nil
        state = .idle
        timer?.invalidate()
        timer = nil
        stopAudio()
        task?.cancel()
        task = nil
        request = nil
        transcript = ""
    }

    private func finish(_ state: String, message: String? = nil, final: Bool = false) {
        let completed = final ? transcript : nil
        cancel()
        send(state, message: message, transcript: completed, final: final)
    }

    private func send(_ state: String, message: String? = nil, transcript: String? = nil, final: Bool = false) {
        var event: [String: Any] = ["state": state]
        event["message"] = message
        event["transcript"] = transcript
        if final { event["final"] = true }
        publish(event)
    }
}
