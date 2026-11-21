import Foundation

/// Follow a selected append-only file. Send complete records and bound both
/// file reads and model input, including when attaching to an old recording.
final class LiveCSV {
    private let url: URL
    private var delivered: String?

    init(url: URL) { self.url = url }

    func poll() throws -> String? {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let size = try file.seekToEnd()
        guard size > 0 else { return nil }
        try file.seek(toOffset: 0)
        let prefix = try file.read(upToCount: 256) ?? Data()
        guard let firstLF = prefix.firstIndex(of: 10),
              let header = String(data: prefix[..<firstLF], encoding: .utf8) else { return nil }
        let columns = header.trimmingCharacters(in: CharacterSet(charactersIn: "\u{feff}\r"))
        guard columns == "time_ms,busy,waiting" else {
            throw NSError(domain: "pinkProcess", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "The watched CSV must start with time_ms,busy,waiting."])
        }
        let limit = UInt64(256 * 1024)
        let offset = size > limit ? size - limit : 0
        try file.seek(toOffset: offset)
        let bytes = try file.read(upToCount: Int(size - offset)) ?? Data()
        guard let lastLF = bytes.lastIndex(of: 10) else { return nil }
        // Skip the header or the incomplete first record of the tail read.
        let complete = bytes[..<lastLF].split(separator: 10, omittingEmptySubsequences: false)
        let rows = complete.dropFirst().suffix(2097)
        guard rows.count >= 50 else { return nil }
        let text = header + "\n" + rows.map { String(decoding: $0, as: UTF8.self) }.joined(separator: "\n") + "\n"
        guard delivered != text else { return nil }
        delivered = text
        return text
    }
}
