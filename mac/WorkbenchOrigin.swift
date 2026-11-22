import Foundation

/// WebKit privileges belong only to the root document served by the child process.
func isWorkbenchPage(_ candidate: URL?, origin: URL?) -> Bool {
    guard let candidate, let origin, origin.scheme == "http", origin.host == "127.0.0.1",
          let port = origin.port, (1...65535).contains(port), origin.path == "/",
          origin.query == nil, origin.user == nil, origin.password == nil else { return false }
    return candidate.scheme == origin.scheme && candidate.host == origin.host && candidate.port == port
        && candidate.path == "/" && candidate.query == nil && candidate.user == nil && candidate.password == nil
}
