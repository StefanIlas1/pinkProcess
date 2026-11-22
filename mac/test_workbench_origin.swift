import Foundation

@main
struct WorkbenchOriginTest {
    static func main() {
        let root = URL(string: "http://127.0.0.1:48179/")!
        precondition(isWorkbenchPage(root, origin: root))
        precondition(isWorkbenchPage(URL(string: root.absoluteString + "#board-panel"), origin: root))
        for untrusted in [
            "https://127.0.0.1:48179/",
            "http://127.0.0.1:48180/",
            "http://localhost:48179/",
            "http://127.0.0.1.evil.invalid:48179/",
            "http://127.0.0.1:48179/api/setup",
            "http://127.0.0.1:48179/?request=listen",
            "http://user@127.0.0.1:48179/",
            "http://127.0.0.1:48179@evil.invalid/",
        ] {
            precondition(!isWorkbenchPage(URL(string: untrusted), origin: root), untrusted)
        }
        precondition(isSignInStart(URL(string: "http://127.0.0.1:48179/auth/login"), origin: root))
        precondition(!isSignInStart(URL(string: "http://127.0.0.1:48180/auth/login"), origin: root))
        precondition(!isSignInStart(URL(string: "http://127.0.0.1:48179/auth/login?x=1"), origin: root))
        precondition(!isSignInStart(URL(string: "http://localhost:48179/auth/login"), origin: root))
        precondition(!isWorkbenchPage(root, origin: nil))
        precondition(!isWorkbenchPage(nil, origin: root))
        let other = URL(string: "http://localhost:48179/")!
        precondition(!isWorkbenchPage(other, origin: other))
    }
}
