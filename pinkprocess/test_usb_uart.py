from contextlib import contextmanager, suppress
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import pty
from queue import Queue
import shutil
import socket
import ssl
import subprocess
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from . import kit_transport
from .usb_uart import ADMIN, APPLIANCE_READY, UsbRelay, appliance_session, attach, claimed_port, matching_uart_ports


class UsbUartTest(unittest.TestCase):
    def test_cooperating_clients_claim_the_uart_before_opening_either_alias(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock_dir = root / "locks"
            master, slave = pty.openpty()
            try:
                callout = root / "cu.usbserial-test_tul1"
                dialin = root / "tty.usbserial-test_tul1"
                opened = []

                @contextmanager
                def open_test_terminal(path):
                    opened.append(path)
                    fd = os.open(os.ttyname(slave), os.O_RDWR | os.O_NOCTTY)
                    try:
                        yield fd
                    finally:
                        os.close(fd)

                with patch("pinkprocess.usb_uart.open_port", open_test_terminal):
                    with claimed_port(callout, lock_dir=lock_dir):
                        with self.assertRaises(OSError) as busy, claimed_port(dialin, lock_dir=lock_dir):
                            self.fail("A competing client opened the same UART.")
                        self.assertEqual(busy.exception.errno, errno.EBUSY)
                    with claimed_port(dialin, lock_dir=lock_dir):
                        self.assertEqual(opened, [callout, dialin])
            finally:
                os.close(master)
                os.close(slave)

    def test_probe_observes_older_serial_clients_before_opening(self):
        with (TemporaryDirectory() as directory,
              patch("pinkprocess.usb_uart.port_is_open", return_value=True) as observed,
              patch("pinkprocess.usb_uart.open_port") as opened):
            path = Path(directory) / "cu.usbserial-test_tul1"
            with self.assertRaises(OSError) as busy, claimed_port(path, lock_dir=Path(directory), observe=True):
                self.fail("The automatic probe opened an occupied UART.")
            self.assertEqual(busy.exception.errno, errno.EBUSY)
            observed.assert_called_once_with(path.resolve())
            opened.assert_not_called()

    def test_only_digilent_uart_interface_is_selected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ("cu.usbserial-1234_tul0", "cu.usbserial-1234_tul1", "tty.usbserial-1234_tul1",
                         "cu.usbserial-other", "cu.debug-console"):
                (path / name).touch()
            self.assertEqual(matching_uart_ports(path), [path / "cu.usbserial-1234_tul1"])

    def test_probe_sends_nothing_to_normal_login(self):
        host, board = socket.socketpair()
        stopped = Event()
        ready = Event()
        with host, board:
            thread = Thread(target=attach, args=(host.fileno(), 0, ready, lambda: not stopped.is_set()))
            thread.start()
            try:
                board.sendall(b"pynq login: ")
                board.settimeout(0.2)
                with self.assertRaises(TimeoutError):
                    board.recv(1)
                self.assertFalse(ready.is_set())
            finally:
                stopped.set()
                thread.join(1)
            self.assertFalse(thread.is_alive())

    def test_admin_escape_leaves_framing_for_a_normal_login(self):
        host, board = socket.socketpair()
        result = Queue()
        with host, board:
            thread = Thread(target=lambda: result.put(appliance_session(board.fileno(), 0)), daemon=True)
            thread.start()
            host.settimeout(1)
            self.assertIn(APPLIANCE_READY, host.recv(1024))
            host.sendall(ADMIN)
            self.assertTrue(result.get(timeout=1))
            thread.join(1)

    @unittest.skipUnless(shutil.which("openssl"), "OpenSSL required for the paired relay test")
    def test_tls_pairing_survives_simulated_disconnect_and_board_restart(self):
        token = "a" * 64

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                authorized = self.headers.get("Authorization") == "Bearer " + token
                self.send_response(200 if authorized else 403)
                self.end_headers()
                self.wfile.write(b'{"ok":true,"backend":"PYNQ-Z2"}' if authorized else b'{"ok":false}')

            def log_message(self, *_):
                pass

        with TemporaryDirectory() as directory:
            cert, key = Path(directory) / "cert.pem", Path(directory) / "key.pem"
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                            "-subj", "/CN=kit-test", "-keyout", str(key), "-out", str(cert)],
                           check=True, capture_output=True)
            pairing = dict(certificate=cert.read_text(), token=token, host="board.local")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
                server.socket = context.wrap_socket(server.socket, server_side=True)
                server_thread = Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                try:
                    self.check_replug(server.server_port, pairing)
                finally:
                    server.shutdown()
                    server_thread.join(1)

    def check_replug(self, board_port, pairing):
        incoming = Queue()

        @contextmanager
        def serial(_):
            with incoming.get_nowait() as endpoint:
                yield endpoint.fileno()

        with socket.create_server(("127.0.0.1", 0)) as placeholder:
            port = placeholder.getsockname()[1]
        monitor = UsbRelay(port, ports=lambda: ["uart"] if not incoming.empty() else [], serial=serial)
        monitor.start()
        try:
            with patch.object(kit_transport, "PORT", port):
                for _ in range(2):
                    mac_end, board_end = socket.socketpair()
                    with board_end:
                        def serve_board():
                            with suppress(OSError, ValueError):
                                appliance_session(board_end.fileno(), board_port)

                        board = Thread(target=serve_board, daemon=True)
                        board.start()
                        incoming.put(mac_end)
                        self.assertTrue(monitor.ready.wait(4))
                        status = kit_transport.exchange("127.0.0.1", pairing, {"op": "status"}, 2)
                        self.assertEqual(status["backend"], "PYNQ-Z2")
                        with self.assertRaises(RuntimeError):
                            kit_transport.exchange("127.0.0.1", {**pairing, "token": "b" * 64}, {"op": "status"}, 2)
                        board_end.shutdown(socket.SHUT_RDWR)
                    board.join(2)
                    deadline = monotonic() + 3
                    while monitor.ready.is_set() and monotonic() < deadline:
                        sleep(0.01)
                    self.assertFalse(monitor.ready.is_set())
        finally:
            monitor.stopped.set()
            monitor.running.join(3)
        self.assertFalse(monitor.running.is_alive())


if __name__ == "__main__":
    unittest.main()
