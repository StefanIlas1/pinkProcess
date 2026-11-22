import json
from pathlib import Path
import ssl
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .kit_transport import BoardTransport, discovery_proof, validate_host


class KitTransportTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pair = {"host": "192.168.2.99", "certificate": "test", "token": "a" * 64}
        self.directory_path = Path(self.directory.name)
        (self.directory_path / "kit.json").write_text(json.dumps(self.pair))
        self.transport = BoardTransport(self.directory_path)

    def test_changed_address_keeps_identity_and_disconnect_forgets_session(self):
        with patch("pinkprocess.kit_transport.exchange",
                   return_value={"ok": True, "backend": "PYNQ-Z2"}) as exchange:
            self.transport.connect("192.168.2.100")
            self.transport.request("query_batch", features=[[1, 2, 3]])
            self.assertEqual(exchange.call_args.args[:3],
                             ("192.168.2.100", self.pair, {"op": "query_batch", "features": [[1, 2, 3]]}))
        self.transport.close()
        with self.assertRaisesRegex(RuntimeError, "Connect"):
            self.transport.request("status")

    def test_certificate_mismatch_and_missing_pairing(self):
        with patch("pinkprocess.kit_transport.exchange", side_effect=ssl.SSLCertVerificationError):
            with self.assertRaisesRegex(RuntimeError, "certificate"):
                self.transport.connect("different.local")
        (self.directory_path / "kit.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "no paired kits"):
            self.transport.connect("192.168.2.99")

    def test_responding_paired_boards_only(self):
        with patch("pinkprocess.kit_transport.sys.platform", "linux"), \
                patch("pinkprocess.kit_transport.lan_hosts", return_value={"192.168.2.100": self.pair}), \
                patch("pinkprocess.kit_transport.exchange",
                      side_effect=lambda host, *_: {"backend": "PYNQ-Z2" if host.endswith(".100") else None}):
            self.assertEqual(self.transport.discover(),
                             [{"host": "192.168.2.100", "name": "pinkProcess · PYNQ-Z2"}])
        self.assertNotEqual(discovery_proof("a" * 64, b"nonce"), discovery_proof("b" * 64, b"nonce"))
        self.assertNotEqual(discovery_proof("a" * 64, b"nonce"), discovery_proof("a" * 64, b"other"))

    def test_automatic_usb_reopens_only_its_selected_tls_pairing(self):
        ready, stopped, rejected = Event(), Event(), Event()
        self.transport.usb_relay = SimpleNamespace(ready=ready, stopped=stopped, rejected=rejected)
        status = {"ok": True, "backend": "PYNQ-Z2"}
        with (patch("pinkprocess.kit_transport.sys.platform", "darwin"),
              patch("pinkprocess.kit_transport.exchange", return_value=status) as exchange):
            self.assertIsNone(self.transport.reconnect_usb())
            exchange.assert_not_called()
            ready.set()
            self.assertEqual(self.transport.reconnect_usb(), status)
            exchange.assert_called_once_with("127.0.0.1", self.pair, {"op": "status"}, 1.2)
            self.transport.close()
            ready.clear()
            self.assertEqual(self.transport.reconnect_usb(), status)
            self.transport.close()
            ready.set()
            exchange.reset_mock()
            other = dict(host="other.local", certificate="other", token="b" * 64)
            (self.directory_path / "other.json").write_text(json.dumps(other))
            exchange.side_effect = ssl.SSLCertVerificationError
            self.assertIsNone(self.transport.reconnect_usb())
            self.assertIsNone(self.transport.host)
            self.assertTrue(rejected.is_set())
            exchange.assert_called_once_with("127.0.0.1", self.pair, {"op": "status"}, 1.2)

    def test_find_boards_starts_usb_even_after_explicit_disconnect(self):
        relay = SimpleNamespace(ready=Mock(), stopped=Event(), rejected=Event(), start=Mock())
        relay.ready.wait.return_value = True
        self.transport.stop_usb()

        def status(host, kit, request, timeout):
            self.assertEqual(kit, self.pair)
            self.assertEqual(request, {"op": "status"})
            return {"ok": True, "backend": "PYNQ-Z2"} if host == "127.0.0.1" else {"ok": False}

        with (patch("pinkprocess.kit_transport.sys.platform", "darwin"),
              patch("pinkprocess.kit_transport.lan_hosts", return_value={}),
              patch("pinkprocess.kit_transport.exchange", side_effect=status),
              patch("pinkprocess.usb_uart.matching_uart_ports", return_value=["fixture"]),
              patch("pinkprocess.usb_uart.UsbRelay", return_value=relay)):
            self.assertIsNone(self.transport.reconnect_usb())
            relay.start.assert_not_called()
            self.assertEqual(self.transport.discover(),
                             [{"host": "127.0.0.1", "name": "pinkProcess · PYNQ-Z2 (local relay)"}])
            relay.start.assert_called_once_with()
            self.assertGreater(relay.ready.wait.call_args.args[0], 0)
            self.assertLessEqual(relay.ready.wait.call_args.args[0], 3)
            self.assertIsNone(self.transport.host)

    def test_urls_are_rejected(self):
        for value in ("http://192.168.2.99", "user@board", "board:8784", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_host(value)


if __name__ == "__main__":
    unittest.main()
