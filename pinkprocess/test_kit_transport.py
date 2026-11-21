import json
from pathlib import Path
import ssl
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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
        with patch("pinkprocess.kit_transport.lan_hosts", return_value={"192.168.2.100": self.pair}), \
                patch("pinkprocess.kit_transport.exchange",
                      side_effect=lambda host, *_: {"backend": "PYNQ-Z2" if host.endswith(".100") else None}):
            self.assertEqual(self.transport.discover(),
                             [{"host": "192.168.2.100", "name": "pinkProcess · PYNQ-Z2"}])
        self.assertNotEqual(discovery_proof("a" * 64, b"nonce"), discovery_proof("b" * 64, b"nonce"))
        self.assertNotEqual(discovery_proof("a" * 64, b"nonce"), discovery_proof("a" * 64, b"other"))

    def test_urls_are_rejected(self):
        for value in ("http://192.168.2.99", "user@board", "board:8784", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_host(value)


if __name__ == "__main__":
    unittest.main()
