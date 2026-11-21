import base64
import hashlib
import os
from pathlib import Path
import pty
import select
import sys
import tempfile
import unittest
import zlib

from kit.serial_upload import RECEIVER


class FrameRecoveryTest(unittest.TestCase):
    def test_receiver_discards_damaged_frame_and_acknowledges_duplicate(self):
        data = bytes(range(256)) * 12
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            pid, master = pty.fork()
            if pid == 0:
                os.environ['HOME'] = tmp
                os.execl(sys.executable, sys.executable, '-c', RECEIVER, 'model', str(len(data)), digest, '115200')
            wire = os.fdopen(master, 'r+b', buffering=0)
            try:
                def read_line():
                    self.assertTrue(select.select([wire], [], [], 5)[0])
                    return wire.readline().strip()
                self.assertEqual(read_line(), b'PP-UPLOAD-READY')
                def frame(number, payload):
                    return (f'\nPPD {number} {zlib.crc32(payload):08x} '.encode() +
                            base64.b64encode(payload) + b'\n')
                packet = frame(0, data[:2048])
                wire.write(packet[:-11] + b'\n' + packet)
                self.assertEqual(read_line(), b'K0')
                wire.write(packet)
                self.assertEqual(read_line(), b'K0')
                wire.write(frame(1, data[2048:]))
                self.assertEqual(read_line(), b'K1')
                self.assertEqual(read_line(), b'PP-VERIFIED ' + digest.encode())
                self.assertEqual((Path(tmp) / 'pinkprocess-transfer/model').read_bytes(), data)
            finally:
                wire.close()
                os.waitpid(pid, 0)
