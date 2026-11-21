"""Copy a file into the logged-in PYNQ user's home over the Mac USB console."""

import argparse
import array
import base64
import fcntl
import hashlib
import os
from pathlib import Path
import re
import select
import shlex
import sys
import termios
import time
import tty
import zlib


# The board flushes each acknowledgment before switching baud or restoring its shell.
RECEIVER = """
import hashlib, os, select, sys, tempfile, termios, tty
from pathlib import Path
name, size, digest, baud = sys.argv[1:]
fd = os.open('/dev/tty', os.O_RDWR)
old = termios.tcgetattr(fd)
partial = None
try:
    tty.setraw(fd)
    os.write(fd, b'PP-UPLOAD-READY\\n')
    termios.tcdrain(fd)
    config = termios.tcgetattr(fd)
    config[4] = config[5] = getattr(termios, 'B' + baud)
    termios.tcsetattr(fd, termios.TCSADRAIN, config)
    directory = Path.home() / 'pinkprocess-transfer'
    directory.mkdir(mode=0o700, exist_ok=True)
    total = int(size)
    check = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
        partial = Path(output.name)
        while total:
            block = min(total, 1024)
            remaining = block
            while remaining:
                if not select.select([fd], [], [], 8)[0]:
                    raise TimeoutError('upload timed out')
                data = os.read(fd, remaining)
                if not data:
                    raise IOError('console closed')
                output.write(data)
                check.update(data)
                remaining -= len(data)
            total -= block
            os.write(fd, b'+')
        output.flush()
        os.fsync(output.fileno())
    if check.hexdigest() != digest:
        raise ValueError('checksum mismatch')
    os.replace(partial, directory / name)
    os.write(fd, b'PP-VERIFIED ' + digest.encode() + b'\\n')
    termios.tcdrain(fd)
finally:
    if partial and partial.exists():
        partial.unlink()
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    os.close(fd)
"""


def wait_read(fd, timeout):
    if not select.select([fd], [], [], timeout)[0]:
        raise TimeoutError("USB console did not acknowledge the transfer.")
    return os.read(fd, 8192)


def write_all(fd, data):
    while data:
        if not select.select([], [fd], [], 8)[1]:
            raise TimeoutError("USB console stopped accepting bytes.")
        data = data[os.write(fd, data):]


def upload(device, source, baud):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', source.name):
        raise ValueError("Use a plain filename for the transfer.")
    size = source.stat().st_size
    check = hashlib.sha256()
    with source.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            check.update(part)
    digest = check.hexdigest()
    code = base64.b64encode(zlib.compress(RECEIVER.encode())).decode()
    loader = "import base64,zlib;exec(zlib.decompress(base64.b64decode(%r)))" % code
    command = shlex.join(['python3', '-c', loader, source.name, str(size), digest, str(baud)])
    fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    old = termios.tcgetattr(fd)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tty.setraw(fd)
        config = termios.tcgetattr(fd)
        config[4] = config[5] = termios.B115200
        termios.tcsetattr(fd, termios.TCSANOW, config)
        termios.tcflush(fd, termios.TCIFLUSH)
        write_all(fd, b'\x15' + command.encode() + b'\r')
        reply = b''
        while b'PP-UPLOAD-READY\n' not in reply:
            reply = (reply + wait_read(fd, 8))[-8192:]
        if sys.platform == 'darwin':
            fcntl.ioctl(fd, 0x80045402, array.array('I', [baud]))  # Darwin IOSSIOSPEED
        else:
            config[4] = config[5] = getattr(termios, 'B' + str(baud))
            termios.tcsetattr(fd, termios.TCSANOW, config)
        time.sleep(.10)
        reply = b''
        started = time.monotonic()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(1024), b''):
                write_all(fd, block)
                reply = wait_read(fd, 8)
                if not reply.startswith(b'+'):
                    raise IOError('Unexpected USB console acknowledgment.')
                reply = reply[1:]
        while b'\n' not in reply:
            reply += wait_read(fd, 8)
        if reply.splitlines()[0] != b'PP-VERIFIED ' + digest.encode():
            raise IOError('Board did not confirm the expected checksum.')
        print(f'{size} bytes in {time.monotonic()-started:.2f}s; verified ~/pinkprocess-transfer/{source.name} {digest}')
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old)
        os.close(fd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('device')
    p.add_argument('file', type=Path)
    p.add_argument('--baud', type=int, choices=[115200, 460800, 921600, 1000000, 1500000, 2000000], default=115200)
    args = p.parse_args()
    upload(args.device, args.file, args.baud)
