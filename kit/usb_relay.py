"""Relay the paired kit's TLS socket through its already logged-in USB console.

Release the serial device from other programs and leave the board's shell at
its prompt. Run python3 kit/usb_relay.py /dev/cu.usbserial-..., then connect the
Mac app to 127.0.0.1 using its existing pairing. Ctrl-C ends the relay and
restores the shell. The board also restores its shell if the relay disappears.
"""

import argparse
import base64
import inspect
import os
from pathlib import Path
import select
import shlex
import signal
import socket
import sys
import termios
import time
import tty
import zlib


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinkprocess.usb_uart import claimed_port, frame, relay, write_all


READY = b"PPUART/1 ready\r\n"
RESTORED = b"PPUART/1 stopped"


def bootstrap(board_port):
    source = "import os,select,socket,struct,time\n"
    source += "\n".join(inspect.getsource(fn) for fn in (write_all, frame, relay))
    source += f"\nwrite_all(1, {READY!r})\nrelay(0, 1, {board_port})\n"
    payload = base64.b64encode(zlib.compress(source.encode())).decode()
    python = shlex.quote(f"import base64,zlib;exec(zlib.decompress(base64.b64decode({payload!r})))")
    return (f'pp_uart_tty=$(stty -g) && stty raw -echo && python3 -u -c {python}; '
            f'[ -z "$pp_uart_tty" ] || stty "$pp_uart_tty"; '
            "printf '\\nPPUART/%s stopped\\n' 1\n").encode()


def wait_for(fd, marker, seconds, running=lambda: True):
    deadline = time.monotonic() + seconds
    pending = bytearray()
    while time.monotonic() < deadline and running():
        if select.select([fd], [], [], 0.1)[0]:
            data = os.read(fd, 1024)
            if not data:
                return False
            pending.extend(data)
            if marker in pending:
                return True
            del pending[:-len(marker)]
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tty", help="board USB serial device, usually /dev/cu.usbserial-…")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--local-port", type=int, default=8784)
    parser.add_argument("--board-port", type=int, default=8784)
    args = parser.parse_args()
    speed = getattr(termios, f"B{args.baud}", None)
    if speed is None:
        parser.error("Unsupported serial speed.")
    if not all(1 <= port <= 65535 for port in (args.local_port, args.board_port)):
        parser.error("Ports must be between 1 and 65535.")

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (socket.create_server(("127.0.0.1", args.local_port)) as listener,
          claimed_port(args.tty) as fd):
        os.set_blocking(fd, True)
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            settings = termios.tcgetattr(fd)
            settings[2] |= termios.CLOCAL | termios.CREAD
            settings[4] = settings[5] = speed
            termios.tcsetattr(fd, termios.TCSANOW, settings)
            termios.tcflush(fd, termios.TCIFLUSH)
            write_all(fd, bootstrap(args.board_port))
            if not wait_for(fd, READY, 5, lambda: running):
                if running:
                    raise RuntimeError("No UART handshake. Check the board's logged-in shell and baud rate.")
                return
            print(f"Kit TLS forwarded at 127.0.0.1:{args.local_port}. Press Ctrl-C to stop.", flush=True)
            try:
                relay(fd, fd, args.board_port, listener, lambda: running)
            finally:
                try:
                    frame(fd, b"X")
                    if not wait_for(fd, RESTORED, 10):
                        print("No shell restoration notice received.", file=sys.stderr)
                except OSError as error:
                    print(f"Serial link closed during cleanup ({error}).", file=sys.stderr)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        sys.exit(str(exc))
