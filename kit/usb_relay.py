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
import select
import shlex
import signal
import socket
import struct
import sys
import termios
import time
import tty
import zlib


def write_all(fd, data):
    while data:
        count = os.write(fd, data)
        if not count:
            raise OSError("Serial connection closed while writing.")
        data = data[count:]


def frame(fd, kind, channel=0, data=b""):
    write_all(fd, struct.pack("!cII", kind, channel, len(data)) + data)


def relay(read_fd, write_fd, board_port, listener=None, running=lambda: True):
    header = struct.Struct("!cII")
    received = bytearray()
    sockets = {}
    next_channel = 0
    last_seen = last_ping = time.monotonic()

    def close(channel, notify=False):
        peer = sockets.pop(channel, None)
        if peer is not None:
            peer.close()
        if notify:
            frame(write_fd, b"C", channel)

    try:
        while running():
            now = time.monotonic()
            if listener is None:
                if now - last_seen > 8:
                    return
            elif now - last_ping >= 1:
                frame(write_fd, b"P")
                last_ping = now
            sources = [read_fd, *sockets.values()]
            if listener is not None:
                sources.append(listener)
            readable, _, _ = select.select(sources, [], [], 0.2)
            if read_fd in readable:
                part = os.read(read_fd, 8192)
                if not part:
                    return
                last_seen = time.monotonic()
                received.extend(part)
                while len(received) >= header.size:
                    kind, channel, size = header.unpack_from(received)
                    if size > 4096:
                        raise OSError("Invalid UART frame length.")
                    end = header.size + size
                    if len(received) < end:
                        break
                    data = bytes(received[header.size:end])
                    del received[:end]
                    if kind == b"P" and listener is None:
                        continue
                    if kind == b"X" and listener is None:
                        return
                    if kind == b"O" and listener is None:
                        try:
                            sockets[channel] = socket.create_connection(("127.0.0.1", board_port), 0.5)
                        except OSError:
                            frame(write_fd, b"C", channel)
                    elif kind == b"C":
                        close(channel)
                    elif kind == b"D":
                        if channel in sockets:
                            try:
                                sockets[channel].sendall(data)
                            except OSError:
                                close(channel, notify=True)
                    else:
                        raise OSError("Invalid UART frame type.")
            if listener is not None and listener in readable:
                peer, _ = listener.accept()
                peer.settimeout(1)
                next_channel = (next_channel % 0xffffffff) + 1
                sockets[next_channel] = peer
                frame(write_fd, b"O", next_channel)
            for channel, peer in list(sockets.items()):
                if peer not in readable:
                    continue
                try:
                    data = peer.recv(4096)
                except OSError:
                    data = b""
                if data:
                    frame(write_fd, b"D", channel, data)
                else:
                    close(channel, notify=True)
    finally:
        for peer in sockets.values():
            peer.close()


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
    with socket.create_server(("127.0.0.1", args.local_port)) as listener:
        fd = os.open(args.tty, os.O_RDWR | os.O_NOCTTY)
        try:
            old = termios.tcgetattr(fd)
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
            try:
                if 'old' in locals():
                    termios.tcsetattr(fd, termios.TCSANOW, old)
            finally:
                os.close(fd)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        sys.exit(str(exc))
