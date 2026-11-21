"""Framed transport for the explicitly provisioned PYNQ USB console."""

import argparse
from contextlib import contextmanager
import errno
import fcntl
from pathlib import Path
import os
import re
import select
import socket
import struct
import subprocess
import sys
import termios
from threading import Event, Thread
import time
import tty


APPLIANCE_READY = b"\r\nPPUART/2 ready; type +++ PP ADMIN for login\r\n"
APPLIANCE_OPEN = b"\0PPUART/2 open\0"
APPLIANCE_CONNECTED = b"\0PPUART/2 connected\0"
ADMIN = b"+++ PP ADMIN\r"
FTDI_UART = re.compile(r"cu\.usbserial-[A-Za-z0-9._-]+_tul1")


def write_all(fd, data):
    while data:
        count = os.write(fd, data)
        if not count:
            raise OSError("Serial connection closed while writing.")
        data = data[count:]


def frame(fd, kind, channel=0, data=b""):
    write_all(fd, struct.pack("!cII", kind, channel, len(data)) + data)


def relay(read_fd, write_fd, board_port, listener=None, running=lambda: True, *, keepalive=False):
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
            if listener is None or keepalive:
                if now - last_seen > 8:
                    return
            if listener is not None and now - last_ping >= 1:
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
                        if keepalive:
                            frame(write_fd, b"p")
                        continue
                    if kind == b"p" and listener is not None and keepalive:
                        continue
                    if kind == b"X" and listener is None:
                        return
                    if kind == b"O" and listener is None:
                        if not channel or channel in sockets or len(sockets) >= 16:
                            raise OSError("Invalid or excessive UART channels.")
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
                if len(sockets) >= 16:
                    peer.close()
                else:
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


def matching_uart_ports(directory=Path("/dev")):
    return sorted(path for path in directory.glob("cu.usbserial-*_tul1") if FTDI_UART.fullmatch(path.name))


@contextmanager
def open_port(path):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        yield fd
    finally:
        os.close(fd)


def port_is_open(path):
    """Check for an existing Mac serial client that predates our shared lease."""
    if sys.platform != "darwin":
        return False
    counterpart = path.with_name(re.sub(r"^cu\.", "tty.", path.name))
    try:
        result = subprocess.run(["/usr/sbin/lsof", "-t", "--", str(path), str(counterpart)],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=1)
        return bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


@contextmanager
def claimed_port(path, *, lock_dir=None, observe=False):
    """Claim the UART before opening it or touching shared termios settings."""
    path = Path(path).resolve()
    name = re.sub(r"^(cu|tty)\.", "", path.name)
    if lock_dir is None:
        lock_dir = Path.home() / "Library/Application Support/pinkProcess/serial-locks"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (lock_dir / (name + ".lock")).open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OSError(errno.EBUSY, "The serial console is already in use", str(path)) from exc
        if observe and port_is_open(path):
            raise OSError(errno.EBUSY, "The serial console is already in use", str(path))
        with open_port(path) as fd:
            # Also honor the previous uploader's device lock and prevent new terminal opens.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(fd, termios.TIOCEXCL)
            try:
                yield fd
            finally:
                try:
                    fcntl.ioctl(fd, termios.TIOCNXCL)
                except OSError:
                    pass  # A disconnected USB device has already released exclusivity.


@contextmanager
def raw_port(path, *, claim=False, blocking=True, observe=False):
    with (claimed_port(path, observe=observe) if claim else open_port(path)) as fd:
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            settings = termios.tcgetattr(fd)
            settings[2] &= ~getattr(termios, "CRTSCTS", 0)
            settings[2] |= termios.CLOCAL | termios.CREAD
            settings[4] = settings[5] = termios.B115200
            termios.tcsetattr(fd, termios.TCSANOW, settings)
            os.set_blocking(fd, blocking)
            yield fd
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, old)
            except termios.error:
                pass  # The USB device can disappear during cleanup.


def read_markers(fd, markers, timeout, running=lambda: True):
    pending = b""
    deadline = time.monotonic() + timeout
    while running() and time.monotonic() < deadline:
        if not select.select([fd], [], [], 0.1)[0]:
            continue
        data = os.read(fd, 1024)
        if not data:
            raise OSError("The serial link closed.")
        pending += data
        for marker in markers:
            if marker in pending:
                return marker
        pending = pending[-max(map(len, markers)):]
    return None


def appliance_session(fd, board_port, running=lambda: True):
    """Advertise until a Mac opens a TLS relay or a local operator requests login."""
    while running():
        write_all(fd, APPLIANCE_READY)
        requested = read_markers(fd, (APPLIANCE_OPEN, ADMIN), 1, running)
        if requested == ADMIN:
            return True
        if requested == APPLIANCE_OPEN:
            write_all(fd, APPLIANCE_CONNECTED)
            try:
                relay(fd, fd, board_port, running=running, keepalive=True)
            except OSError:
                pass  # Re-advertise after a disconnected or invalid peer.
    return False


def attach(fd, port, ready, running):
    """Listen for the appliance banner before sending anything on a serial device."""
    if not read_markers(fd, (APPLIANCE_READY,), 2, running):
        return
    with socket.create_server(("127.0.0.1", port)) as listener:
        write_all(fd, APPLIANCE_OPEN)
        if not read_markers(fd, (APPLIANCE_CONNECTED,), 2, running):
            return
        ready.set()
        try:
            relay(fd, fd, 0, listener, running, keepalive=True)
        finally:
            ready.clear()
            try:
                frame(fd, b"X")
            except OSError:
                pass


class UsbRelay:
    """Reopen the recognized Mac UART whenever the explicitly enabled kit reappears."""

    def __init__(self, port, *, ports=matching_uart_ports, serial=None):
        self.port = port
        self.ports = ports
        self.serial = serial if serial is not None else lambda path: raw_port(path, claim=True, observe=True)
        self.ready = Event()
        self.rejected = Event()
        self.stopped = Event()
        self.running = Thread(target=self.watch, daemon=True, name="PYNQ USB relay")

    def start(self):
        self.running.start()

    def watch(self):
        while not self.stopped.is_set():
            for path in self.ports():
                if self.stopped.is_set():
                    return
                self.rejected.clear()
                try:
                    with self.serial(path) as fd:
                        attach(fd, self.port, self.ready,
                               lambda: not self.stopped.is_set() and not self.rejected.is_set())
                except (OSError, termios.error):
                    continue
            self.stopped.wait(1)


def run_board(terminal, port):
    """Use the UART before login and hand it to ordinary Linux login on request."""
    if not re.fullmatch(r"ttyPS\d+", terminal):
        raise ValueError("Expected a PYNQ PS UART device name.")
    while True:
        with raw_port(Path("/dev") / terminal) as fd:
            login_requested = appliance_session(fd, port)
        if login_requested:
            subprocess.run(["/sbin/agetty", "--noissue", "--noclear", "-L", "--timeout", "90",
                            terminal, "115200", "vt100"], check=False, start_new_session=True)


if __name__ == "__main__":
    from .kit_transport import PORT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("terminal", help="Linux console device, e.g. ttyPS0")
    run_board(parser.parse_args().terminal, PORT)

