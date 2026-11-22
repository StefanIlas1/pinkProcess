"""Discover paired kits on the LAN and query the selected board over pinned TLS."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
from http.client import HTTPSConnection, HTTPException
import ipaddress
import json
from pathlib import Path
import re
import secrets
import socket
import ssl
import subprocess
import sys
import time

PORT = 8784
DISCOVER = b"pinkprocess-kit/1 "
PAIRINGS = Path.home() / "Library/Application Support/pinkProcess/kits"
HOSTNAME = re.compile(r"(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.?")


def validate_host(host):
    if not isinstance(host, str):
        raise ValueError("Enter a board hostname or IP address.")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        if not HOSTNAME.fullmatch(host):
            raise ValueError("Enter a board hostname or IP address without a URL or port.") from None
        return host


def discovery_proof(token, nonce):
    return hmac.new(token.encode(), DISCOVER + nonce, hashlib.sha256).hexdigest()


def paired_kits(directory):
    kits = []
    for path in sorted(directory.glob("*.json")):
        try:
            kit = json.loads(path.read_text())
            validate_host(kit["host"])
            if not isinstance(kit["certificate"], str) or not re.fullmatch(r"[0-9a-f]{64}", kit["token"]):
                raise ValueError("Invalid kit pairing.")
            kits.append(kit)
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return kits


def lan_hosts(kits):
    if not kits:
        return {}
    nonce = secrets.token_hex(16).encode()
    proofs = {discovery_proof(kit["token"], nonce): kit for kit in kits}
    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        endpoints = {"255.255.255.255"}
        # A Mac can use Wi-Fi and a directly attached Ethernet kit at the same time.
        if sys.platform == "darwin":
            try:
                interfaces = subprocess.check_output(["/sbin/ifconfig", "-a"], text=True, timeout=2)
                endpoints.update(re.findall(r"\bbroadcast (\d+\.\d+\.\d+\.\d+)", interfaces))
            except (OSError, subprocess.SubprocessError):
                pass
        for kit in kits:
            try:
                address = ipaddress.IPv4Address(kit["host"])
            except ipaddress.AddressValueError:
                continue
            endpoints.add(str(address))
        for address in endpoints:
            try:
                sock.sendto(DISCOVER + nonce, (address, PORT))
            except OSError:
                continue
        deadline = time.monotonic() + 0.7
        while time.monotonic() < deadline:
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            try:
                data, (host, _) = sock.recvfrom(128)
            except OSError:
                break
            kit = proofs.get(data.decode("ascii", errors="replace"))
            if kit:
                found[host] = kit
    return found


class BoardOperationError(RuntimeError):
    """A paired board replied to the request but rejected the operation."""


def exchange(host, kit, request, timeout, *, path="/v1/query"):
    context = ssl.create_default_context(cadata=kit["certificate"])
    # Each kit has its own self-signed certificate. It follows the board across DHCP addresses.
    context.check_hostname = False
    conn = HTTPSConnection(host, PORT, context=context, timeout=timeout)
    try:
        conn.request("POST", path, json.dumps(request).encode(),
                     {"Content-Type": "application/json", "Authorization": "Bearer " + kit["token"]})
        response = conn.getresponse()
        body = response.read(128 * 1024 + 1)
        if len(body) > 128 * 1024:
            raise ValueError("Invalid board response size.")
        result = json.loads(body)
        if not isinstance(result, dict) or "ok" not in result:
            raise ValueError("Invalid board response.")
        if response.status != 200 or not result["ok"]:
            raise BoardOperationError(result.get("error", "The board rejected the request."))
        return result
    finally:
        conn.close()


class BoardTransport:
    def __init__(self, directory=PAIRINGS):
        self.directory = directory
        self.usb_relay = None
        self.usb_pairing = None
        self.usb_paused = False
        self.close()

    def start_usb_relay(self):
        if self.usb_relay is None or self.usb_relay.stopped.is_set():
            from .usb_uart import UsbRelay

            self.usb_relay = UsbRelay(PORT)
            self.usb_relay.start()
        return self.usb_relay

    def reconnect_usb(self):
        if sys.platform != "darwin" or self.usb_paused:
            return None
        kits = paired_kits(self.directory)
        if self.usb_pairing is not None:
            kits = [kit for kit in kits if all(kit[key] == self.usb_pairing[key] for key in ("certificate", "token"))]
        if not kits:
            return None
        relay = self.start_usb_relay()
        owns_relay = relay.ready.is_set()
        if not owns_relay and self.usb_pairing is None:
            return None
        for kit in kits:
            try:
                status = exchange("127.0.0.1", kit, {"op": "status"}, 1.2)
            except (OSError, HTTPException, ValueError, RuntimeError):
                continue
            if status.get("backend") == "PYNQ-Z2":
                self.host, self.kit = "127.0.0.1", kit
                self.usb_pairing = kit
                return status
        if owns_relay:
            relay.rejected.set()
        return None

    def stop_usb(self):
        self.usb_paused = True
        if self.usb_relay is not None:
            self.usb_relay.stopped.set()

    def discover(self):
        kits = paired_kits(self.directory)
        self.usb_paused = False
        relay = None
        usb_ready_deadline = time.monotonic() + 3
        if sys.platform == "darwin" and kits:
            from .usb_uart import matching_uart_ports

            if matching_uart_ports():
                relay = self.start_usb_relay()
        candidates = {kit["host"]: kit for kit in kits}
        candidates.update(lan_hosts(kits))
        candidates.pop("127.0.0.1", None)

        def probe(item, timeout=1.2):
            host, kit = item
            try:
                status = exchange(host, kit, {"op": "status"}, timeout)
                if status.get("backend") == "PYNQ-Z2":
                    return {"host": host, "name": "pinkProcess · PYNQ-Z2"}
            except (OSError, HTTPException, ValueError, RuntimeError):
                pass
            return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            boards = [board for board in pool.map(probe, candidates.items()) if board]
        relay_ready = relay is not None and relay.ready.wait(max(0, usb_ready_deadline - time.monotonic()))
        if relay_ready or not boards:
            deadline = time.monotonic() + 1.5
            usb = None
            for kit in kits:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                usb = probe(("127.0.0.1", kit), min(1.2, remaining))
                if usb:
                    usb["name"] = "pinkProcess · PYNQ-Z2 (local relay)"
                    boards.append(usb)
                    break
            if relay_ready and usb is None:
                relay.rejected.set()
        return boards

    def connect(self, host):
        self.close()
        host = validate_host(host)
        kits = sorted(paired_kits(self.directory), key=lambda kit: kit["host"] != host)
        if not kits:
            raise RuntimeError("This Mac has no paired kits. Import the kit pairing file first.")
        for kit in kits:
            try:
                status = exchange(host, kit, {"op": "status"}, 3)
            except ssl.SSLCertVerificationError:
                continue
            except (OSError, HTTPException, ValueError) as exc:
                raise RuntimeError(f"Cannot reach the board ({exc}). Check Ethernet, the board address and macOS Local Network access.") from exc
            if status.get("backend") != "PYNQ-Z2":
                raise RuntimeError("The connected device did not identify a PYNQ-Z2 model.")
            self.host, self.kit = host, kit
            if host == "127.0.0.1":
                self.usb_pairing = kit
                self.usb_paused = False
            else:
                self.stop_usb()
            return status
        raise RuntimeError("This board's certificate does not match a kit paired to this Mac.")

    def request(self, action, **params):
        if not self.kit:
            raise RuntimeError("Connect a board first.")
        try:
            return exchange(self.host, self.kit, {**params, "op": action}, 30)
        except (OSError, HTTPException, ValueError) as exc:
            self.close()
            raise RuntimeError("Lost the board connection. Reconnect to read its current model version.") from exc

    def close(self):
        self.host = None
        self.kit = None
