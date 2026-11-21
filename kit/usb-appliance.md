# USB appliance setup

The PYNQ USB connection exposes the Digilent FTDI JTAG and UART interfaces.
The Mac app probes only callout devices matching `/dev/cu.usbserial-*_tul1`
and reads until it sees the appliance relay banner. It sends no shell commands.
The relay forwards bytes to the kit's loopback HTTPS endpoint. The same paired
certificate and bearer token authenticate the board on USB and on Ethernet.
The Mac app retries a previously selected USB kit while its window is open.
The Mac app, engineer relay and uploader share a console lock. The app skips
an occupied UART and retries once the engineering tool releases it.

The existing engineer relay in `kit/usb_relay.py` is for a normally logged-in
serial shell. The appliance service below replaces the `agetty` instance on
one board console at boot. Enable it only with the board administrator's
approval and a second administrative route or a tested recovery plan.

## Enable on a provisioned board

Confirm the UART name in the kernel command line and the active serial getty.
The commands below use `ttyPS0` as an example. Copy
`kit/pinkprocess-usb-uart@.service` to the board after installing a pinkProcess
release that includes `pinkprocess.usb_uart`. From an authenticated board shell,
install and enable the unit for the verified console. Reboot when ready to
switch the console over.

```sh
cat /proc/cmdline
systemctl status serial-getty@ttyPS0.service
sudo install -m 0644 pinkprocess-usb-uart@.service /etc/systemd/system/
sudo systemctl mask serial-getty@ttyPS0.service
sudo systemctl daemon-reload
sudo systemctl enable pinkprocess-usb-uart@ttyPS0.service
sudo reboot
```

Start the Mac app with its existing kit pairing, then connect the USB UART. The
app detects the relay and authenticates the kit. Disconnecting in the app stops
automatic probing for the rest of that app session. Reopening the app restarts
probing. A kit using the original console configuration continues to work by
Ethernet or the logged-in engineer relay.

## Administrative console and rollback

Quit the Mac app, open a serial terminal on the UART at 115200 baud, and wait
for `PPUART/2 ready`. Type `+++ PP ADMIN` and press Return. The board starts an
ordinary Linux login prompt. Log in with the board's existing credentials.
Logging out restores the appliance relay. If a login prompt is left unused,
the board resumes advertising after 90 seconds.

From an authenticated board shell, restore the standard console at the next
reboot with the verified device name.

```sh
sudo systemctl disable pinkprocess-usb-uart@ttyPS0.service
sudo systemctl unmask serial-getty@ttyPS0.service
sudo systemctl daemon-reload
sudo reboot
```
