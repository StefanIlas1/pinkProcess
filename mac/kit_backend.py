"""Private loopback server for the native pinkProcess window."""

from pinkprocess.kit_app import make_server

if __name__ == "__main__":
    with make_server() as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()
