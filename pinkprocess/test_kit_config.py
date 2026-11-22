import tempfile
from pathlib import Path
import unittest

from pinkprocess.kit_config import load_env, parse_env


class ParseEnvTest(unittest.TestCase):
    def test_reads_quotes_comments_and_export(self):
        text = "# note\nexport PYNQ_AUTH0_DOMAIN=\"tenant.us.auth0.com\"\nPYNQ_AUTH0_CLIENT_ID='abc'\nbroken line\n"
        self.assertEqual(parse_env(text), {"PYNQ_AUTH0_DOMAIN": "tenant.us.auth0.com", "PYNQ_AUTH0_CLIENT_ID": "abc"})


class LoadEnvTest(unittest.TestCase):
    def test_existing_values_win_and_first_file_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.env", Path(folder) / "b.env"
            first.write_text("PYNQ_AUTH0_CLIENT_ID=first\nPYNQ_AUTH0_DOMAIN=\n")
            second.write_text("PYNQ_AUTH0_CLIENT_ID=second\nPYNQ_AUTH0_DOMAIN=tenant.auth0.com\nPYNQ_SIM_BOARD=0\n")
            environ = {"PYNQ_SIM_BOARD": "1"}
            load_env([first, Path(folder) / "missing.env", second], environ)
            self.assertEqual(environ, {"PYNQ_SIM_BOARD": "1", "PYNQ_AUTH0_CLIENT_ID": "first",
                                       "PYNQ_AUTH0_DOMAIN": "tenant.auth0.com"})


if __name__ == "__main__":
    unittest.main()
