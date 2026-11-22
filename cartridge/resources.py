"""Read cartridge assets from the checkout or their installed wheel locations."""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def asset_directory(package_name: str, source_name: str) -> Path:
    bundled = PACKAGE / package_name
    return bundled if bundled.is_dir() else ROOT / source_name


APP = asset_directory("_app", "app")
DECKS = asset_directory("_decks", "decks")
WEIGHTS = asset_directory("_weights", "cartridges")


def output_path(source_relative: str, installed_relative: str) -> Path:
    if (ROOT / "pyproject.toml").is_file():
        return ROOT / source_relative
    return Path.cwd() / installed_relative
