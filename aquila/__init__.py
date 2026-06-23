from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aquila")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
