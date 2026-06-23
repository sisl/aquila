import os
import sys

from aquila import __version__

_LIGHT = (167, 178, 196)
_MID   = (57, 72, 99)
_DARK  = (17, 24, 39)
_MUTED = (93, 101, 112)


def _rgb(r: int, g: int, b: int, text: str) -> str:
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return True


def banner() -> str:
    color = _supports_color()

    def w(shade: tuple[int, int, int], s: str) -> str:
        return _rgb(*shade, s) if color else s

    def t(s: str) -> str:
        return _rgb(*_DARK, s) if color else s

    def m(s: str) -> str:
        return _rgb(*_MUTED, s) if color else s

    lines = [
        w(_LIGHT, "  /    ") + t("   _             _ _      "),
        w(_MID,   " //    ") + t("  /_\\  __ _ _  _(_) |__ _ "),
        w(_DARK,  "///    ") + t(" / _ \\/ _` | || | | / _` |"),
        "       " + t("/_/ \\_\\__, |\\_,_|_|_\\__,_|"),
        "       " + t("         |_|"),
        "",
        "       " + m(f"GPU Inference Management  v{__version__}"),
        "",
    ]
    return "\n".join(lines)
