"""Local Polytopia control API (screenshot + xdotool, not an official game API)."""

from .driver import (
    PolytopiaError,
    activate,
    back,
    click,
    confirm,
    end_turn,
    find_window,
    hud,
    pixel,
    press,
    screenshot,
)
from .observe import observe
from .play import plan, play_n, step
