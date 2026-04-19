"""Sample baseline plugin: rolling-ball (morphological opening on a 1D signal).

This is a minimal reference implementation demonstrating the plugin
contract. The radius parameter controls the ball size in sample units.

Not a scientific replacement for pybaselines.morphological — provided as
an example of how to hook in a custom algorithm.
"""

import numpy as np
from scipy.ndimage import grey_opening


def _rolling_ball(x, y, params):
    radius = int(params.get("radius", 50))
    size = max(3, 2 * radius + 1)
    # Grey opening ≈ erosion followed by dilation
    baseline = grey_opening(y, size=size)
    return baseline


def register(registry):
    registry["register"](
        short_code="rollball",
        display_name="Rolling Ball (plugin)",
        compute_fn=_rolling_ball,
        default_params={"radius": 50},
    )
