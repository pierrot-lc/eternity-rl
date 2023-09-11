from itertools import product
from pathlib import Path
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, path

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

color_dict = {
    GRAY: "gray",
    1: "lightcoral",
    2: "tab:blue",
    3: "tab:orange",
    4: "tab:green",
    5: "gold",
    6: "tab:purple",
    7: "tab:brown",
    8: "tab:pink",
    9: "tab:olive",
    10: "tab:cyan",
    11: "deeppink",
    12: "blue",
    13: "slateblue",
    14: "darkslateblue",
    15: "darkviolet",
    16: "teal",
    17: "wheat",
    18: "darkkhaki",
    19: "indigo",
    20: "fuchsia",
    21: "lime",
    22: "rosybrown",
    BLACK: "black",
    RED: "tab:red",
    WHITE: "white",
}


def draw_padding(ax: plt.Axes, x: int, y: int):
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            x + 1,
            y + 1,
            fill=True,
            facecolor=color_dict[GRAY],
            edgecolor=color_dict[BLACK],
        )
    )


def draw_triangles(ax: plt.Axes, x: int, y: int, tile: np.ndarray):
    left_bot = (x, y)
    right_bot = (x + 1, y)
    right_top = (x + 1, y + 1)
    left_top = (x, y + 1)
    middle = (x + 0.5, y + 0.5)
    instructions = [
        path.Path.MOVETO,
        path.Path.LINETO,
        path.Path.LINETO,
        path.Path.CLOSEPOLY,
    ]

    triangle_paths = {
        SOUTH: path.Path(
            [left_bot, middle, right_bot, left_bot],
            instructions,
        ),
        NORTH: path.Path(
            [right_top, middle, left_top, right_top],
            instructions,
        ),
        EAST: path.Path(
            [right_top, middle, right_bot, right_bot],
            instructions,
        ),
        WEST: path.Path(
            [left_bot, middle, left_top, left_bot],
            instructions,
        ),
    }

    for orientation, triangle_path in triangle_paths.items():
        patch = patches.PathPatch(
            triangle_path,
            facecolor=color_dict[tile[orientation]],
            edgecolor=color_dict[BLACK],
        )
        ax.add_patch(patch)


def draw_instance(
    instance: np.ndarray,
    score: float,
    filename: Optional[Path | str] = None,
) -> np.ndarray:
    _, height, width = instance.shape

    # Add padding so that we can draw empty patches.
    height += 2
    width += 2

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for x, y in product(range(width), range(height)):
        if x in [0, width - 1] or y in [0, height - 1]:
            # We are in the padding zone, we can draw the padding.
            draw_padding(ax, x, y)
            continue

        draw_triangles(ax, x, y, instance[:, y - 1, x - 1])  # Remove the padding.

    fig.suptitle(f"Score: {score:.2f}")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    fig.canvas.draw()

    if filename:
        fig.savefig(str(filename))

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image


def draw_gif(
    instances: np.ndarray,
    scores: np.ndarray,
    filename: Path | str,
):
    images = [
        draw_instance(instance, score, filename=None)
        for instance, score in zip(instances, scores)
    ]

    imageio.mimsave(filename, images, duration=500)
