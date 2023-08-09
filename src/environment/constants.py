"""Some constant variables to use across the program."""
from pathlib import Path

ENV_DIR = Path("./instances")
ENV_ORDERED = [
    "eternity_trivial_A.txt",
    "eternity_trivial_B.txt",
    "eternity_A.txt",
    "eternity_B.txt",
    "eternity_C.txt",
    "eternity_D.txt",
    "eternity_E.txt",
    "eternity_complet.txt",
]

N_SIDES = 4

# Ids of the sides of the tiles.
# The y-axis has its origin at the bottom.
# The x-axis has its origin at the left.
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ORIGIN_NORTH = 0
ORIGIN_SOUTH = 1
ORIGIN_WEST = 2
ORIGIN_EAST = 3
