"""Generate random instances of the puzzle."""
import torch


from .gym import EAST, NORTH, SOUTH, WEST


def random_perfect_instances(
    size: int,
    n_classes: int,
    n_instances: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate a random solved instance.

    ---
    Args:
        size: The width and height of the instance.
        n_classes: The number of different classes.
        n_instances: How many instances to be generated.

    ---
    Returns:
        The generated solved instances.
            Shape of [n_instances, N_SIDES, size, size].
    """
    instances = torch.zeros((n_instances, 4, size, size), dtype=torch.long)

    for y in range(size):
        for x in range(size):
            tiles = torch.randint(
                1, n_classes, size=(n_instances, 4), generator=generator
            )

            # Fill the south side to be equal to the north value of the under tile.
            under_y = max(0, y - 1)
            under_tiles = instances[..., under_y, x]
            tiles[:, SOUTH] = under_tiles[:, NORTH]

            # Fill the west side to be equal to the east value of the left tile.
            left_x = max(0, x - 1)
            left_tiles = instances[..., y, left_x]
            tiles[:, WEST] = left_tiles[:, EAST]

            instances[..., y, x] = tiles

    # Place empty walls at the borders.
    instances[:, SOUTH, 0, :] = 0
    instances[:, NORTH, -1, :] = 0
    instances[:, WEST, :, 0] = 0
    instances[:, EAST, :, -1] = 0

    return instances
