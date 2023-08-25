"""Generate random instances of the puzzle."""
import torch

from .constants import EAST, N_SIDES, NORTH, SOUTH, WEST


def random_perfect_instances(
    size: int,
    n_classes: int,
    n_instances: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate a random solved instance.
    The instance's device is the same as the device of the given generator.

    ---
    Args:
        size: The width and height of the instance.
        n_classes: The number of different classes (wall class included).
        n_instances: How many instances to be generated.

    ---
    Returns:
        The generated solved instances.
            Shape of [n_instances, N_SIDES, size, size].
    """
    device = generator.device
    instances = torch.zeros(
        (n_instances, N_SIDES, size, size), dtype=torch.long, device=device
    )

    for y in range(size):
        for x in range(size):
            tiles = torch.randint(
                low=1,
                high=n_classes,
                size=(n_instances, N_SIDES),
                generator=generator,
                device=device,
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
