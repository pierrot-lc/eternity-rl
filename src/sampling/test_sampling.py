import pytest
import torch

from .sampling import (
    epsilon_distributions,
    epsilon_greedy_distributions,
    nucleus_distributions,
)


@pytest.mark.parametrize(
    "distributions, top_p, true_distributions",
    [
        (
            torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            1.0,
            torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
        ),
        (
            torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            0.71,
            torch.FloatTensor([[0.0, 0.0, 0.4285714285714286, 0.5714285714285715]]),
        ),
        (
            torch.FloatTensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.3, 0.4, 0.1, 0.2],
                ]
            ),
            0.71,
            torch.FloatTensor(
                [
                    [0.0, 0.0, 0.4285714285714286, 0.5714285714285715],
                    [0.4285714285714286, 0.5714285714285715, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_nucleus_distributions(
    distributions: torch.Tensor, top_p: float, true_distributions: torch.Tensor
):
    assert torch.allclose(
        nucleus_distributions(distributions, top_p), true_distributions
    )


@pytest.mark.parametrize(
    "distributions, epsilon, true_distributions",
    [
        (
            torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            0.0,
            torch.FloatTensor([[0.0, 0.0, 0.0, 1.0]]),
        ),
        (
            torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            0.05,
            torch.FloatTensor([[0.0125, 0.0125, 0.0125, 0.9625]]),
        ),
        (
            torch.FloatTensor(
                [
                    [0.4, 0.2, 0.3, 0.1],
                    [0.1, 0.2, 0.3, 0.4],
                ]
            ),
            0.05,
            torch.FloatTensor(
                [
                    [0.9625, 0.0125, 0.0125, 0.0125],
                    [0.0125, 0.0125, 0.0125, 0.9625],
                ]
            ),
        ),
    ],
)
def test_greedy_distributions(
    distributions: torch.Tensor, epsilon: float, true_distributions: torch.Tensor
):
    assert torch.allclose(
        epsilon_greedy_distributions(distributions, epsilon), true_distributions
    )


@pytest.mark.parametrize(
    "distributions, epsilon, true_distributions",
    [
        (
            torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            0.0,
            torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
        ),
        (
            torch.FloatTensor([[0.0, 0.0, 0.0, 1.0]]),
            0.05,
            torch.FloatTensor([[0.0125, 0.0125, 0.0125, 0.9625]]),
        ),
        (
            torch.FloatTensor(
                [
                    [0.4, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.4],
                ]
            ),
            0.05,
            torch.FloatTensor(
                [
                    [0.3925, 0.2025, 0.2025, 0.2025],
                    [0.2025, 0.2025, 0.2025, 0.3925],
                ]
            ),
        ),
    ],
)
def test_epsilon_distributions(
    distributions: torch.Tensor, epsilon: float, true_distributions: torch.Tensor
):
    assert torch.allclose(
        epsilon_distributions(distributions, epsilon), true_distributions
    )
