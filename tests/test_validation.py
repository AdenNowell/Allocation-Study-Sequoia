"""Unit tests for soldier allocation constraint validation."""

from __future__ import annotations

import numpy as np
import pytest

from blotto.core.rules import ValidationError, validate_allocation, validate_population_matrix


def test_valid_allocations() -> None:
    v1 = validate_allocation([10, 17, 25, 45, 3])
    assert v1.shape == (5,)
    assert v1.dtype == np.int64
    assert int(np.sum(v1)) == 100

    v2 = validate_allocation(np.array([0, 0, 0, 0, 100], dtype=np.int32))
    assert v2.tolist() == [0, 0, 0, 0, 100]

    v3 = validate_allocation([13, 21, 3, 60, 3])
    assert v3.tolist() == [13, 21, 3, 60, 3]


def test_rejects_wrong_sum() -> None:
    with pytest.raises(ValidationError, match="sum to 100"):
        validate_allocation([10, 17, 25, 45, 4])  # sums to 101

    with pytest.raises(ValidationError, match="sum to 100"):
        validate_allocation([10, 16, 25, 45, 3])  # sums to 99


def test_rejects_negative_bids() -> None:
    with pytest.raises(ValidationError, match="Negative"):
        validate_allocation([-1, 18, 25, 45, 13])


def test_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError, match="5"):
        validate_allocation([25, 25, 25, 25])

    with pytest.raises(ValidationError, match="5"):
        validate_allocation([20, 20, 20, 20, 10, 10])


def test_rejects_floats_and_booleans() -> None:
    with pytest.raises(ValidationError):
        validate_allocation([10.5, 16.5, 25, 45, 3])  # type: ignore[list-item]

    with pytest.raises(ValidationError):
        validate_allocation([True, 16, 25, 55, 3])


def test_validate_population_matrix() -> None:
    mat = validate_population_matrix(
        [
            [10, 17, 25, 45, 3],
            [13, 21, 3, 60, 3],
            [0, 0, 0, 0, 100],
        ]
    )
    assert mat.shape == (3, 5)

    with pytest.raises(ValidationError, match="Row 1 sums to 99"):
        validate_population_matrix(
            [
                [10, 17, 25, 45, 3],
                [13, 20, 3, 60, 3],
            ]
        )
