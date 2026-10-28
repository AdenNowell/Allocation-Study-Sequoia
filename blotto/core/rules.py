"""
Rule definitions and strict constraint validation for the 5-castle Colonel Blotto game.

Rules enforced:
- 5 castles valued at [1, 2, 3, 4, 5] points (total 15 points).
- Every legal submission is a 5-element vector of non-negative integers summing to 100.
- Game 2 rank payouts are [9, 4, 2, 0] for 1st, 2nd, 3rd, and 4th places (total 15 payout units,
  mean 3.75 per player per pod).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

NUM_CASTLES: int = 5
TOTAL_SOLDIERS: int = 100

CASTLE_VALUES: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
CASTLE_VALUES_INT12: NDArray[np.int64] = np.array([12, 24, 36, 48, 60], dtype=np.int64)
TOTAL_CASTLE_POINTS: float = 15.0

GAME2_RANK_PAYOUTS: NDArray[np.float64] = np.array([9.0, 4.0, 2.0, 0.0], dtype=np.float64)
GAME2_MEAN_PAYOUT: float = 3.75


class ValidationError(ValueError):
    """Raised when a soldier allocation violates game constraints."""


def validate_allocation(allocation: Sequence[int | np.integer] | NDArray[np.integer]) -> NDArray[np.int64]:
    """
    Validate a single 5-castle soldier allocation and return a clean int64 NumPy array.

    Requirements:
    1. Exactly 5 elements.
    2. All elements are integer-valued (no fractional soldiers or floats with non-zero decimals).
    3. All elements are non-negative (0 <= x_i <= 100).
    4. Sum of elements equals exactly 100.

    Parameters
    ----------
    allocation : Sequence[int] | NDArray
        Candidate 5-element allocation vector.

    Returns
    -------
    NDArray[np.int64]
        Validated 1D array of shape (5,) with dtype int64.

    Raises
    ------
    ValidationError
        If any constraint is violated.
    """
    if isinstance(allocation, np.ndarray):
        if allocation.ndim != 1 or allocation.shape[0] != NUM_CASTLES:
            raise ValidationError(
                f"Allocation must be a 1D vector of length {NUM_CASTLES}, got shape {allocation.shape}."
            )
        if np.issubdtype(allocation.dtype, np.floating):
            if not np.all(np.isfinite(allocation)) or not np.all(np.equal(np.mod(allocation, 1), 0)):
                raise ValidationError(f"Allocation contains non-integer float values: {allocation}")
        elif not np.issubdtype(allocation.dtype, np.integer):
            raise ValidationError(f"Allocation must have integer dtype, got {allocation.dtype}.")
        arr = allocation.astype(np.int64, copy=True)
    else:
        try:
            items = list(allocation)
        except TypeError as exc:
            raise ValidationError("Allocation must be an iterable sequence of 5 integers.") from exc

        if len(items) != NUM_CASTLES:
            raise ValidationError(f"Allocation must have exactly {NUM_CASTLES} castles, got {len(items)}.")

        for idx, val in enumerate(items):
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
                raise ValidationError(
                    f"Castle {idx + 1} bid must be an integer, got {val!r} ({type(val).__name__})."
                )
        arr = np.array(items, dtype=np.int64)

    if np.any(arr < 0):
        raise ValidationError(f"Negative soldier bids are illegal: {arr.tolist()}")

    if np.any(arr > TOTAL_SOLDIERS):
        raise ValidationError(f"Individual castle bid cannot exceed {TOTAL_SOLDIERS}: {arr.tolist()}")

    total = int(np.sum(arr))
    if total != TOTAL_SOLDIERS:
        raise ValidationError(f"Allocation must sum to {TOTAL_SOLDIERS}, got sum={total} for {arr.tolist()}.")

    return arr


def validate_population_matrix(matrix: NDArray[np.integer] | Sequence[Sequence[int]]) -> NDArray[np.int64]:
    """
    Validate a 2D matrix of shape (N, 5) representing N legal allocations.

    Parameters
    ----------
    matrix : NDArray | Sequence[Sequence[int]]
        2D collection of allocations.

    Returns
    -------
    NDArray[np.int64]
        Validated 2D array of shape (N, 5).
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[1] != NUM_CASTLES:
        raise ValidationError(
            f"Population matrix must have shape (N, {NUM_CASTLES}), got {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValidationError("Population matrix cannot be empty.")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)) or not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValidationError("Population matrix contains non-integer floats.")
    elif not np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        raise ValidationError(f"Population matrix must contain integers, got dtype {arr.dtype}.")

    arr_int = arr.astype(np.int64, copy=True)
    if np.any(arr_int < 0) or np.any(arr_int > TOTAL_SOLDIERS):
        raise ValidationError("Population matrix contains out-of-range bids outside [0, 100].")

    row_sums = np.sum(arr_int, axis=1)
    bad_rows = np.where(row_sums != TOTAL_SOLDIERS)[0]
    if bad_rows.size > 0:
        first_bad = int(bad_rows[0])
        raise ValidationError(
            f"Row {first_bad} sums to {row_sums[first_bad]} instead of {TOTAL_SOLDIERS}: "
            f"{arr_int[first_bad].tolist()}"
        )
    return arr_int
