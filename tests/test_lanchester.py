import math

import pytest

from lanchester import run_battle


def test_run_battle_r_wins_mirror_case():
    result = run_battle(80, 100, 0.3, 0.5)

    assert result["winner"] == "R", "R should win (beta*R0^2 > alpha*S0^2)"
    assert result["remaining"] == pytest.approx(78.5, abs=0.1), "R remaining is incorrect"
    assert result["time"] == pytest.approx(1.85, abs=0.01), "Battle time is incorrect"


def test_run_battle_draw():
    result = run_battle(100, 100, 0.5, 0.5)

    assert result["winner"] == "draw", "Equal strength should end in a draw"


def test_run_battle_raises_when_side_not_above_threshold():
    with pytest.raises(ValueError, match="greater than threshold"):
        run_battle(1.0, 80, 0.5, 0.3, threshold=1.0)


@pytest.mark.parametrize(
    "S0, R0, alpha, beta, threshold",
    [
        (100, 80, 0.5, 0.3, 1.0),   # S wins
        (80, 100, 0.3, 0.5, 1.0),   # R wins
        (50, 30, 1.0, 2.0, 1.0),    # S wins, other coefficients
        (100, 80, 0.5, 0.3, 5.0),   # larger threshold
    ],
)
def test_run_battle_winner_remaining_matches_invariant(S0, R0, alpha, beta, threshold):
    result = run_battle(S0, R0, alpha, beta, threshold)

    Q = alpha * S0**2 - beta * R0**2
    if Q > 0:
        expected = math.sqrt(Q / alpha)
        shift_bound = (beta * threshold**2 / alpha) / (2 * expected)
    else:
        expected = math.sqrt(-Q / beta)
        shift_bound = (alpha * threshold**2 / beta) / (2 * expected)

    # The loser stops at `threshold`, not at 0, so the true remaining is
    # sqrt(expected^2 + c*threshold^2); by concavity of sqrt the gap to
    # `expected` is at most shift_bound. 1e-4 covers the solver error.
    assert result["remaining"] == pytest.approx(expected, abs=shift_bound + 1e-4), (
        "Winner's remaining should match the conserved-quantity prediction"
    )