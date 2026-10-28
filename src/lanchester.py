import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, y, alpha, beta):
    S, R = y
    return [-beta * R, -alpha * S]


def run_battle(S0, R0, alpha, beta, threshold=1.0):
    """Lanchester square-law battle: runs until one side drops to `threshold`.

    Returns a dict:
      winner    : "S", "R" or "draw"
      remaining : survivors on the winning side (on a draw, the level at stop time)
      time      : time at which the battle ended
    """
    if min(S0, R0) <= threshold:
        raise ValueError("S0 and R0 must be greater than threshold")

    # solve_ivp passes `args` to events too, so they need the same signature as rhs
    def s_dead(t, y, alpha, beta):
        return y[0] - threshold

    def r_dead(t, y, alpha, beta):
        return y[1] - threshold

    for ev in (s_dead, r_dead):
        ev.terminal = True    # stop integrating when the event fires
        ev.direction = -1     # fire only when the value crosses zero going down

    # Solution is A*e^(kt) + B*e^(-kt), k = sqrt(alpha*beta). A draw ends at
    # ln(S0/threshold)/k; the extra 40/k covers near-draws, where a tiny growing
    # mode delays the end (imbalance down to ~e^-40, below double precision).
    k = np.sqrt(alpha * beta)
    t_max = (np.log(max(S0, R0) / threshold) + 40) / k

    sol = solve_ivp(
        rhs, (0, t_max), [S0, R0],
        args=(alpha, beta),
        events=(s_dead, r_dead),
        rtol=1e-8, atol=1e-8,
    )

    # status 1 means a terminal event fired; anything else is not a finished battle
    if sol.status != 1:
        raise RuntimeError(f"Battle did not finish: {sol.message}")

    S_end, R_end = sol.y[:, -1]
    t_end = sol.t[-1]

    # The winner is decided by the conserved quantity alpha*S^2 - beta*R^2
    Q = alpha * S0**2 - beta * R0**2
    if abs(Q) <= 1e-9 * (alpha * S0**2 + beta * R0**2):
        winner, remaining = "draw", max(S_end, R_end)
    elif Q > 0:
        winner, remaining = "S", S_end
    else:
        winner, remaining = "R", R_end

    return {"winner": winner, "remaining": float(remaining), "time": float(t_end)}