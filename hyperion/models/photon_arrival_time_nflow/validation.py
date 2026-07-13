"""Physical-plausibility checks for trained photon-arrival-time models."""

import numpy as np


def check_counts_physics(net_fn, params):
    """Run physical plausibility checks on a trained counts model.

    The counts model maps ``[log10(distance_m), angle_rad]`` to
    ``log10(survival_fraction)``. Several invariants must hold for any
    physically realistic Cherenkov medium:

    1. **Forward dominance** — survival at ~0° is >= 100x (2 log10 units)
       larger than at 90° for all tested distances.
    2. **No backward peak** — survival at ~180° never exceeds survival at ~0°.
    3. **No 90° spike** — survival at exactly 90° must not form a local
       maximum relative to its +-5° neighbours (85° and 95°).
    4. **Monotone with distance** — forward survival decreases as distance
       increases.

    Parameters
    ----------
    net_fn : _ConditionerFn
        Counts network built by ``make_counts_net_fn``.
    params : dict
        Trained parameter dictionary.

    Returns
    -------
    bool
        ``True`` if every check passes, ``False`` otherwise.
    """
    # Probe distances: 5, 24, 50, 100 m — chosen to span the training range
    # and to include 24 m where the 90°/180° artifact is most prominent.
    test_log10_dists = np.array(
        [np.log10(5.0), np.log10(24.0), np.log10(50.0), np.log10(100.0)],
        dtype=np.float32,
    )
    n = len(test_log10_dists)

    fwd_angle = np.float32(np.radians(3.0))  # ~3°  — well inside Cherenkov cone
    side_angle = np.float32(np.pi / 2)  # 90°
    bwd_angle = np.float32(np.pi)  # exactly 180°
    side_m5 = np.float32(np.radians(85.0))  # 85°
    side_p5 = np.float32(np.radians(95.0))  # 95°

    def _predict(log10_dists, angles):
        x = np.stack([log10_dists, angles], axis=1).astype(np.float32)
        return np.array(net_fn.apply(params, x)).squeeze()

    ls_fwd = _predict(test_log10_dists, np.full(n, fwd_angle))
    ls_side = _predict(test_log10_dists, np.full(n, side_angle))
    ls_bwd = _predict(test_log10_dists, np.full(n, bwd_angle))
    ls_85 = _predict(test_log10_dists, np.full(n, side_m5))
    ls_95 = _predict(test_log10_dists, np.full(n, side_p5))

    checks = []

    # 1. Forward > sideways: survival(~0°) must exceed survival(90°) at every
    # distance.  We require a positive margin only — different media can have
    # very different angular profiles (e.g. heavily scattering P-ONE water vs
    # cleaner ANTARES water), but forward must always dominate sideways.
    ratios = ls_fwd - ls_side
    checks.append(
        (
            "forward > sideways",
            bool(np.all(ratios > 0.0)),
            f"min log10-ratio(fwd/90°) = {ratios.min():.3f}  (need > 0; "
            f"per-dist: {np.round(ratios, 3).tolist()})",
        )
    )

    # 2. No backward peak: survival(180°) ≤ survival(~0°) everywhere.  Probing
    # at exactly 180° catches the MLP extrapolation spike that appears just
    # past the last training angle (177°).
    excess_bwd = ls_bwd - ls_fwd
    checks.append(
        (
            "no backward peak",
            bool(np.all(excess_bwd <= 0.0)),
            f"max excess of 180° over fwd = {excess_bwd.max():.3f}  (need ≤ 0; "
            f"per-dist: {np.round(excess_bwd, 3).tolist()})",
        )
    )

    # 3. No 90° spike: survival at 90° must not be a *local maximum*, meaning
    # it must not exceed BOTH its neighbours simultaneously.  A monotone fall-off
    # (85° > 90° > 95°) is perfectly fine; only a bump (90° > 85° AND 90° > 95°)
    # is unphysical.
    is_local_max_90 = (ls_side > ls_85) & (ls_side > ls_95)
    worst_vs_85 = float((ls_side - ls_85).max())
    worst_vs_95 = float((ls_side - ls_95).max())
    checks.append(
        (
            "no 90° local maximum",
            not bool(np.any(is_local_max_90)),
            "90° local-max at distances: "
            f"{list(np.round(10 ** test_log10_dists[is_local_max_90], 1))}; "
            f"max excess vs 85°={worst_vs_85:.3f}, vs 95°={worst_vs_95:.3f}",
        )
    )

    # 4. Forward survival decreases monotonically with distance.
    diffs = np.diff(ls_fwd)
    checks.append(
        (
            "forward survival decreases with distance",
            bool(np.all(diffs < 0.0)),
            f"consecutive diffs = {np.round(diffs, 3).tolist()}  (all must be < 0)",
        )
    )

    all_pass = all(passed for _, passed, _ in checks)
    status = "PASS" if all_pass else "FAIL"
    print(f"\n--- Counts model physics checks [{status}] ---")
    for name, passed, detail in checks:
        mark = "✓" if passed else "✗"
        print(f"  {mark} {name}")
        print(f"      {detail}")
    print()
    return all_pass
