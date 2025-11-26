import numpy as np
from amuse.units import quantities


def tau_for_migration_duration(a_initial, a_final, duration):
    """
    Convert a desired point-to-point migration duration into the exponential
    e-folding time constant used by the current migration force.

    The implemented kick follows d(ln a)/dt = -1/tau_a, so
    a(t) = a_initial * exp(-t / tau_a). To migrate from a_initial to a_final
    in a specified duration, set tau_a to:

        tau_a = duration / ln(a_initial / a_final)

    Sign convention:
    - Inward migration (a_final < a_initial) -> ln(...) > 0 -> tau_a > 0
    - Outward migration (a_final > a_initial) -> ln(...) < 0 -> tau_a < 0

    Args:
        a_initial (units.length): Starting semi-major axis.
        a_final (units.length): Target semi-major axis.
        duration (units.time): Desired time to go from a_initial to a_final.

    Returns:
        units.time: The tau_a value that produces the requested migration time.
    """
    for name, val in (("a_initial", a_initial), ("a_final", a_final)):
        if not hasattr(val, "unit") or not hasattr(val, "value_in"):
            raise ValueError(f"{name} must be an AMUSE quantity with length units.")
    if not hasattr(duration, "unit") or not hasattr(duration, "value_in"):
        raise ValueError("duration must be an AMUSE quantity with time units.")

    ratio = a_initial / a_final
    if ratio <= 0:
        raise ValueError("Semi-major axes must be positive.")
    try:
        ratio_val = ratio.value_in(1 | ratio.unit)
    except Exception:
        ratio_val = float(ratio)
    log_ratio = np.log(ratio_val)
    if np.isclose(log_ratio, 0.0):
        raise ValueError("Initial and final semi-major axes are too close; tau_a would diverge.")

    return duration / log_ratio
