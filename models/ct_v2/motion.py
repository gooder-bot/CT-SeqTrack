"""Continuous-time motion prior used by CT-SeqTrack v2.

The validated transition encoder already lives in ``models.dynamics``.  This
paper-facing name intentionally reuses that implementation so the v2 path has
one motion definition instead of introducing another subtly different
velocity target.
"""

from models.dynamics import DynamicsEncoder


class ContinuousTimeMotionEncoder(DynamicsEncoder):
    """Encode box transitions in metres/second and query them at real ``dt``.

    The public forward contract is kept compatible with ``DynamicsEncoder``:

    ``(trajectory_feature, velocity, query_displacement, valid_transition)``.

    Keeping the contract stable lets legacy checkpoints remain readable while
    the v2 configuration and documentation use one unambiguous module name.
    """

