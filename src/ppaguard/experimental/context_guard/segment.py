"""Moved to ppaguard.engine2.context.segment (now part of Engine 2). Re-exported for the experiment scripts."""
from ...engine2.context.segment import *  # noqa: F401,F403
from ...engine2.context import segment as _m

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
