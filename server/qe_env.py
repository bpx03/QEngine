"""Environment switches for the server (the engines' C++ side: src/env.h, same rules).

An empty or whitespace-only value means "unset, use the default": a Docker Compose pass-through such as
`QE_X: "${QE_X:-}"` exports an empty string. A value that does not parse prints a one-line warning (once per
variable) and uses the default instead of failing or silently turning into 0.
"""
import os
import sys

_warned = set()


def _warn(name, val, kind):
    if name not in _warned:
        _warned.add(name)
        print(f"[env] warning: {name}={val!r} is not {kind}, using the default", file=sys.stderr, flush=True)


def env_str(name, default=None):
    """The value of `name`, or `default` when it is unset, empty or only whitespace."""
    v = os.environ.get(name)
    return v if v is not None and v.strip() else default


def env_int(name, default):
    v = env_str(name)
    if v is None:
        return default
    try:
        return int(v.strip(), 10)
    except ValueError:
        _warn(name, v, "an integer")
        return default


def env_float(name, default):
    v = env_str(name)
    if v is None:
        return default
    try:
        return float(v.strip())
    except ValueError:
        _warn(name, v, "a number")
        return default
