"""Environment switches for the server (the engines' C++ side: src/env.h, same rules).

An empty or whitespace-only value means "unset, use the default": a Docker Compose pass-through such as
`QE_X: "${QE_X:-}"` exports an empty string. A value that does not parse prints a one-line warning (once per
variable) and uses the default instead of failing or silently turning into 0.
"""
import os
import sys
import threading

_warned = set()

# Environment writes during the start vs the download threads of server/assets.py, dflash_fetch.py and lsa_map.py, which
# may outlive their wait (QE_FETCH_WAIT_S): glibc's setenv can move the environment array while getenv walks it, and a
# thread's connection set-up reads the environment in C (getaddrinfo: RES_OPTIONS / LOCALDOMAIN, OpenSSL: SSL_CERT_*),
# the issue #11 pattern (review FX11-L1). The threads hold the lock while they open a connection (fetch_open), the start
# holds it for each write (set_env) and while the engine library loads (its constructor writes the environment).
# A writer waits at most WRITE_WAIT_S: a connection set-up stalled in DNS reads the environment at its start and then
# only waits on the network, and a stalled link must never hold the start (review of rel136-server, finding 4).
ENV_LOCK = threading.Lock()
WRITE_WAIT_S = 5.0


class env_write:
    """with env_write(): the start's environment writes (ENV_LOCK, at most WRITE_WAIT_S of waiting)."""

    def __enter__(self):
        self.held = ENV_LOCK.acquire(timeout=WRITE_WAIT_S)
        return self

    def __exit__(self, *a):
        if self.held:
            ENV_LOCK.release()


def set_env(name, value, env=None):
    """env[name] = value (default os.environ), under ENV_LOCK (env_write)."""
    with env_write():
        (os.environ if env is None else env)[name] = value


def fetch_open(req, timeout):
    """urllib.request.urlopen(req, timeout) under ENV_LOCK (the connection set-up and its redirects); reading the
    response needs no lock."""
    import urllib.request
    with ENV_LOCK:
        return urllib.request.urlopen(req, timeout=timeout)


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
