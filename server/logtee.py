"""Persistent server log: everything the server process writes to its stdout / stderr (Python, the
engine's C / CUDA code, tracebacks) is also appended to a size-capped, rotated file with a timestamp
per line, so the reason of a crash survives the restart (llama-swap keeps only the current
process's output).

install() (server/app.py, before the engine loads) puts a pipe on file descriptors 1 and 2 and
starts this file as a separate process that reads it: the console / llama-swap still gets every
byte, and whatever the server wrote before it died is in the file even when it was killed (abort,
SIGKILL by the OOM killer), because the bytes are in the pipe, not in the dying process. When the
pipe closes the tee appends the server's exit status (when it can still read it) and the kernel's
lines about it (OOM kill, Xid) from the last minutes.

usage (internal): python3 -I server/logtee.py <path> <max MB> <rotations> <server pid>
"""
import ctypes
import faulthandler
import os
import re
import signal
import subprocess
import sys
import time


def default_path(name):
    return os.path.join(os.path.expanduser("~/.cache/qengine/logs"), re.sub(r"[^\w.-]+", "_", name) + ".log")


def install(path, max_mb=64, keep=3, header=""):
    """Tee fd 1 / fd 2 into `path` (rotated at max_mb, `keep` old files). Returns the tee's pid."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    out = os.dup(1)  # the original stdout (terminal / llama-swap pipe) for the tee
    r, w = os.pipe()
    # own session: a SIGINT / SIGTERM to the server's process group does not end the tee before the server's
    # last lines (it ends at the pipe's EOF, i.e. when the server and everything that inherited its fds exit)
    child = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), path, str(max_mb), str(keep),
                              str(os.getpid())], stdin=r, stdout=out, stderr=out, close_fds=True,
                             start_new_session=True)
    os.close(r)
    os.dup2(w, 1)
    os.dup2(w, 2)
    os.close(w)
    os.close(out)
    # the C stdio stdout is fully buffered on a pipe: line-buffered, so an abort() loses at most a partial line
    try:
        libc = ctypes.CDLL(None)
        libc.setvbuf(ctypes.c_void_p.in_dll(libc, "stdout"), None, 1, 0)  # _IOLBF
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    faulthandler.enable(all_threads=True)  # SIGSEGV / SIGABRT / SIGBUS: Python stacks of every thread
    if header:
        print(header, flush=True)
    return child.pid


# ----------------------------------------------------------------------------- the tee process

def _exit_status(pid):
    """The server's exit status while it is a zombie (its parent has not reaped it yet), else None."""
    for _ in range(10):
        try:
            with open(f"/proc/{pid}/stat") as f:
                s = f.read()
        except OSError:
            return None
        rest = s[s.rfind(")") + 2:].split()
        if rest and rest[0] in ("Z", "X") and len(rest) >= 50:
            code = int(rest[49])  # field 52: exit_code (waitpid format)
            if code & 0x7F:
                sig = code & 0x7F
                try:
                    name = signal.Signals(sig).name
                except ValueError:
                    name = str(sig)
                return f"killed by signal {sig} ({name}){' (core dumped)' if code & 0x80 else ''}"
            return f"exit code {code >> 8}"
        time.sleep(0.05)
    return None


def _kernel_lines(pid, since):
    """The kernel's lines about the end of the server: OOM kills, Xid errors, segfaults since `since`."""
    try:
        r = subprocess.run(["journalctl", "-k", "--no-pager", "-o", "short-iso", "--since",
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))],
                           capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    pat = re.compile(rf"Xid|Out of memory|oom-kill|Killed process {pid}\b|\[{pid}\]|pid={pid}\b")
    return [x for x in r.stdout.splitlines() if pat.search(x)][-20:]


def main(path, max_mb, keep, pid):
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)
    cap = max(1, int(float(max_mb) * (1 << 20)))
    keep = max(0, int(keep))
    pid = int(pid)
    t_start = time.time()

    def rotate(f):
        f.close()
        for i in range(keep, 0, -1):
            src = path if i == 1 else f"{path}.{i - 1}"
            if os.path.exists(src):
                try:
                    os.replace(src, f"{path}.{i}")
                except OSError:
                    pass
        if keep == 0:
            try:
                os.unlink(path)
            except OSError:
                pass
        return open(path, "ab", buffering=0)

    f = open(path, "ab", buffering=0)
    console = True
    part = b""

    def ts():
        t = time.time()
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)).encode() + b".%03d " % int(t % 1 * 1000)

    def put(lines):
        nonlocal f
        if not lines:
            return
        stamp = ts()
        try:
            f.write(b"".join(stamp + x + b"\n" for x in lines))
            if f.tell() > cap:
                f = rotate(f)
        except OSError:
            pass

    put([b"==== log of server pid %d ====" % pid])
    while True:
        try:
            d = os.read(0, 1 << 16)
        except InterruptedError:
            continue
        except OSError:
            break
        if not d:
            break
        if console:
            try:
                os.write(1, d)
            except OSError:  # the console / llama-swap went away: the file still gets everything
                console = False
        lines = (part + d).split(b"\n")
        part = lines.pop()
        if len(part) > 1 << 16:  # a very long line without a newline
            lines.append(part)
            part = b""
        put(lines)
    if part:
        put([part])
    st = _exit_status(pid)
    put([f"==== server pid {pid}: output closed, {st or 'exit status not available (already reaped)'} ====".encode()])
    for x in _kernel_lines(pid, max(t_start, time.time() - 600)):
        put([b"[kernel] " + x.encode(errors="replace")])
    f.close()


if __name__ == "__main__":
    main(*sys.argv[1:5])
