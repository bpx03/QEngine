#!/usr/bin/env python3
"""qetop — live terminal monitor of the QEngine server (like htop / nvtop).
Reads http://127.0.0.1:9393/api/stats (server --metrics-port); waits while the server is down.
  qetop [--url http://127.0.0.1:9393] [--interval 0.5] [--lang en|pl]
keys: l = switch language (EN / PL), Ctrl+R = reset "since start" stats (POST /api/reset), q = quit"""
import argparse
import json
import select
import sys
import termios
import tty
import time
import urllib.error
import urllib.request

from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

EN = {"dekoduje": "decoding", "czeka": "waiting", "działa": "up", "start nr": "run #",
      "tok/s teraz": "tok/s now", "aktywnych": "active", "w kolejce": "queued", "tok/krok MTP": "tok/step MTP", "śr.": "avg", "czas": "time", "przel.": "computed", "zakończony": "finished",
      "W na GPU": "W on GPUs", "stan": "state", "kontekst": "context", "z cache": "cached", "wygen.": "generated",
      "nikt teraz nie pracuje": "nobody is working right now", "pula": "pool", "zarezerwowane": "reserved",
      "w użyciu": "in use", "archiwum": "archive", "dysk": "disk", "wzrosty": "grown", "przesunięcia": "moved",
      "wysiedlenia": "evicted", "dla głównego": "for main", "dopuszczone obok": "backfilled",
      "nie rozpoznano głównego agenta w sesji": "main agent not identified in session",
      "pętla między turami": "cross-turn loop", "rozmowa": "conversation", "ta sama tura": "same turn",
      "bez lookup": "no lookup", "razem": "total", "wstrzymane": "paused", "wstrzymania": "preempted", "wznowienia": "resumed", "od startu": "since start", "łącznie": "all time", "czas pracy": "uptime",
      "zapytania": "requests", "wygenerowane tokeny": "generated tokens", "tokeny promptów": "prompt tokens",
      "trafienia cache": "cache hits", "śr. tok/s przy pracy": "avg tok/s busy", "energia": "energy",
      "tokenów / dżul": "tokens / joule", "rekord tok/s (10 s)": "record tok/s (10 s)",
      "rekord agentów naraz": "record agents at once", "najdłuższy kontekst": "longest context",
      "dekodowanie": "decode", "przepustowość (3 min)": "throughput (3 min)", "agenci i kolejka": "agents and queue",
      "karty": "GPUs", "pula KV": "KV pool", "statystyki": "statistics",
      "qengine nie działa — czekam na": "qengine is not running — waiting for",
      "statystyki wyzerowane": "stats reset", "reset nie powiódł się": "reset failed",
      "ten serwer nie obsługuje resetu (potrzebny restart)": "reset not supported by this server (restart needed)"}
LANG = "en"


def _(pl):
    return EN.get(pl, pl) if LANG == "en" else pl


SPARK = "▁▂▃▄▅▆▇█"
SLOT_COLORS = ["blue", "green", "dark_orange", "magenta", "red", "cyan", "yellow", "orchid"]
STATE = {"decode": ("dekoduje", "green"), "prefill": ("prefill", "dark_orange"), "queued": ("czeka", "grey62")}  # _()


def fmt(n, d=0):
    return "–" if n is None else f"{n:,.{d}f}".replace(",", " ")


def big(n):
    return (f"{n / 1e9:.2f} G" if n >= 1e9 else f"{n / 1e6:.2f} M" if n >= 1e6 else
            f"{n / 1e3:.1f} k" if n >= 1e4 else fmt(n))


def dur(s):
    s = int(s)
    d, h, m = s // 86400, s % 86400 // 3600, s % 3600 // 60
    return f"{d} d {h} h" if d else f"{h} h {m} min" if h else f"{m} min {s % 60} s" if m else f"{s} s"


def spark(vals, width):
    vals = vals[-width:]
    top = max(max(vals, default=0), 1)
    return "".join(SPARK[min(7, int(v / top * 7.999))] if v > 0 else " " for v in vals), top


def view(s, width=140, note=None):
    r, se, gl, en, rec = s["rate"], s["session"], s["global"], s.get("engine") or {}, s["records"]
    jobs = s["jobs"]
    act = [j for j in jobs if j["state"] != "queued"]
    queued = [j for j in jobs if j["state"] == "queued"]
    last_pre = next((j.get("prefill_tps") for j in act + s["recent"] if j.get("prefill_tps")), None)
    power = sum(g["power"] for g in s["gpus"])

    head = Text.assemble(("qengine ", "bold"), (f"{s.get('name', '')}  ", "cyan"),
                         (f"{_('działa')} {dur(se['uptime_s'])} · {_('start nr')} {s['runs']}", "grey62"),
                         (f"   {note[0]}", note[1]) if note else ("   l PL/EN · Ctrl+R reset · q", "grey35"))
    al = (s.get("agents") or {}).get("alarm")
    if al:  # the scheduler could not tell the main agent thread (server/agents.py): its priority is off
        head = Group(head, Text(f"⚠ {_('nie rozpoznano głównego agenta w sesji')} {al['session'][:8]}… "
                                f"({al['requests']}) · QE_MAIN_AGENT_RE", style="bold red"))
    lg = (s.get("loop_guard") or {}).get("recent") or []
    if lg and time.time() - lg[-1]["t"] < 600:  # a conversation repeated the same turn (server/loop_guard.py)
        g = lg[-1]
        head = Group(head, Text(f"⚠ {_('pętla między turami')}: {_('rozmowa')} {g['conversation']} "
                                f"({g['turns']}× {_('ta sama tura')}) · presence {g['presence']:.1f}, "
                                f"{_('bez lookup')} · {_('razem')} {s['loop_guard']['triggers']}", style="bold yellow"))
    kpi = Table.grid(expand=True, padding=(0, 2))
    for _col in range(7):
        kpi.add_column(justify="center")
    cell = lambda v, l, st="bold": Text.assemble((v + "\n", st), (l, "grey62"))
    kpi.add_row(cell(fmt(r["tok_s_10"]), "tok/s (10 s)", "bold bright_blue"), cell(fmt(r["tok_s_1"]), _("tok/s teraz")),
                cell(fmt(last_pre), "prefill tok/s", "bold dark_orange"), cell(str(len(act)), _("aktywnych"), "bold green"),
                cell(str(len(queued)), _("w kolejce")), cell(fmt(r["tokens_per_step"], 2), _("tok/krok MTP")),
                cell(fmt(power), _("W na GPU")))

    # throughput sparkline (last ~minutes, 1 s per column)
    hist = s["history"]
    now = int(s["time"])
    by = {b["t"]: b for b in hist}
    dec = [by.get(t, {}).get("dec", 0) for t in range(now - 180, now)]
    pre = [by.get(t, {}).get("pre", 0) for t in range(now - 180, now)]
    sw = max(20, width - 34)
    sd, td = spark(dec, sw)
    sp, tp = spark(pre, sw)
    chart = Group(Text.assemble((f"{_('dekodowanie'):<12}", "bright_blue"), (sd, "bright_blue"), (f"  max {fmt(td)}", "grey62")),
                  Text.assemble((f"{'prefill':<12}", "dark_orange"), (sp, "dark_orange"), (f"  max {fmt(tp)}", "grey62")))

    t = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, padding=(0, 1))
    for c, j in (("agent", "left"), ("stan", "left"), ("#", "right"), ("ctx", "right"),
                 ("cache", "right"), ("przel.", "right"), ("pf/s", "right"), ("gen", "right"), ("tok/s", "right"), ("śr.", "right"),
                 ("czas", "right")):
        t.add_column(_(c), justify=j, no_wrap=True, overflow="ellipsis",
                     max_width=16 if c == "agent" else (14 if c == "stan" else None),
                     min_width=None if c in ("agent", "stan") else len(_(c)))
    order = {"decode": 0, "prefill": 1, "queued": 2}
    ended = s.get("ended", [])
    rows = [(j, False) for j in sorted(jobs, key=lambda j: (order[j["state"]], -j.get("gen", 0)))]
    for j, done in rows + [(j, True) for j in ended]:
        lab, col = (("zakończony", "grey50") if done else STATE[j["state"]])
        extra = f" {j['wait_s']:.1f} s" if j["state"] == "queued" else (
            f" {j['elapsed_s']:.1f} s" if j["state"] == "prefill" else "") if not done else ""
        t.add_row(j["label"], Text(_(lab) + extra, style=col), str(j["slot"]) if j["slot"] >= 0 else "–",
                  big(j["prompt"] + j.get("gen", 0)),
                  f"{100 * j['cached'] / max(j['prompt'], 1):.0f} %" if j["state"] == "decode" else "–",
                  big(j["prefill_tok"]) if "prefill_tok" in j else "–",
                  fmt(j["prefill_tps"]) if j.get("prefill_tps") else "–",
                  fmt(j.get("gen")), fmt(j["tps_now"]) if j.get("tps_now") is not None else "–",
                  fmt(j.get("tps")) if j.get("tps") else "–", "{}:{:02d}".format(*divmod(int(j["age_s"]), 60)) if "age_s" in j else "–",
                  style="grey50" if done else None)
    if not jobs and not ended:
        t.add_row(Text(_("nikt teraz nie pracuje"), style="grey62"), *[""] * 10)

    # KV pool bar
    pool = Text()
    if en.get("pool_tokens"):
        P, W = en["pool_tokens"], max(30, width // 2 - 8)
        cells = [("·", "grey35")] * W
        used = held = 0
        for i, x in enumerate(en["slots"]):
            if not x["cap"]:
                continue
            used += x["cap"]
            held += x["pos"]
            a, b = int(x["off"] / P * W), max(int((x["off"] + x["cap"]) / P * W), int(x["off"] / P * W) + 1)
            fill = a + int(min(x["pos"], x["cap"]) / x["cap"] * (b - a))
            col = SLOT_COLORS[i % len(SLOT_COLORS)]
            for k in range(a, min(b, W)):
                cells[k] = (str(i) if k == a else ("█" if k < fill else "░"), ("bold " if x["active"] else "") + col)
        for ch, st in cells:
            pool.append(ch, style=st)
        a, ev, sc = en["archives"], en["events"], s.get("sched") or {}
        pool = Group(pool, Text(f"{_('pula')} {big(P)} · {_('zarezerwowane')} {big(used)} ({100 * used / P:.0f} %) · {_('w użyciu')} {big(held)}",
                                style="grey62"),
                     Text(f"{_('archiwum')}: RAM {a['ram']} ({a['ram_bytes'] / 2**30:.1f} GB), {_('dysk')} {a['disk']} "
                          f"({a['disk_bytes'] / 2**30:.1f} GB) · {_('wzrosty')} {ev['grow']}, {_('przesunięcia')} {ev['move']}, "
                          f"{_('wysiedlenia')} {ev['evict']} · {_('wstrzymane')} {en.get('paused', 0)} "
                          f"({_('wstrzymania')} {ev.get('pause', 0)}, {_('wznowienia')} {ev.get('resume', 0)}"
                          f"{', ' + _('dla głównego') + ' ' + str(sc['main_pauses']) if sc.get('main_pauses') else ''})"
                          f"{' · ' + _('dopuszczone obok') + ' ' + str(sc['backfills']) if sc.get('backfills') else ''}",
                          style="grey62"))

    g = []
    for x in s["gpus"]:
        frac = x["mem"] / x["mem_total"]
        n = int(frac * 10)
        g.append(Text.assemble((f"GPU{x['i']} ", "bold"), ("█" * n, "bright_blue"), ("·" * (10 - n), "grey35"),
                               f" {x['mem'] / 2**30:4.1f}/{x['mem_total'] / 2**30:.0f} GB {x['util']:3d}%"))
        g.append(Text(f"     {x['power']:3.0f}/{x['limit']:.0f} W  {x['temp']}°C  {x['clock']} MHz", style="grey50"))

    st = Table(box=None, expand=True, pad_edge=False)
    st.add_column("", style="grey62", no_wrap=True)
    st.add_column(_("od startu"), justify="right", no_wrap=True)
    st.add_column(_("łącznie"), justify="right", no_wrap=True)
    hit = lambda x: f"{100 * x['cached_tokens'] / x['prompt_tokens']:.0f} %" if x["prompt_tokens"] else "–"
    avg = lambda x: fmt(x["gen_tokens"] / x["decode_s"]) if x["decode_s"] else "–"
    tpj = lambda x: fmt(x["gen_tokens"] / x["energy_j"], 2) if x["energy_j"] else "–"
    for k, f in (("czas pracy", lambda x: dur(x["uptime_s"])), ("zapytania", lambda x: fmt(x["requests"])),
                 ("wygenerowane tokeny", lambda x: big(x["gen_tokens"])), ("tokeny promptów", lambda x: big(x["prompt_tokens"])),
                 ("trafienia cache", hit), ("śr. tok/s przy pracy", avg),
                 ("energia", lambda x: f"{x['energy_j'] / 3.6e6:.3f} kWh"), ("tokenów / dżul", tpj)):
        st.add_row(_(k), f(se), f(gl))
    srec = s.get("session_records") or dict.fromkeys(rec, 0)
    st.add_row(_("rekord tok/s (10 s)"), fmt(srec["peak_tok_s"]), fmt(rec["peak_tok_s"]))
    st.add_row(_("rekord agentów naraz"), str(srec["max_agents"]), str(rec["max_agents"]))
    st.add_row(_("najdłuższy kontekst"), big(srec["longest_context"]), big(rec["longest_context"]))

    lay = Layout()
    lay.split_column(Layout(Panel(Group(head, kpi), box=box.ROUNDED), size=6 if al else 5),
                     Layout(Panel(chart, title=_("przepustowość (3 min)"), box=box.ROUNDED), size=4),
                     Layout(name="mid"), Layout(name="bot", size=15))
    lay["mid"].split_row(Layout(Panel(t, title=_("agenci i kolejka"), box=box.ROUNDED), ratio=5),
                         Layout(Panel(Group(*g), title=_("karty"), box=box.ROUNDED), size=35))
    lay["bot"].split_row(Layout(Panel(pool, title=_("pula KV"), box=box.ROUNDED), ratio=1),
                         Layout(Panel(st, title=_("statystyki"), box=box.ROUNDED), ratio=1))
    return lay


def reset_stats(url):
    """POST /api/reset (same as the dashboard's "reset session"). Returns (message, style) for the status line."""
    try:
        req = urllib.request.Request(url + "/api/reset", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            r.read()
        return _("statystyki wyzerowane"), "bold green"
    except urllib.error.HTTPError as e:
        if e.code in (404, 405, 501):  # older server without the endpoint
            return _("ten serwer nie obsługuje resetu (potrzebny restart)"), "bold yellow"
        return f"{_('reset nie powiódł się')}: HTTP {e.code}", "bold red"
    except Exception as e:
        return f"{_('reset nie powiódł się')}: {e}", "bold red"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9393")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--lang", choices=("pl", "en"), default=None, help="interface language (default: en)")
    a = ap.parse_args()
    global LANG
    LANG = a.lang or "en"
    waiting = Panel(Text(f"{_('qengine nie działa — czekam na')} {a.url} …", style="grey62"), box=box.ROUNDED)
    # keys: l = language, Ctrl+R = reset session stats, q = quit (stdin in cbreak mode when it is a terminal:
    # ICANON off, so ^R (0x12) is not taken by the line discipline's REPRINT and arrives as a byte)
    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old = termios.tcgetattr(fd) if fd is not None else None
    if fd is not None:
        tty.setcbreak(fd)
    note, note_until = None, 0.0
    try:
        with Live(waiting, screen=True, refresh_per_second=4) as live:
            while True:
                try:
                    with urllib.request.urlopen(a.url + "/api/stats", timeout=2) as r:
                        live.update(view(json.load(r), live.console.width, note if time.time() < note_until else None))
                except Exception:
                    waiting = Panel(Text(f"{_('qengine nie działa — czekam na')} {a.url} …", style="grey62"), box=box.ROUNDED)
                    live.update(waiting)
                end = time.time() + a.interval
                while time.time() < end:
                    if fd is None:
                        time.sleep(end - time.time())
                        break
                    if select.select([sys.stdin], [], [], max(end - time.time(), 0))[0]:
                        k = sys.stdin.read(1).lower()
                        if k == "q":
                            return
                        if k == "\x12":  # Ctrl+R
                            note, note_until = reset_stats(a.url), time.time() + 2.0
                            break
                        if k == "l":
                            LANG = "en" if LANG == "pl" else "pl"
                            break
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
