"""Image input end to end over HTTP against a running server started with --vision 1 (either model, any --think
default: the requests turn thinking off; for the batching check the engine must be in a deterministic mode, e.g.
Flash-Next FLASH_DETERMINISTIC=1 FLASH_MOE_DN_1W=1 FLASH_ROUTE=4). Stops at the first failure.

  python3 tests/test_vision_server.py PORT [screen.png] [second image]
"""
import base64, io, json, os, sys, threading, time, urllib.error, urllib.request

PORT = int(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data", "test_screen.png")
SECOND = sys.argv[3] if len(sys.argv) > 3 else None


def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def second_image():
    if SECOND:
        return b64(SECOND), "user's second image"
    from PIL import Image, ImageDraw  # a simple synthetic picture: a red circle on a blue background
    im = Image.new("RGB", (640, 480), (30, 60, 200))
    ImageDraw.Draw(im).ellipse((220, 140, 420, 340), fill=(230, 20, 20))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), "red circle on blue"


def chat(messages, max_tokens=300, **kw):
    # thinking off per request: the checks read the answer text, so a server started with thinking on (the default)
    # must not spend the token budget on reasoning
    kw.setdefault("chat_template_kwargs", {"enable_thinking": False})
    t0 = time.time()
    st, r = post("/v1/chat/completions", dict(model="x", messages=messages, max_tokens=max_tokens, temperature=0, **kw))
    if st != 200:
        raise SystemExit(f"FAIL: HTTP {st}: {r}")
    return r["choices"][0]["message"]["content"] or "", r["usage"], time.time() - t0


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    if not ok:
        sys.exit(1)


IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(SCREEN)}}
Q = "Describe this screenshot: what kind of log is it, which error appears, and what do the resource bars show?"

# 1. describe the screenshot (OpenAI image_url)
txt, use, dt = chat([{"role": "user", "content": [IMG, {"type": "text", "text": Q}]}], 400)
print(f"-- reply ({use['prompt_tokens']} prompt tokens, {dt:.1f}s):\n{txt}\n")
low = txt.lower()
check("answer text", len(txt.strip()) > 0, f"({use.get('completion_tokens')} completion tokens)")
check("describe screenshot", ("deploy" in low or "pipeline" in low or "build" in low) and "ts" in low
      and all(w in low for w in ("cpu", "ram", "disk")), f"({len(txt)} chars)")

# 2. Anthropic image block, same picture
st, r = post("/v1/messages", {"model": "x", "max_tokens": 200, "temperature": 0, "thinking": {"type": "disabled"},
                              "messages": [{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(SCREEN)}},
    {"type": "text", "text": "In one sentence: what is in this image?"}]}]})
check("anthropic image block", st == 200 and any(b.get("type") == "text" and b["text"].strip() for b in r["content"]),
      repr(r["content"][0].get("text", "")[:160]) if st == 200 else str(r)[:300])

# 3. two images in one message
data2, what2 = second_image()
txt, use, dt = chat([{"role": "user", "content": [
    {"type": "text", "text": "Image 1:"}, IMG, {"type": "text", "text": "Image 2:"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data2}},
    {"type": "text", "text": "Describe each of the two images in one sentence each."}]}], 250)
print(f"-- two images ({use['prompt_tokens']} prompt tokens, {dt:.1f}s):\n{txt}\n")
low = txt.lower()
ok2 = ("red" in low and ("circle" in low or "dot" in low or "ball" in low)) if not SECOND else True
check("two images", ok2 and ("log" in low or "build" in low or "terminal" in low or "pipeline" in low))

# 4. multi-turn: the next turn reuses the image prefix from the prompt cache
m1 = [{"role": "user", "content": [IMG, {"type": "text", "text": "What is the build number shown?"}]}]
a1, u1, _ = chat(m1, 60)
m2 = m1 + [{"role": "assistant", "content": a1},
           {"role": "user", "content": "And which percentage does the DISK bar show?"}]
a2, u2, _ = chat(m2, 60)
cached = u2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
print(f"-- turn 1: {a1!r}\n-- turn 2: {a2!r}")
check("multi-turn prompt cache", cached >= u1["prompt_tokens"] - 8, f"(turn 2: {cached} of {u2['prompt_tokens']} cached)")

# 5. text-only request alone vs batched next to an image prefill / decode: the same tokens (deterministic engine)
TQ = [{"role": "user", "content": "Write a short paragraph about the history of the lighthouse at Alexandria."}]
alone, _, _ = chat(TQ, 160)
res = {}


def img_req():
    res["img"] = chat([{"role": "user", "content": [IMG, {"type": "text", "text": Q + " Be thorough."}]}], 300)


th = threading.Thread(target=img_req)
th.start()
time.sleep(0.3)
batched, _, _ = chat(TQ, 160)
th.join()
check("text batched next to an image request == alone", batched == alone,
      "" if batched == alone else f"\nalone:   {alone!r}\nbatched: {batched!r}")

# 6. unsupported inputs
st, r = post("/v1/chat/completions", {"model": "x", "max_tokens": 10, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,bm90IGFuIGltYWdl"}}, {"type": "text", "text": "hi"}]}]})
check("broken image -> 400", st == 400, str(r)[:200])
print("ALL PASS")
