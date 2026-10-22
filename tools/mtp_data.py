"""Training data for the MTP head (self-distillation): the engine answers varied prompts (code
tasks on Python source files from the installed site-packages, general questions in English and
Polish, thinking on and off), then every (prompt + answer) is prefilled again with the final
normed hidden states dumped.

  python tools/mtp_data.py gen  <out_dir> <n_samples> [seed]   -> out_dir/seqs.jsonl
  python tools/mtp_data.py dump <out_dir>                      -> out_dir/tokens.npy, hidden.bin, index.json
"""
import ctypes, glob, json, os, random, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-INT4")

GENERAL_EN = """Explain how a B-tree differs from a binary search tree and when each is preferable.
What are the trade-offs between optimistic and pessimistic locking in databases?
Describe how TCP congestion control works, including slow start and congestion avoidance.
Write a Python function that merges overlapping intervals and explain its complexity.
How does garbage collection work in Go compared to Java?
Explain the difference between processes and threads, with examples of when to use each.
What is the CAP theorem? Give concrete examples of systems choosing different trade-offs.
Write a bash script that finds the 10 largest files in a directory tree and prints their sizes.
How would you design a rate limiter for a public REST API?
Explain eventual consistency to a junior developer.
Write a SQL query that returns the second highest salary per department, and explain it.
What are Rust lifetimes and why does the borrow checker need them?
Explain how HTTPS establishes a secure connection, step by step.
Compare REST, GraphQL and gRPC for a mobile backend.
Implement an LRU cache in TypeScript with O(1) operations.
How do you debug a memory leak in a long-running Node.js service?
Explain Big-O notation with examples of O(1), O(log n), O(n log n) and O(n^2) algorithms.
Write unit tests in pytest for a function that parses ISO 8601 dates.
What is the difference between a mutex and a semaphore?
Explain how Docker containers differ from virtual machines.
Describe a good branching strategy for a team of eight developers using git.
How does a hash map handle collisions? Compare chaining and open addressing.
Write a C++ class implementing a thread-safe bounded queue.
Explain the attention mechanism in transformers in simple terms.
What happens when you type a URL into a browser and press enter?
How would you migrate a monolith to microservices without downtime?
Explain dependency injection and show an example in Python.
Write a Dockerfile for a FastAPI application with a multi-stage build.
What are the SOLID principles? Give a short example for each.
Explain how Kubernetes schedules pods onto nodes.
Write a regular expression that validates IPv4 addresses and explain each part.
How do database indexes speed up queries, and when can they slow things down?
Explain the event loop in JavaScript with an example involving promises and setTimeout.
Implement binary search in Python and list the common off-by-one mistakes.
What is a race condition? Show one in code and how to fix it.
Explain how a CPU cache works and why data locality matters for performance.
Write a GitHub Actions workflow that runs tests on Python 3.10, 3.11 and 3.12.
Compare PostgreSQL and MongoDB for an e-commerce application.
Explain what a closure is in Python and show a practical use.
How would you structure logging and monitoring for a production web service?
Write a function that detects a cycle in a linked list and explain why it works.
Explain the difference between unit, integration and end-to-end tests.
What is backpressure in streaming systems and how is it handled?
Explain how public-key cryptography works without heavy math.
Write a Makefile for a small C project with separate debug and release builds.
How do you choose between SQL transactions isolation levels?
Explain memoization and dynamic programming using the knapsack problem.
What are the main causes of slow web pages and how do you measure them?
Write a Python async crawler that limits concurrency to 10 requests.
Explain how floating point numbers are represented and why 0.1 + 0.2 != 0.3.
Summarize the history of the Unix operating system in a few paragraphs.
Write a short story about a robot learning to paint.
Plan a three-day trip to Kyoto for someone interested in history and food.
Explain the causes of inflation and how central banks respond to it.
Write a professional email asking a client to approve a delayed project deadline.
What are the health benefits and risks of intermittent fasting?
Explain photosynthesis to a ten-year-old.
Give me a weekly workout plan for a beginner with no equipment.
Compare electric cars and hybrids for someone who drives 60 km per day.
Write a product description for a mechanical keyboard aimed at programmers.""".strip().split("\n")

GENERAL_PL = """Wyjaśnij, czym różni się proces od wątku i kiedy używać każdego z nich.
Napisz funkcję w Pythonie, która sprawdza, czy napis jest palindromem, i omów jej złożoność.
Jak działa indeks w bazie danych PostgreSQL i kiedy warto go dodać?
Wyjaśnij zasadę działania protokołu HTTPS krok po kroku.
Porównaj języki Rust i Go pod kątem pisania serwerów sieciowych.
Napisz skrypt w bashu, który robi kopię zapasową katalogu i usuwa kopie starsze niż 7 dni.
Czym jest wstrzykiwanie zależności? Pokaż przykład w TypeScript.
Jak zaprojektować system kolejkowania zadań dla aplikacji webowej?
Wyjaśnij, czym są transakcje w bazach danych i co oznacza ACID.
Napisz testy jednostkowe w pytest dla funkcji liczącej silnię.
Jak działa garbage collector w Pythonie?
Opisz dobre praktyki pracy z gitem w małym zespole.
Wyjaśnij mechanizm attention w modelach językowych prostym językiem.
Napisz klasę w C++ implementującą stos z obsługą wyjątków.
Jakie są zalety i wady architektury mikroserwisów?
Wyjaśnij różnicę między REST a GraphQL na przykładzie sklepu internetowego.
Napisz zapytanie SQL, które zwraca trzech klientów z największą sumą zamówień.
Jak debugować wolne zapytania w MySQL?
Czym jest Docker i jak napisać prosty Dockerfile dla aplikacji Flask?
Wyjaśnij notację dużego O na przykładach.
Napisz funkcję w JavaScript, która usuwa duplikaty z tablicy obiektów po polu id.
Jak działa pamięć podręczna procesora i dlaczego ma znaczenie dla wydajności?
Opisz, jak zabezpieczyć API przed atakami typu brute force.
Napisz krótkie opowiadanie o programistce, która znalazła błąd w kodzie z lat 80.
Zaplanuj trzydniowy wyjazd do Krakowa dla rodziny z dziećmi.
Wyjaśnij, jak działa inflacja i jak banki centralne na nią reagują.
Napisz uprzejmy e-mail do klienta z prośbą o przesunięcie terminu projektu.
Jakie są zalety regularnej aktywności fizycznej dla osób pracujących przy komputerze?
Wyjaśnij fotosyntezę dziesięcioletniemu dziecku.
Porównaj samochody elektryczne i hybrydowe dla osoby dojeżdżającej 50 km dziennie.
Napisz opis produktu dla ergonomicznego krzesła biurowego.
Jak przygotować się do rozmowy rekrutacyjnej na stanowisko backend developera?
Wyjaśnij, czym jest rekurencja, i pokaż przykład z drzewem katalogów.
Napisz program w Pythonie, który zlicza słowa w pliku tekstowym i wypisuje 10 najczęstszych.
Jakie są różnice między listą a krotką w Pythonie?
Opisz historię Polski w XX wieku w kilku akapitach.
Wyjaśnij, jak działa szyfrowanie asymetryczne.
Jak zaprojektować schemat bazy danych dla systemu rezerwacji biletów?
Napisz konfigurację nginx jako reverse proxy dla dwóch aplikacji.
Wyjaśnij różnicę między testami jednostkowymi, integracyjnymi i end-to-end.""".strip().split("\n")

CODE_TASKS = [
    "Explain what this code does, section by section.",
    "Review this code: list bugs, risky edge cases and style problems, with fixes.",
    "Write pytest unit tests covering the main functions of this code.",
    "Refactor this code to be more readable and explain each change.",
    "Add type hints and docstrings to this code. Show the full updated code.",
    "Rewrite the core logic of this code in Rust.",
    "Rewrite the core logic of this code in TypeScript.",
    "Find performance problems in this code and propose optimizations.",
    "Summarize the public API of this module in a markdown table.",
    "Suggest how to split this module into smaller modules and why.",
    "Write a short tutorial showing how to use this code.",
    "Identify any security concerns in this code.",
    "Wyjaśnij, co robi ten kod, krok po kroku.",
    "Zrób przegląd tego kodu: znajdź błędy i zaproponuj poprawki.",
    "Napisz testy jednostkowe do tego kodu.",
    "Zrefaktoryzuj ten kod i opisz zmiany.",
    "Dodaj obsługę błędów do tego kodu i pokaż poprawioną wersję.",
    "Return a JSON object listing every function in this code with fields name, args, returns, purpose.",
]

SYSTEMS = [
    None,
    "You are a helpful assistant.",
    "You are an expert software engineer. Answer precisely and show code when useful.",
    "You are an interactive CLI coding assistant. Be concise; use markdown code blocks.",
    "Jesteś pomocnym asystentem. Odpowiadaj po polsku.",
]


def source_files():
    import site
    roots = site.getsitepackages()
    pk = ["transformers/models/llama", "transformers/generation", "torch/nn/modules", "torch/optim",
          "requests", "jinja2", "numpy/lib", "aiohttp", "click", "rich", "httpx", "tokenizers",
          "pydantic", "fastapi", "starlette", "uvicorn", "yaml"]
    files = []
    for r in roots:
        for p in pk:
            files += glob.glob(os.path.join(r, p, "**", "*.py"), recursive=True)
    return [f for f in files if 2000 < os.path.getsize(f) < 400000]


def make_prompts(n, seed):
    rnd = random.Random(seed)
    files = source_files()
    out = []
    for i in range(n):
        r = rnd.random()
        if r < 0.5:
            f = rnd.choice(files)
            lines = open(f, errors="ignore").read().split("\n")
            a = rnd.randrange(0, max(1, len(lines) - 40))
            snip = "\n".join(lines[a:a + rnd.randint(40, 260)])
            user = f"{rnd.choice(CODE_TASKS)}\n\n```python\n{snip}\n```"
        elif r < 0.78:
            user = rnd.choice(GENERAL_EN)
        else:
            user = rnd.choice(GENERAL_PL)
        msgs = []
        sysm = rnd.choice(SYSTEMS)
        if sysm:
            msgs.append({"role": "system", "content": sysm})
        msgs.append({"role": "user", "content": user})
        out.append({"messages": msgs, "think": rnd.random() < 0.5})
    return out


def gen(out_dir, n, seed):
    from tokenizers import Tokenizer
    from app import Template
    from engine import Engine
    os.makedirs(out_dir, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    tp = os.path.join(ROOT, "server", "templates", "qwen38_syssafe.jinja")
    tmpl = Template(tp if os.path.exists(tp) else os.path.join(MODEL, "chat_template.jinja"))
    eos = {tok.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")}
    prompts = make_prompts(n, seed)
    e = Engine(MODEL, max_t=32768)
    e.set_sampling(0.7, 0.95, 40, 0.0, seed)
    e.set_concurrent(True)
    out = open(os.path.join(out_dir, "seqs.jsonl"), "a")
    queue = list(range(n))
    active = {}  # slot -> (idx, ids, gen, max_new)
    t0 = time.time()
    total = 0
    while queue or active:
        while queue and len(active) < 3:
            i = queue.pop(0)
            p = prompts[i]
            ids = tok.encode(tmpl.render(p["messages"], None, enable_thinking=p["think"]),
                             add_special_tokens=False).ids
            if len(ids) > 12000:
                continue
            first, _, slot = e.start(ids, 7, affinity=i + 1)
            active[slot] = (i, ids, [first], 1536 if p["think"] else 1024)
        res = e.step_batch(list(active))
        for slot, toks in zip(list(active), res):
            i, ids, g, mx = active[slot]
            g += toks
            stop = next((j for j, t in enumerate(g) if t in eos), None)
            if stop is not None or len(g) >= mx:
                g = g[:stop + 1] if stop is not None else g[:mx]
                out.write(json.dumps({"i": i, "think": prompts[i]["think"], "prompt": ids, "answer": g}) + "\n")
                out.flush()
                total += len(g)
                e.release(slot)
                del active[slot]
        if total and int(time.time() - t0) % 60 == 0:
            pass
    dt = time.time() - t0
    print(f"generated {total} tokens in {dt:.0f}s ({total / dt:.0f} tok/s)")
    e.close()


def dump(out_dir):
    from engine import Engine
    seqs = [json.loads(l) for l in open(os.path.join(out_dir, "seqs.jsonl"))]
    H = 5120
    e = Engine(MODEL, max_t=32768)
    e.lib.qe_set_dump_hidden.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    maxn = max(len(s["prompt"]) + len(s["answer"]) for s in seqs)
    buf = np.zeros((maxn, H), dtype=np.uint16)
    hid = open(os.path.join(out_dir, "hidden.bin"), "wb")
    toks, index = [], []
    off = 0
    t0 = time.time()
    for k, s in enumerate(seqs):
        ids = s["prompt"] + s["answer"]
        e.reset()
        e.lib.qe_set_dump_hidden(e.h, buf.ctypes.data)
        e.start(ids, 1)
        e.lib.qe_set_dump_hidden(e.h, None)
        hid.write(buf[:len(ids)].tobytes())
        toks += ids
        index.append({"off": off, "n": len(ids), "prompt": len(s["prompt"]), "think": s["think"]})
        off += len(ids)
        if k % 100 == 0:
            print(f"{k}/{len(seqs)} sequences, {off} tokens, {time.time() - t0:.0f}s", flush=True)
    hid.close()
    np.save(os.path.join(out_dir, "tokens.npy"), np.array(toks, dtype=np.int32))
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"))
    print(f"dumped {off} tokens in {time.time() - t0:.0f}s")
    e.close()


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    else:
        dump(sys.argv[2])
