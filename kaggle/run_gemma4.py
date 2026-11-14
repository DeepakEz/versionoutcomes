# Gemma 4 family on the version-outcome items (2026-11-11). Kaggle GPU (T4 x2) script kernel.
# Builds llama.cpp at the same pinned commit used locally (b11552, 23b0202a), serves each official QAT-Q4 GGUF from
# Kaggle Models with thinking off, greedy decoding (temperature 0, seed 42), and writes raw answers per model.
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
BUDGET = float(os.environ.get('BUDGET_SEC', 11 * 3600))
OUT = '/kaggle/working/answers'
os.makedirs(OUT, exist_ok=True)
LOG = open('/kaggle/working/run.log', 'a')


def log(*a):
    msg = f'[{time.time() - T0:7.0f}s] ' + ' '.join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + '\n')
    LOG.flush()


def sh(cmd, **kw):
    log('$', cmd)
    subprocess.run(cmd, shell=True, check=True, **kw)


items_path = glob.glob('/kaggle/input/**/items.jsonl', recursive=True)[0]
ITEMS = [json.loads(l) for l in open(items_path)]
log('items', len(ITEMS), items_path)

if not os.path.exists('/tmp/llama.cpp/build/bin/llama-server'):
    sh('cd /tmp && git clone -q https://github.com/ggml-org/llama.cpp.git && cd llama.cpp && '
       'git checkout -q 23b0202a189c44a54625aadcb37a946dd1d6278d')
    found = subprocess.run("find / -name 'libcuda.so*' -not -path '*/proc/*' 2>/dev/null | head -20", shell=True,
                           capture_output=True, text=True).stdout.split()
    log('libcuda candidates', found)
    os.makedirs('/tmp/cudalib', exist_ok=True)
    src = sorted(found, key=lambda f: ('stubs' in f, len(f)))[0]
    if not os.path.exists('/tmp/cudalib/libcuda.so'):
        os.symlink(src, '/tmp/cudalib/libcuda.so')
    stub = '/tmp/cudalib/libcuda.so'
    sh('cd /tmp/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 '
       f'-DCUDA_cuda_driver_LIBRARY={stub} -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /kaggle/working/cmake.log 2>&1 && '
       'cmake --build build --target llama-server -j 4 > /kaggle/working/build.log 2>&1')
SERVER = '/tmp/llama.cpp/build/bin/llama-server'

MODELS = [  # (name, gguf slug, parallel slots, item limit)
    ('gemma4-e2b-qat-q4', 'gemma-4-e2b-it-qat-q4_0-gguf', 4, None),
    ('gemma4-e4b-qat-q4', 'gemma-4-e4b-it-qat-q4_0-gguf', 4, None),
    ('gemma4-26b-a4b-qat-q4', 'gemma-4-26b-a4b-it-qat-q4_0-gguf', 4, None),
    ('gemma4-31b-qat-q4', 'gemma-4-31b-it-qat-q4_0-gguf', 2, None),
]


def ask(port, it):
    body = json.dumps({'model': 'm', 'messages': [{'role': 'system', 'content': it['system']},
                                                   {'role': 'user', 'content': it['prompt']}],
                       'temperature': 0, 'seed': 42, 'max_tokens': 1536}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body,
                                 {'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(5)
    return {'error': err}


for name, slug, slots, limit in MODELS:
    if time.time() - T0 > BUDGET:
        log('budget exhausted before', name)
        break
    files = sorted(glob.glob(f'/kaggle/input/**/{slug}/**/*.gguf', recursive=True))
    files = [f for f in files if 'mmproj' not in os.path.basename(f).lower()]
    if not files:
        log('missing gguf for', slug)
        continue
    out_path = f'{OUT}/{name}.jsonl'
    done = {json.loads(l)['id'] for l in open(out_path)} if os.path.exists(out_path) else set()
    todo = [it for it in ITEMS if it['id'] not in done][:limit]
    log(name, files[0], 'todo', len(todo))
    port = 8090
    ctx = 10240 * slots
    proc = subprocess.Popen([SERVER, '-m', files[0], '--host', '127.0.0.1', '--port', str(port), '-ngl', '99',
                             '-c', str(ctx), '-np', str(slots), '--reasoning-budget', '0', '--no-cache-prompt',
                             '--alias', name], stdout=open(f'/kaggle/working/{name}.server.log', 'w'),
                            stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    lock = threading.Lock()
    fh = open(out_path, 'a')
    n = [0]

    def work(it):
        if time.time() - T0 > BUDGET:
            return
        r = ask(port, it)
        ch = (r.get('choices') or [{}])[0]
        row = {'id': it['id'], 'model': name, 'raw': (ch.get('message') or {}).get('content'),
               'finish': ch.get('finish_reason'), 'usage': r.get('usage'), 'error': r.get('error')}
        with lock:
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            n[0] += 1
            if n[0] % 25 == 0:
                log(name, n[0], '/', len(todo))

    with ThreadPoolExecutor(slots) as pool:
        list(pool.map(work, todo))
    fh.close()
    proc.terminate()
    proc.wait(timeout=60)
    log(name, 'finished', n[0])
log('all done')
