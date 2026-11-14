"""Query a local OpenAI-compatible server (llama.cpp llama-server) for every evaluation item (2026-11-11).

Greedy decoding (temperature 0, seed 42), thinking off (server started with --reasoning-budget 0), max 1536 tokens.
Resumable: items already in the output file are skipped. Writes rows {"id", "model", "raw", "usage", "finish"}.
Usage: run_local.py --items eval_v1/items.jsonl --out answers/gemma4-12b-q4.jsonl --model gemma4-12b-qat-q4
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path


def ask(url, model, system, prompt, max_tokens):
    body = json.dumps({'model': model, 'messages': [{'role': 'system', 'content': system},
                                                     {'role': 'user', 'content': prompt}],
                       'temperature': 0, 'seed': 42, 'max_tokens': max_tokens}).encode()
    req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--url', default='http://127.0.0.1:8087/v1/chat/completions')
    ap.add_argument('--max-tokens', type=int, default=1536)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['id'] for l in open(out, encoding='utf-8')} if out.exists() else set()
    items = [json.loads(l) for l in open(a.items, encoding='utf-8')]
    t0 = time.time()
    todo = [it for it in items if it['id'] not in done]
    lock = threading.Lock()
    count = [0]
    with open(out, 'a', encoding='utf-8') as fh:
        def work(it):
            r = {'error': 'not run'}
            for attempt in range(3):
                try:
                    r = ask(a.url, a.model, it['system'], it['prompt'], a.max_tokens)
                    break
                except Exception as e:  # noqa: BLE001
                    r = {'error': str(e)}
                    time.sleep(5)
            ch = (r.get('choices') or [{}])[0]
            row = json.dumps({'id': it['id'], 'model': a.model, 'raw': (ch.get('message') or {}).get('content'),
                              'finish': ch.get('finish_reason'), 'usage': r.get('usage'), 'error': r.get('error')})
            with lock:
                fh.write(row + '\n')
                fh.flush()
                count[0] += 1
                if count[0] % 25 == 0:
                    print(f'{count[0]}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(work, todo))


if __name__ == '__main__':
    main()
