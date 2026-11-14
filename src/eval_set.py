"""Select the evaluation set from cohort pairs and write the model prompts (2026-11-11).

Rule (fixed before any model output): dedupe pairs on (repo, function, invocation, before_source, after_source,
before outcome, after outcome); keep at most --per-function pairs per changed function (first in docstring order);
keep every changed pair and an equal number of unchanged pairs drawn with a fixed seed, per repository.
Prompts show both function versions WITHOUT their docstrings (the docstring examples carry expected outputs),
the earlier example lines of the same docstring as setup (inputs only), and the invocation.

Usage: eval_set.py --pairs cohort_v1/pairs.jsonl --out eval_v1 [--per-function 3] [--seed 42]
Writes eval_v1/items.jsonl (gold + prompt) and eval_v1/selection.json.
"""
import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

SYSTEM = ('You are an expert Python engineer. You predict what a single Python invocation does when run against two '
          'versions of the same function. Answer with JSON only.')

TEMPLATE = """Module: {module}
Function: {function}

BEFORE version of the function:
```python
{before}
```

AFTER version of the function:
```python
{after}
```

The invocation runs in the module's namespace after these setup lines (from the function's documentation):
```python
{setup}
```

Invocation:
```python
{invocation}
```

For each version, predict the outcome of the invocation:
- "kind": "returned" or "raised"
- "type": fully qualified type of the returned value (e.g. "builtins.list"), null if raised
- "value": repr() of the returned value (or the printed output if it returns None), null if raised
- "exception": fully qualified exception class (e.g. "builtins.ValueError"), null if returned
Then say which fields differ between the versions.

Answer with exactly this JSON object:
{{"before": {{"kind": ..., "type": ..., "value": ..., "exception": ...}},
 "after": {{"kind": ..., "type": ..., "value": ..., "exception": ...}},
 "changed": {{"kind": true/false, "type": true/false, "value": true/false, "exception": true/false}}}}"""


def prompt(p):
    return TEMPLATE.format(module=p['module'], function=p['function'], before=p['before_source'].rstrip(),
                           after=p['after_source'].rstrip(),
                           setup='\n'.join(p['setup']) if p['setup'] else '# (none)', invocation=p['invocation'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-function', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-chars', type=int, default=24000)
    ap.add_argument('--repos', default='repos')
    a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs, encoding='utf-8')]
    seen, uniq = set(), []
    for p in pairs:
        key = (p['repo'], p['function'], p['invocation'], p['before_source'], p['after_source'],
               json.dumps(p['before'], sort_keys=True), json.dumps(p['after'], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    per_fn = defaultdict(int)
    capped = []
    for p in uniq:
        k = (p['repo'], p['commit'], p['function'])
        if per_fn[k] >= a.per_function:
            continue
        per_fn[k] += 1
        capped.append(p)
    rng = random.Random(a.seed)
    by_repo = defaultdict(lambda: ([], []))
    for p in capped:
        if len(prompt(p)) > a.max_chars:
            continue
        by_repo[p['repo']][0 if p['changed']['any'] else 1].append(p)
    items, sel = [], {}
    for repo, (ch, un) in sorted(by_repo.items()):
        take = rng.sample(un, min(len(un), len(ch)))
        sel[repo] = {'changed': len(ch), 'unchanged_available': len(un), 'unchanged_taken': len(take)}
        items += ch + take
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = {}
    for p in items:
        key = (p['repo'], p['commit'])
        if key not in dates:
            dates[key] = subprocess.run(['git', '-C', str(Path(a.repos) / p['repo']), 'show', '-s', '--format=%cI',
                                         p['commit']], capture_output=True, text=True).stdout.strip()
    with open(out / 'items.jsonl', 'w', encoding='utf-8') as fh:
        for p in items:
            fh.write(json.dumps({'id': p['id'], 'repo': p['repo'], 'source': p.get('source', 'doc'),
                                 'date': dates[(p['repo'], p['commit'])], 'function': p['function'],
                                 'system': SYSTEM, 'prompt': prompt(p),
                                 'gold': {'before': p['before'], 'after': p['after'], 'changed': p['changed']}}) + '\n')
    sel['total'] = len(items)
    sel['changed_total'] = sum(v['changed'] for k, v in sel.items() if isinstance(v, dict))
    sel['rule'] = {'per_function': a.per_function, 'seed': a.seed, 'max_chars': a.max_chars,
                   'dedupe': 'repo, function, invocation, both sources, both outcomes'}
    (out / 'selection.json').write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


if __name__ == '__main__':
    main()
