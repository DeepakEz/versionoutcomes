"""Score model answers on the evaluation set (2026-11-11).

answers.jsonl rows: {"id", "model", "raw"} (raw model text). The first JSON object in the text is parsed.
Field rules (fixed before scoring): kind exact; type and exception match fully qualified or bare class name;
value exact after stripping surrounding whitespace, or equal after ast.literal_eval of both strings (numbers within
1e-9 relative). A phase is complete when all four fields match. Change detection uses the predicted change flags
(any field) against the observed change.

Usage: score.py --items eval_v1/items.jsonl --answers ANSWERS.jsonl [--answers ...] --out scores.json
"""
import argparse
import ast
import json
import math
import re
from collections import defaultdict


def first_json(text):
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M)
    start = text.find('{')
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    return None


def same_name(pred, gold):
    if gold is None:
        return pred in (None, '', 'null', 'None')
    if not isinstance(pred, str):
        return False
    return pred.strip() == gold or pred.strip() == gold.split('.')[-1] or \
        pred.strip().split('.')[-1] == gold.split('.')[-1] and gold.startswith('builtins.')


_BAD = object()


def lit(s):
    try:
        return ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        return _BAD


def close(x, y):
    if isinstance(x, float) or isinstance(y, float):
        try:
            return math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if type(x) is not type(y):
        return False
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(close(a, b) for a, b in zip(x, y))
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(close(x[k], y[k]) for k in x)
    return x == y


def same_value(pred, gold):
    if gold is None:
        return pred in (None, 'null', 'None', '')
    if not isinstance(pred, str):
        pred = json.dumps(pred) if pred is not None else None
        if pred is None:
            return False
    if pred.strip() == gold.strip():
        return True
    a, b = lit(pred), lit(gold)
    if a is _BAD or b is _BAD:
        return False
    return close(a, b)


def phase_scores(pred, gold):
    pred = pred if isinstance(pred, dict) else {}
    s = {'kind': pred.get('kind') == gold['kind'],
         'type': same_name(pred.get('type'), gold['type']),
         'value': same_value(pred.get('value'), gold['value']),
         'exception': same_name(pred.get('exception'), gold['exception'])}
    s['complete'] = all(s.values())
    g = lit(gold['value']) if isinstance(gold.get('value'), str) else _BAD
    lenient = s['value'] or (isinstance(g, str) and isinstance(pred.get('value'), str) and pred['value'] == g)
    s['complete_lenient'] = s['kind'] and s['type'] and s['exception'] and lenient
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True)
    ap.add_argument('--answers', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split-out', default='', help='also write pre/post training-cutoff split here')
    ap.add_argument('--cutoff', default='2025-01-01', help='commit date (ISO) at or after which an item is "post"')
    a = ap.parse_args()
    items = {json.loads(l)['id']: json.loads(l) for l in open(a.items, encoding='utf-8')}
    report = {}
    split = defaultdict(lambda: {'pre': defaultdict(float), 'post': defaultdict(float)})
    for path in a.answers:
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        model = rows[0]['model'] if rows else path
        agg = defaultdict(float)
        per_repo = defaultdict(lambda: defaultdict(float))
        n = 0
        for r in rows:
            it = items.get(r['id'])
            if it is None:
                continue
            n += 1
            g = it['gold']
            ans = first_json(r.get('raw') or '')
            agg['parsed'] += ans is not None
            ans = ans or {}
            # An answer with no parseable JSON or no `changed` object gives no verdict and scores zero on
            # every measure: no verdict credit and no field credit (a missing field is never a correct null).
            # Corrected 2026-11-12.
            ch = ans.get('changed') if isinstance(ans.get('changed'), dict) else None
            if ch is None:
                zero = {k: False for k in ('kind', 'type', 'value', 'exception', 'complete', 'complete_lenient')}
                sb, sa = dict(zero), dict(zero)
            else:
                sb, sa = phase_scores(ans.get('before'), g['before']), phase_scores(ans.get('after'), g['after'])
            for k in sb:
                agg['before_' + k] += sb[k]
                agg['after_' + k] += sa[k]
            gold_any = g['changed']['any']
            if gold_any:
                agg['changed_after_complete'] += sa['complete']
                agg['changed_after_value'] += sa['value']
            if ch is None:
                pred_any = None
                agg['no_verdict'] += 1
                agg['fn'] += gold_any
            else:
                pred_any = any(bool(ch.get(f)) for f in ('kind', 'type', 'value', 'exception'))
                agg['tp'] += pred_any and gold_any
                agg['fp'] += pred_any and not gold_any
                agg['fn'] += (not pred_any) and gold_any
                agg['tn'] += (not pred_any) and not gold_any
            verdict_ok = pred_any is not None and pred_any == gold_any
            pair_ok = sb['complete'] and sa['complete'] and verdict_ok
            agg['change_correct'] += verdict_ok
            agg['paired_complete'] += pair_ok
            pr = per_repo[it['repo']]
            pr['n'] += 1
            pr['change_correct'] += verdict_ok
            pr['paired_complete'] += pair_ok
            post = it.get('date', '') >= a.cutoff
            sp = split[model]['post' if post else 'pre']
            sp['n'] += 1
            sp['change_correct'] += verdict_ok
            sp['paired_complete'] += pair_ok
        out = {k: v for k, v in agg.items()}
        out['n'] = n
        out['changed_n'] = out.get('tp', 0) + out.get('fn', 0)
        out['change_recall'] = out.get('tp', 0) / max(1, out['changed_n'])
        out['change_precision'] = out.get('tp', 0) / max(1, out.get('tp', 0) + out.get('fp', 0))
        out['per_repo'] = {k: dict(v) for k, v in per_repo.items()}
        report[model] = out
        print(model, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()
                                 if k != 'per_repo'}))
    json.dump(report, open(a.out, 'w'), indent=1)
    if a.split_out:
        out = {}
        for m, s in split.items():
            out[m] = {k: {'n': int(v['n']), 'change_correct': int(v['change_correct']),
                          'paired_complete': int(v['paired_complete']),
                          'change_acc': v['change_correct'] / max(1, v['n']),
                          'paired_complete_rate': v['paired_complete'] / max(1, v['n'])} for k, v in s.items()}
        json.dump(out, open(a.split_out, 'w'), indent=1)


if __name__ == '__main__':
    main()
