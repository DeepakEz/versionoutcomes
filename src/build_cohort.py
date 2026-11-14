"""Build the held-out docstring-example cohort (2026-11-11).

Rule-based, no hand selection: for every non-merge commit since --since in each library, every Python function or
method whose source changed between the parent and the commit and whose docstring (after version, else before) has
doctest examples becomes a candidate. Its examples run in order in both snapshots (fresh processes, worker.py,
PYTHONHASHSEED=0, 20 s limit). Every expression example whose source names the changed function is one input pair.

Outputs (in --out): candidates.jsonl, pairs.jsonl (both outcomes + flags), and build.json (counts).
Usage: build_cohort.py --repos DIR --out DIR [--since 2021-01-01] [--max-per-repo 400] [--procs 24]
"""
import argparse
import ast
import doctest
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import io
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP_PARTS = {'tests', 'test', 'docs', 'doc', 'benchmarks', 'examples', 'scripts', 'tasks', 'tools', '.github'}


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, check=True).stdout


def functions(src):
    """qualname -> (source without docstring, docstring) for top-level functions and class methods."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    lines = src.splitlines(keepends=True)
    out = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                doc = ast.get_docstring(child, clean=False) or ''
                start = (child.decorator_list[0].lineno if child.decorator_list else child.lineno) - 1
                body = ''.join(lines[start:child.end_lineno])
                if doc and child.body and isinstance(child.body[0], ast.Expr):
                    d = child.body[0]
                    seg = ''.join(lines[d.lineno - 1:d.end_lineno])
                    body = body.replace(seg, '', 1)
                out[q] = (body, doc)
            elif isinstance(child, ast.ClassDef) and not prefix:
                visit(child, child.name + '.')
    visit(tree, '')
    return out


def module_of(path):
    parts = Path(path).parts
    root = 'src' if parts[0] == 'src' else ''
    rel = parts[1:] if root else parts
    mod = '.'.join(rel)[:-3]
    if mod.endswith('.__init__'):
        mod = mod[:-9]
    return root, mod


def test_calls(repo, p, c, short):
    """Self-contained call expressions of `short` on lines ADDED to test files by this commit (regression tests)."""
    try:
        diff = git(repo, 'diff', '-U0', p, c, '--', '*test*.py').decode('utf-8', 'replace')
    except subprocess.CalledProcessError:
        return []
    out = []
    for line in diff.splitlines():
        if not line.startswith('+') or line.startswith('+++') or short + '(' not in line:
            continue
        s = line[1:].strip()
        for prefix in ('assert ', 'return '):
            if s.startswith(prefix):
                s = s[len(prefix):]
        s = s.rstrip(':').rstrip(',')
        try:
            tree = ast.parse(s, mode='exec')
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if name == short:
                    seg = ast.get_source_segment(s, node)
                    if seg and len(seg) <= 300 and seg not in out:
                        out.append(seg)
                    break
    return out[:6]


def candidates(repo, since, cap):
    revs = git(repo, 'log', '--no-merges', f'--since={since}', '--format=%H %P', '--', '*.py').decode().split('\n')
    out = []
    for line in revs:
        if not line.strip() or len(line.split()) != 2:
            continue
        c, p = line.split()
        files = git(repo, 'diff', '--name-only', p, c, '--', '*.py').decode().split()
        for f in files:
            parts = set(Path(f).parts[:-1])
            name = Path(f).name
            if parts & SKIP_PARTS or name.startswith('test') or name in ('setup.py', 'conftest.py', 'noxfile.py'):
                continue
            try:
                a = git(repo, 'show', f'{c}:{f}').decode('utf-8', 'replace')
                b = git(repo, 'show', f'{p}:{f}').decode('utf-8', 'replace')
            except subprocess.CalledProcessError:
                continue
            fa, fb = functions(a), functions(b)
            for q in sorted(set(fa) & set(fb)):
                if fa[q][0] == fb[q][0]:
                    continue                         # body (without docstring) unchanged
                doc = fa[q][1] or fb[q][1]
                try:
                    ex = doctest.DocTestParser().get_examples(doc) if doc else []
                except ValueError:
                    ex = []
                tests = test_calls(repo, p, c, q.split('.')[-1])
                if not ex and not tests:
                    continue
                root, mod = module_of(f)
                out.append({'repo': repo.name, 'commit': c, 'parent': p, 'file': f, 'root': root, 'module': mod,
                            'function': q, 'examples': [e.source.rstrip('\n') for e in ex] + tests,
                            'sources': ['doc'] * len(ex) + ['test'] * len(tests),
                            'wants': [e.want.rstrip('\n') for e in ex] + [''] * len(tests),
                            'before_source': fb[q][0], 'after_source': fa[q][0]})
                if len(out) >= cap:
                    return out
    return out


def snapshot(repo, rev, cache):
    d = cache / repo.name / rev
    if not d.exists():
        tmp = Path(tempfile.mkdtemp(dir=cache))
        data = git(repo, 'archive', rev)
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(tmp, filter='data')
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.rename(d)
        except OSError:
            pass
    return d


def run_one(args):
    cand, rev, snap_dir = args
    snap = Path(snap_dir).resolve()               # the worker runs in another cwd: the root must be absolute
    root = str(snap / cand['root']) if cand['root'] else str(snap)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as fh:
        json.dump(cand['examples'], fh)
        ex_path = fh.name
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONHASHSEED'] = '0'
    try:
        r = subprocess.run([sys.executable, '-E', '-P', str(HERE / 'worker.py'), root, cand['module'], ex_path],
                           capture_output=True, timeout=20, env=env, cwd=tempfile.gettempdir())
        res = json.loads(r.stdout.decode('utf-8', 'replace').strip().splitlines()[-1]) if r.stdout.strip() \
            else {'setup': f'worker failed rc={r.returncode}: {r.stderr.decode("utf-8", "replace")[-200:]}'}
    except subprocess.TimeoutExpired:
        res = {'setup': 'timeout'}
    except (json.JSONDecodeError, IndexError):
        res = {'setup': 'unparseable worker output'}
    finally:
        os.unlink(ex_path)
    return res


def outcome(rec):
    if rec is None or rec.get('kind') not in ('returned', 'raised'):
        return None
    if rec['kind'] == 'raised':
        return {'kind': 'raised', 'type': None, 'value': None, 'exception': rec['exception']}
    value = rec['value'] if rec['value'] is not None else (rec['stdout'] or None)
    return {'kind': 'returned', 'type': rec['type'], 'value': value, 'exception': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repos', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--since', default='2021-01-01')
    ap.add_argument('--max-per-repo', type=int, default=400)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--reuse', action='store_true', help='reuse candidates.jsonl from --out (skip mining)')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'snapshots'
    cache.mkdir(exist_ok=True)
    cands = []
    if a.reuse and (out / 'candidates.jsonl').exists():
        cands = [json.loads(l) for l in open(out / 'candidates.jsonl', encoding='utf-8')]
        print('reusing', len(cands), 'candidates', flush=True)
    else:
        for repo in sorted(Path(a.repos).iterdir()):
            if (repo / '.git').exists():
                c = candidates(repo, a.since, a.max_per_repo)
                print(repo.name, 'candidates', len(c), flush=True)
                cands += c
    for i, c in enumerate(cands):
        c['cid'] = f"{c['repo']}:{c['commit'][:10]}:{c['function']}"
    with open(out / 'candidates.jsonl', 'w', encoding='utf-8') as fh:
        for c in cands:
            fh.write(json.dumps(c) + '\n')
    jobs = []
    for c in cands:
        repo = Path(a.repos) / c['repo']
        jobs.append((c, 'parent', snapshot(repo, c['parent'], cache)))
        jobs.append((c, 'commit', snapshot(repo, c['commit'], cache)))
    with ProcessPoolExecutor(a.procs) as pool:
        results = list(pool.map(run_one, [(c, r, str(s)) for c, r, s in jobs], chunksize=4))
    with open(out / 'worker_results.jsonl', 'w', encoding='utf-8') as fh:
        for (c, r, s), res in zip(jobs, results):
            fh.write(json.dumps({'cid': c['cid'], 'rev': r, 'result': res}) + '\n')
    pairs, stats = [], {'candidates': len(cands), 'setup_failures': 0}
    for k, c in enumerate(cands):
        rb, ra = results[2 * k], results[2 * k + 1]
        if rb.get('setup') != 'complete' or ra.get('setup') != 'complete':
            stats['setup_failures'] += 1
            continue
        short = c['function'].split('.')[-1]
        db = {r['index']: r for r in rb['results']}
        da = {r['index']: r for r in ra['results']}
        for i, src in enumerate(c['examples']):
            if short not in src:
                continue
            ob, oa = outcome(db.get(i)), outcome(da.get(i))
            if ob is None or oa is None:
                continue
            bad = ('builtins.NameError', 'builtins.ImportError', 'builtins.ModuleNotFoundError')
            if ob['exception'] in bad or oa['exception'] in bad:
                continue                                 # setup artefacts, not behaviour
            srcs = c.get('sources', ['doc'] * len(c['examples']))
            docs = [e for e, k in zip(c['examples'], srcs) if k == 'doc']
            setup = c['examples'][:i] if srcs[i] == 'doc' else docs
            pairs.append({'id': f"{c['cid']}#{i}", 'repo': c['repo'], 'commit': c['commit'], 'parent': c['parent'],
                          'file': c['file'], 'module': c['module'], 'function': c['function'],
                          'setup': setup, 'invocation': src, 'source': srcs[i], 'before_source': c['before_source'],
                          'after_source': c['after_source'], 'before': ob, 'after': oa,
                          'changed': {f: ob[f] != oa[f] for f in ('kind', 'type', 'value', 'exception')}})
    for p in pairs:
        p['changed']['any'] = any(p['changed'].values())
    stats['pairs'] = len(pairs)
    stats['changed_pairs'] = sum(p['changed']['any'] for p in pairs)
    stats['by_repo'] = {}
    for p in pairs:
        s = stats['by_repo'].setdefault(p['repo'], [0, 0])
        s[0] += 1
        s[1] += p['changed']['any']
    with open(out / 'pairs.jsonl', 'w', encoding='utf-8') as fh:
        for p in pairs:
            fh.write(json.dumps(p) + '\n')
    (out / 'build.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
