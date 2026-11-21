"""Pilot watcher: public-API behaviour changes in a library's new commits (2026-11-21).

For every non-merge commit in --range that changes Python source, changed functions and their calls are mined with the
rule of build_cohort.py: the function's doctest examples plus the calls of it on test lines added by the same commit.
Two filters then keep only what a user of the library could notice:
  public API  the function must be exported by the package at that commit (public_api_worker.py: the package's
              __all__, or its non-underscore top-level names; methods of exported classes count);
  baseline    each call's outcome in the commit snapshot is compared with the latest release tag reachable from the
              parent (--baseline release, the default) or with the parent commit (--baseline parent). Against the
              release, a fix to a regression that was never released does not show up, and a function absent from
              the release is listed as added rather than changed.
Every call runs in a fresh process per snapshot (worker.py, PYTHONHASHSEED=0, 20 s). Missing-name and import errors
are setup artefacts and are skipped, as in build_cohort.py.
Outputs in --out: changes.json (every compared function and call, with the reason it was kept or filtered) and
report.md (the note to post; it says so explicitly when nothing changed).
Usage: watch_repo.py --repo DIR --package NAME --range A..B [--baseline release|parent] --out DIR
"""
import argparse
import doctest
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_cohort import SKIP_PARTS, functions, git, module_of, outcome, run_one, snapshot, test_calls  # noqa: E402

SETUP_ERRORS = ('builtins.NameError', 'builtins.ImportError', 'builtins.ModuleNotFoundError')
FIELDS = ('kind', 'type', 'value', 'exception')


def commits(repo, rng):
    out = []
    for line in git(repo, 'log', '--no-merges', '--reverse', '--format=%H %P', rng, '--', '*.py').decode().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def mine(repo, c, p):
    """Changed functions of one commit with their calls: the candidates() rule of build_cohort.py for one commit."""
    out = []
    for f in git(repo, 'diff', '--name-only', p, c, '--', '*.py').decode().split():
        if set(Path(f).parts[:-1]) & SKIP_PARTS or Path(f).name.startswith('test') or \
                Path(f).name in ('setup.py', 'conftest.py', 'noxfile.py'):
            continue
        try:
            a = git(repo, 'show', f'{c}:{f}').decode('utf-8', 'replace')
            b = git(repo, 'show', f'{p}:{f}').decode('utf-8', 'replace')
        except subprocess.CalledProcessError:
            continue
        fa, fb = functions(a), functions(b)
        for q in sorted(set(fa) & set(fb)):
            if fa[q][0] == fb[q][0]:
                continue
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
                        'sources': ['doc'] * len(ex) + ['test'] * len(tests)})
    return out


def public(snap, cand, package):
    snap = Path(snap).resolve()
    root = str(snap / cand['root']) if cand['root'] else str(snap)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONHASHSEED'] = '0'
    try:
        r = subprocess.run([sys.executable, '-E', '-P', str(HERE / 'public_api_worker.py'), root, package,
                            cand['module'], cand['function']], capture_output=True, timeout=20, env=env,
                           cwd=tempfile.gettempdir())
        return json.loads(r.stdout.decode('utf-8', 'replace').strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as e:
        return {'exists': False, 'public': False, 'exported_as': [], 'error': f'worker: {type(e).__name__}'}


def baseline(repo, parent, mode):
    if mode == 'parent':
        return parent, f'parent {parent[:10]}'
    try:
        tag = git(repo, 'describe', '--tags', '--abbrev=0', parent).decode().strip()
    except subprocess.CalledProcessError:
        return parent, f'parent {parent[:10]} (no release tag reachable)'
    return git(repo, 'rev-list', '-n', '1', tag).decode().strip(), f'release {tag}'


def show(o):
    return f"raises {o['exception'].removeprefix('builtins.')}" if o['kind'] == 'raised' else f"returns {o['value']}"


def cell(s):
    return str(s).replace('|', '\\|').replace('\n', ' ')[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--range', required=True, help='git revision range, e.g. 8.4.0..main or SHA^..SHA')
    ap.add_argument('--baseline', choices=('release', 'parent'), default='release')
    ap.add_argument('--out', required=True)
    ap.add_argument('--url', help='web URL for commit links (default: the remote origin)')
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'snapshots'
    cache.mkdir(exist_ok=True)
    url = a.url or git(repo, 'config', '--get', 'remote.origin.url').decode().strip().removesuffix('.git')
    rows, n_commits = [], 0
    for c, p in commits(repo, a.range):
        n_commits += 1
        cands = mine(repo, c, p)
        if not cands:
            continue
        snap_c = snapshot(repo, c, cache)
        brev, blabel = baseline(repo, p, a.baseline)
        snap_b = snapshot(repo, brev, cache)
        for cand in cands:
            base = {'commit': c, 'function': cand['function'], 'module': cand['module'], 'baseline': blabel}
            pub = public(snap_c, cand, a.package)
            if not pub['public']:
                rows.append({**base, 'status': 'filtered: not in the public API', 'detail': pub})
                continue
            if not public(snap_b, cand, a.package)['exists']:
                rows.append({**base, 'status': 'added since baseline', 'exported_as': pub['exported_as']})
                continue
            ra, rb = run_one((cand, 'commit', str(snap_c))), run_one((cand, 'baseline', str(snap_b)))
            if ra.get('setup') != 'complete' or rb.get('setup') != 'complete':
                rows.append({**base, 'status': 'skipped: setup failed',
                             'detail': {'commit': ra.get('setup'), 'baseline': rb.get('setup')}})
                continue
            da = {r['index']: r for r in ra['results']}
            db = {r['index']: r for r in rb['results']}
            short = cand['function'].split('.')[-1]
            for i, src in enumerate(cand['examples']):
                ob, oa = outcome(db.get(i)), outcome(da.get(i))
                if short not in src or ob is None or oa is None or ob['exception'] in SETUP_ERRORS \
                        or oa['exception'] in SETUP_ERRORS:
                    continue
                changed = any(ob[f] != oa[f] for f in FIELDS)
                rows.append({**base, 'exported_as': pub['exported_as'], 'call': src, 'source': cand['sources'][i],
                             'before': ob, 'after': oa,
                             'status': 'REPORT: public behaviour change' if changed else 'unchanged vs baseline'})
    (out / 'changes.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    rep = [r for r in rows if r['status'].startswith('REPORT')]
    added = sorted({(r['commit'], r['function']) for r in rows if r['status'] == 'added since baseline'})
    filt = sorted({(r['commit'], r['function']) for r in rows if r['status'].startswith('filtered')})
    unchanged = sum(r['status'] == 'unchanged vs baseline' for r in rows)
    base_desc = 'the latest release before each commit' if a.baseline == 'release' else 'each commit\'s parent'
    lines = [f'# Public-API behaviour changes: {a.package}, {a.range}', '',
             f'Each commit compared with {base_desc}. {n_commits} commits touched Python files; '
             f'{len({(r["commit"], r["function"]) for r in rows})} changed functions had runnable calls; '
             f'{len(filt)} were not in the public API and were left out; {unchanged} public calls were unchanged. '
             'Only calls found in docstrings and in test lines added by each commit are run, so a change that none '
             'of them exercises is not seen.', '']
    if rep:
        lines += [f'## Changed behaviour ({len(rep)} calls)', '', '| commit | function | call | before | after |',
                  '|---|---|---|---|---|']
        lines += [f"| [{r['commit'][:8]}]({url}/commit/{r['commit']}) | `{r['exported_as'][0]}` | `{cell(r['call'])}` "
                  f"| {cell(show(r['before']))} ({r['baseline']}) | {cell(show(r['after']))} |" for r in rep]
    else:
        lines.append('No public-API behaviour changes in this range.')
    if added:
        lines += ['', f'Added to the public API since the baseline: ' +
                  ', '.join(f'`{f}` ([{c[:8]}]({url}/commit/{c}))' for c, f in added)]
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
