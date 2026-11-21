"""Tests for src/public_api_worker.py on a synthetic package (run: python -m pytest tests)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / 'src' / 'public_api_worker.py'


@pytest.fixture()
def pkg(tmp_path):
    p = tmp_path / 'demo'
    p.mkdir()
    (p / '__init__.py').write_text(
        'from demo.core import api, Widget, wrapped\n'
        '__all__ = ["api", "Widget", "wrapped"]\n')
    (p / 'core.py').write_text(
        'import functools\n'
        'def api(x): return x\n'
        'def helper(x): return x\n'
        'def _inner(x): return x\n'
        'def wrapped(x): return x\n'
        'wrapped = functools.wraps(wrapped)(lambda x: x)\n'
        'class Widget:\n'
        '    def __init__(self): pass\n'
        '    def run(self): return 1\n'
        '    def _private(self): return 2\n'
        'class Hidden:\n'
        '    def run(self): return 3\n')
    (p / 'compat.py').write_text('def fast(x): return x\n')
    return tmp_path


def check(root, module, qualname, package='demo'):
    r = subprocess.run([sys.executable, '-E', '-P', str(WORKER), str(root), package, module, qualname],
                       capture_output=True, text=True, timeout=30)
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('module,qualname,public', [
    ('demo.core', 'api', True),             # exported by __all__
    ('demo.core', 'helper', False),         # defined but not exported
    ('demo.core', '_inner', False),
    ('demo.compat', 'fast', False),         # only reachable through a submodule
    ('demo.core', 'Widget.run', True),      # method of an exported class
    ('demo.core', 'Widget.__init__', True),  # dunders affect users
    ('demo.core', 'Widget._private', False),
    ('demo.core', 'Hidden.run', False),     # class not exported
    ('demo.core', 'wrapped', True),         # functools.wraps unwrapped
])
def test_public_rule(pkg, module, qualname, public):
    out = check(pkg, module, qualname)
    assert out['error'] is None and out['exists'] and out['public'] is public


def test_missing_function_reports_absent(pkg):
    out = check(pkg, 'demo.core', 'gone')
    assert out['exists'] is False and out['public'] is False


def test_no_all_uses_non_underscore_names(tmp_path):
    p = tmp_path / 'plain'
    p.mkdir()
    (p / '__init__.py').write_text('def visible(): pass\ndef _hidden(): pass\n')
    assert check(tmp_path, 'plain', 'visible', 'plain')['public'] is True
    assert check(tmp_path, 'plain', '_hidden', 'plain')['public'] is False


@pytest.fixture()
def modpkg(tmp_path):
    """Package whose __all__ lists a submodule (the sqlparse pattern), next to one it only imports."""
    p = tmp_path / 'mp'
    p.mkdir()
    (p / '__init__.py').write_text('from mp import sql, helpers\n__all__ = ["parse", "sql"]\n'
                                   'def parse(s): return s\n')
    (p / 'sql.py').write_text('class Token:\n    def get_name(self): return 1\n    def _peek(self): return 2\n'
                              'def group(x): return x\ndef _inner(x): return x\n')
    (p / 'helpers.py').write_text('def fast(x): return x\n')
    return tmp_path


@pytest.mark.parametrize('module,qualname,public', [
    ('mp', 'parse', True),
    ('mp.sql', 'Token.get_name', True),     # method of a class in a listed submodule
    ('mp.sql', 'group', True),
    ('mp.sql', 'Token._peek', False),
    ('mp.sql', '_inner', False),
    ('mp.helpers', 'fast', False),          # imported by the package but not listed in __all__
])
def test_listed_submodule_rule(modpkg, module, qualname, public):
    out = check(modpkg, module, qualname, 'mp')
    assert out['error'] is None and out['exists'] and out['public'] is public


def test_no_all_does_not_expand_submodules(tmp_path):
    p = tmp_path / 'na'
    p.mkdir()
    (p / '__init__.py').write_text('from na import util\ndef top(): pass\n')
    (p / 'util.py').write_text('def deep(): pass\n')
    assert check(tmp_path, 'na', 'top', 'na')['public'] is True
    assert check(tmp_path, 'na.util', 'deep', 'na')['public'] is False


def test_import_failure_is_reported(tmp_path):
    p = tmp_path / 'broken'
    p.mkdir()
    (p / '__init__.py').write_text('raise RuntimeError("boom")\n')
    out = check(tmp_path, 'broken', 'x', 'broken')
    assert out['public'] is False and 'RuntimeError' in out['error']
