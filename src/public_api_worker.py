"""Is a changed function part of a package's public API in one source snapshot? (fresh process; watch_repo.py)

Usage: python -E -P public_api_worker.py ROOT PACKAGE MODULE QUALNAME
ROOT is put first on sys.path. The public API is what the package's top level exports: its __all__ when defined,
else every top-level name that does not start with '_'. When __all__ explicitly lists a module (sqlparse lists 'sql'
and 'cli'), that module's public names are exported too, as 'sql.Function': its own __all__ when defined, else its
non-underscore names; one level only. Without __all__, submodules are not expanded. A function is public when one of those exported objects IS
it (functools.wraps wrappers unwrapped). A method Class.name is public when the class itself is exported and the
method name is not private (a leading '_' that is not a dunder such as __init__). Functions reachable only through
a submodule the package does not export (pkg.sub.func) do not count.
Prints one JSON line: {"exists": bool, "public": bool, "exported_as": [names], "error": str|None}.
"""
import importlib
import json
import sys
import types


def unwrap(obj):
    seen = set()
    while hasattr(obj, '__wrapped__') and id(obj) not in seen:
        seen.add(id(obj))
        obj = obj.__wrapped__
    return obj


def private(name):
    return name.startswith('_') and not (name.startswith('__') and name.endswith('__'))


def main():
    root, package, module, qualname = sys.argv[1:5]
    sys.path.insert(0, root)
    out = {'exists': False, 'public': False, 'exported_as': [], 'error': None}
    try:
        pkg = importlib.import_module(package)
        listed = getattr(pkg, '__all__', None)
        names = list(listed) if listed is not None else [n for n in dir(pkg) if not n.startswith('_')]
        exported = {n: getattr(pkg, n, None) for n in names}
        for n, o in list(exported.items()):
            if listed is not None and isinstance(o, types.ModuleType):
                sub = getattr(o, '__all__', None)
                sub = list(sub) if sub is not None else [s for s in dir(o) if not s.startswith('_')]
                exported.update({f'{n}.{s}': getattr(o, s, None) for s in sub})
        mod = importlib.import_module(module)
        parts = qualname.split('.')
        if len(parts) == 1:
            target = getattr(mod, parts[0], None)
            out['exists'] = target is not None
            if target is not None:
                t = unwrap(target)
                out['exported_as'] = sorted(n for n, o in exported.items() if o is not None and unwrap(o) is t)
        else:
            cls = getattr(mod, parts[0], None)
            out['exists'] = cls is not None and hasattr(cls, parts[-1])
            if out['exists'] and not private(parts[-1]):
                out['exported_as'] = sorted(f'{n}.{parts[-1]}' for n, o in exported.items() if o is cls)
        out['public'] = bool(out['exported_as'])
    except Exception as e:                      # the snapshot's own import failure is reported, not raised
        out['error'] = f'{type(e).__module__}.{type(e).__qualname__}: {e}'[:300]
    print(json.dumps(out))


if __name__ == '__main__':
    main()
