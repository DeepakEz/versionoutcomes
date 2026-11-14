"""Run the docstring examples of one function in one source snapshot (fresh process; called by build_cohort.py).

Usage: python -I worker.py ROOT MODULE EXAMPLES_JSON
ROOT is put first on sys.path; MODULE is imported; its globals (copied, as doctest does) are the namespace in which
the examples run in order. Each example that compiles as an expression is evaluated and its outcome recorded:
kind (returned/raised), root type, repr of the value (or captured stdout when the value is None), exception class.
Statements are executed for setup. Output: one JSON object on stdout.
"""
import contextlib
import io
import json
import re
import sys

ADDR = re.compile(r'0x[0-9a-fA-F]+')


def qual(t):
    return f'{t.__module__}.{t.__qualname__}'


def main():
    root, module, examples_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, root)
    examples = json.load(open(examples_path, encoding='utf-8'))
    out = {'setup': 'complete', 'results': []}
    try:
        mod = __import__(module, fromlist=['_'])
    except BaseException as e:  # noqa: BLE001
        out['setup'] = f'import failed: {type(e).__name__}: {str(e)[:200]}'
        print(json.dumps(out))
        return
    ns = dict(mod.__dict__)
    top = module.split('.')[0]
    if top not in ns:
        try:
            ns[top] = __import__(top)
        except BaseException:  # noqa: BLE001
            pass
    for i, src in enumerate(examples):
        buf = io.StringIO()
        rec = {'index': i}
        try:
            code = compile(src, f'<example {i}>', 'eval')
            is_expr = True
        except SyntaxError:
            try:
                code = compile(src, f'<example {i}>', 'exec')
            except SyntaxError as e:
                rec['kind'] = 'syntax_error'
                out['results'].append(rec)
                continue
            is_expr = False
        try:
            with contextlib.redirect_stdout(buf):
                value = eval(code, ns) if is_expr else exec(code, ns)
            printed = buf.getvalue()
            if not is_expr:
                rec.update(kind='statement', stdout=ADDR.sub('0xADDR', printed)[:400])
            else:
                rec.update(kind='returned', type=qual(type(value)),
                           value=ADDR.sub('0xADDR', repr(value))[:400] if value is not None else None,
                           stdout=ADDR.sub('0xADDR', printed)[:400])
        except BaseException as e:  # noqa: BLE001
            rec.update(kind='raised', exception=qual(type(e)), message=ADDR.sub('0xADDR', str(e))[:200],
                       stdout=ADDR.sub('0xADDR', buf.getvalue())[:400])
        out['results'].append(rec)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
