# VersionOutcomes v2: real before/after outcomes of Python function calls across library commits

Also on Kaggle: https://www.kaggle.com/datasets/kradle2grave/versionoutcomes-v2. The full write-up, with results for all five Gemma 4 sizes, is [PAPER.md](PAPER.md).

Rule-mined part of the resource described in the Kaggle Gemma 4 Developer Agent paper-track writeup
"Same Function, Different Outcome". The curated v1 part (42 pairs) is the earlier release, unchanged.

## Contents
- `cohort/candidates.jsonl` - every changed function (body differs, docstring ignored) in non-merge commits since
  2021-01-01 of 16 libraries (`cohort/libraries.txt`: URL and the HEAD commit we mined), with its doc examples and
  the calls on test lines added by the same commit.
- `cohort/pairs.jsonl` - 1,973 executed invocation pairs: before/after outcome (kind, type, repr value or printed
  output, exception) and per-field change flags. `cohort/build.json` has the counts.
- `eval/items.jsonl` - the 82-pair evaluation set (all 42 changed pairs after dedupe and a 3-per-function cap, plus
  40 unchanged pairs sampled per library with seed 42; `eval/selection.json`), with the exact prompts and gold.
  `eval/items_obs.jsonl` is the observed-BEFORE condition.
- `answers/` - raw model answers (one JSON line per item), scored with `src/score.py`.
- `src/` - `build_cohort.py` (mine + execute), `worker.py` (runs one snapshot's calls in a fresh process),
  `eval_set.py` (selection + prompts), `run_local.py` (OpenAI-compatible client for llama.cpp), `score.py`,
  `make_figs.py`; `kaggle/run_gemma4.py` runs the other Gemma 4 sizes on a Kaggle T4 x2 session.

## Reproduce
1. Clone the libraries at the listed commits into `repos/`.
2. `python src/build_cohort.py --repos repos --out cohort --since 2021-01-01 --max-per-repo 250`
   (executes the libraries' own documentation and test calls; run it in a sandbox you trust).
3. `python src/eval_set.py --pairs cohort/pairs.jsonl --out eval --per-function 3 --seed 42`
4. Serve an official Gemma 4 QAT Q4_0 GGUF with llama.cpp b11552 (`--reasoning-budget 0 --no-cache-prompt`), then
   `python src/run_local.py --items eval/items.jsonl --out answers/<model>.jsonl --model <model>` and
   `python src/score.py --items eval/items.jsonl --answers answers/<model>.jsonl --out scores.json --split-out split.json`.
   Answers without a parseable change verdict score zero on every measure, including field credit (corrected 2026-11-12; see PAPER.md, Limitations).

## Watching a library's new commits (maintainer pilot)
`src/watch_repo.py` runs the same mining rule on a range of commits of one library and reports only what a user could
notice: functions the package exports (`__all__`, or its non-underscore top-level names; methods of exported classes;
`src/public_api_worker.py` checks this in the commit's own snapshot) whose calls behave differently from the latest
release tag (default) or from the parent commit (`--baseline parent`). Fixes to regressions that were never released
drop out, and functions missing from the release are listed as added. Only calls found in docstrings and in test
lines added by each commit are run, so a change none of them exercises is not seen.

    python src/watch_repo.py --repo repos/natsort --package natsort --range 8.4.0..origin/main --out pilot/natsort

`report.md` is the note to post; `changes.json` lists every compared function and call with the reason it was kept or
filtered. Tests: `python -m pytest tests`.

## Licences
Original code and data: Apache-2.0. Function sources and invocations quoted from the 16 libraries remain under their
own permissive licences (MIT, BSD, Apache-2.0); their licence texts are in `third_party_licenses/`.
