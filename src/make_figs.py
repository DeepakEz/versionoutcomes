"""Figures for the paper (2026-11-11). Usage: make_figs.py --scores scores.json --items eval_v1/items.jsonl --out figs
Fig 1 pipeline, Fig 2 results by model, Fig 3 change-detection confusion per model, Fig 4 pre/post cutoff.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ORDER = ['gemma4-e2b-qat-q4', 'gemma4-e4b-qat-q4', 'gemma4-12b-qat-q4', 'gemma4-26b-a4b-qat-q4', 'gemma4-31b-qat-q4']
LABEL = {'gemma4-e2b-qat-q4': 'E2B', 'gemma4-e4b-qat-q4': 'E4B', 'gemma4-12b-qat-q4': '12B',
         'gemma4-26b-a4b-qat-q4': '26B A4B', 'gemma4-31b-qat-q4': '31B'}
INK, MUTED, A1, A2, A3 = '#1f2933', '#7b8794', '#2f6fb5', '#d9822b', '#3e9d6b'


def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=150)
    ax.axis('off')
    steps = [('Git history\n16 Python libraries\nsince 2021', A1),
             ('Changed function\n(body differs,\ndocstring ignored)', A1),
             ("Invocations\ndoc examples +\ncommit's new test calls", A1),
             ('Run in BEFORE and\nAFTER snapshots\n(fresh processes)', A3),
             ('Outcome pair\nkind, type, value,\nexception + flags', A3),
             ('Prompt without\ndocstrings\n-> Gemma 4 sizes', A2)]
    w, h, gap = 1.75, 1.5, 0.28
    for i, (txt, col) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                                    fc=col, ec='none', alpha=0.13))
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                                    fc='none', ec=col, lw=1.6))
        ax.text(x + w / 2, h / 2, txt, ha='center', va='center', fontsize=10, color=INK)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + w + gap - 0.02, h / 2), xytext=(x + w + 0.02, h / 2),
                        arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.5))
    ax.set_xlim(-0.1, len(steps) * (w + gap))
    ax.set_ylim(-0.2, h + 0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_results(scores, path):
    models = [m for m in ORDER if m in scores]
    x = range(len(models))
    acc = [scores[m]['change_correct'] / scores[m]['n'] for m in models]
    rec = [scores[m]['change_recall'] for m in models]
    comp = [scores[m]['paired_complete'] / scores[m]['n'] for m in models]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    bw = 0.26
    for k, (vals, lab, col) in enumerate([(acc, 'Change verdict correct (all pairs; no verdict = wrong)', A1),
                                          (rec, 'Changed outcomes caught (recall)', A2),
                                          (comp, 'Complete pair: both outcomes + change verdict right', A3)]):
        bars = ax.bar([i + (k - 1) * bw for i in x], vals, bw, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.0%}', ha='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls='--')
    ax.text(-0.45, 0.515, 'coin flip on a balanced set', color=MUTED, fontsize=8, ha='left')
    ax.set_xticks(list(x), [LABEL[m] for m in models])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('share of evaluation pairs')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cutoff(split, path):
    models = [m for m in ORDER if m in split]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    bw = 0.36
    for k, (key, lab, col) in enumerate([('pre', 'commits before Jan 2025 (training cutoff)', MUTED),
                                         ('post', 'commits after Jan 2025', A1)]):
        vals = [split[m][key]['change_acc'] if isinstance(split[m][key], dict) else split[m][key] for m in models]
        bars = ax.bar([i + (k - 0.5) * bw for i in range(len(models))], vals, bw, label=lab, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.0%}', ha='center', fontsize=8)
    ax.set_xticks(range(len(models)), [LABEL[m] for m in models])
    ax.set_ylim(0, 1)
    ax.set_ylabel('change verdict correct\n(no verdict counts as wrong)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--split', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = json.load(open(a.scores))
    fig_pipeline(out / 'fig1_pipeline.png')
    fig_results(scores, out / 'fig2_results.png')
    if a.split:
        fig_cutoff(json.load(open(a.split)), out / 'fig3_cutoff.png')
    print('figures in', out)


if __name__ == '__main__':
    main()
