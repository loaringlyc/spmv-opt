#!/usr/bin/env python3
"""
Plot comparison between custom kernel avg times and cuSPARSE avg times (ms)

Usage:
  python3 scripts/plot_compare_timings.py -o timings_compare.png

This script embeds your measured data from the table and produces a grouped
bar chart on a logarithmic y-axis with ticks emphasizing 1ms, 10ms, 100ms.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main(output):
    names = [
        'Ga41As41H72',
        'mip1',
        'shyy41',
        'Si41Ge41H72',
        'webbase-1M',
    ]

    # avg times per-iteration (ms) extracted from your table
    custom_ms = np.array([0.548, 0.249, 0.0194, 0.432, 1.02])
    cusparse_ms = np.array([0.406, 0.252, 0.0594, 0.351, 0.174])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(x - width/2, custom_ms, width, label='Custom kernel (avg ms)', color='#1f77b4')
    ax.bar(x + width/2, cusparse_ms, width, label='cuSPARSE (avg ms)', color='#ff7f0e')

    ax.set_yscale('log')

    # Set log ticks; emphasize 1ms,10ms,100ms as requested
    ticks = [0.01, 0.1, 1, 10, 100]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:g} ms' for t in ticks])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Time (ms, log scale)')
    ax.set_title('SpMV: Custom kernel vs cuSPARSE (avg per-iteration)')
    ax.legend()

    # annotate bars with values (in ms) for clarity
    def annotate(bars, vals):
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.annotate(f'{v:.3g}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    bars1 = ax.patches[:len(names)]
    bars2 = ax.patches[len(names):len(names)*2]
    annotate(bars1, custom_ms)
    annotate(bars2, cusparse_ms)

    plt.tight_layout()
    fig.savefig(output, dpi=200)
    print('Saved', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', help='output image path', default='timings_compare.png')
    args = parser.parse_args()
    main(args.out)
