#!/usr/bin/env python3
"""
Plot sparsity pattern of a Matrix Market (.mtx) file.
Usage:
  python scripts/plot_sparse_matrix.py matrix/test.mtx
  python scripts/plot_sparse_matrix.py matrix/test.mtx -o out.png --dpi 200 --markersize 1

Dependencies: numpy, scipy, matplotlib
"""
import argparse
import glob
import os
import sys
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import coo_matrix


def _plot_file(path, out, dpi, markersize):
    A = mmread(path)
    A = coo_matrix(A)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(A.col, A.row, s=markersize, marker='s', color='black')
    ax.set_xlim(-0.5, A.shape[1] - 0.5)
    ax.set_ylim(A.shape[0] - 0.5, -0.5)
    ax.set_xlabel('col j')
    ax.set_ylabel('row i')
    ax.set_title(f'Sparsity: {os.path.basename(path)}  {A.shape[0]}x{A.shape[1]}  nnz={A.nnz}')
    plt.tight_layout()

    out_path = out or os.path.splitext(path)[0] + '_sparsity.png'
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser(description='Plot sparsity pattern of a .mtx matrix or directory of .mtx files')
    parser.add_argument('matrix', help='path to .mtx file or directory containing .mtx files')
    parser.add_argument('-o', '--out', help='output image path (png) or output directory when input is a folder', default=None)
    parser.add_argument('--dpi', type=int, help='output DPI', default=200)
    parser.add_argument('--markersize', type=float, help='marker size for points', default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.matrix):
        print('Path not found:', args.matrix, file=sys.stderr)
        sys.exit(1)

    # If input is a directory, process all .mtx files inside
    if os.path.isdir(args.matrix):
        mtx_files = sorted(glob.glob(os.path.join(args.matrix, '*.mtx')))
        if not mtx_files:
            print('No .mtx files found in directory:', args.matrix, file=sys.stderr)
            sys.exit(1)

        out_dir = args.out or args.matrix
        if os.path.exists(out_dir) and not os.path.isdir(out_dir):
            print('Output path exists and is not a directory:', out_dir, file=sys.stderr)
            sys.exit(1)
        os.makedirs(out_dir, exist_ok=True)

        for p in mtx_files:
            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(out_dir, base + '_sparsity.png')
            _plot_file(p, out_path, args.dpi, args.markersize)
    else:
        _plot_file(args.matrix, args.out, args.dpi, args.markersize)


if __name__ == '__main__':
    main()
