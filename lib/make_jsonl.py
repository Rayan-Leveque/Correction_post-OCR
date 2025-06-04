#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_jsonl.py

Extract every <line>…</line> from paired OCR and GT files in a single directory or nested layout,
leave sentence empty, and write one JSON object per line into a JSONL.

Supports:
  • Flat layout: OCR_AN_XXX.txt and GT_AN_XXX.md in the same folder.
  • Nested layout: .../OCR_AN_XXX/YYY.txt and corresponding GT_AN_XXX_YYY.md.

Usage:
  python make_jsonl.py <ocr_dir> <gt_dir> -d AN -o out.jsonl
"""
import os
import re
import sys
import json
import argparse
from glob import glob

LINE_RE = re.compile(r"<line>(.*?)</line>", re.DOTALL | re.IGNORECASE)
OCR_PREFIX = 'OCR_'
GT_PREFIX  = 'GT_'
GT_EXTS    = ('.md', '.txt')


def extract_lines(path):
    """Return list of all <line>…</line> contents."""
    text = open(path, encoding='utf-8').read()
    return [m.strip() for m in LINE_RE.findall(text)]


def traverse_ocr(ocr_dir):
    """Yield all .txt under ocr_dir (recursive)."""
    for root, _, files in os.walk(ocr_dir):
        for fn in files:
            if fn.lower().endswith('.txt'):
                yield os.path.join(root, fn)


def find_gt_path(ocr_path, gt_dir):
    """
    Determine the "core" ID from ocr_path, then look for GT_PREFIX+core+ext in gt_dir.
    Supports both:
      - Flat: basename startswith OCR_
      - Nested: parent folder startswith OCR_
    """
    base = os.path.basename(ocr_path)
    core = None
    # flat case: OCR_AN_...filename.txt → core = AN_...filename
    if base.startswith(OCR_PREFIX):
        core = os.path.splitext(base[len(OCR_PREFIX):])[0]
    else:
        # nested: .../OCR_AN_.../filename.txt
        parent = os.path.basename(os.path.dirname(ocr_path))
        if parent.startswith(OCR_PREFIX):
            part1 = parent[len(OCR_PREFIX):]
            part2 = os.path.splitext(base)[0]
            core = f"{part1}_{part2}"

    if not core:
        return None

    for ext in GT_EXTS:
        candidate = os.path.join(gt_dir, f"{GT_PREFIX}{core}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def make_jsonl(ocr_dir, gt_dir, dataset, output=None):
    out_f = open(output, 'w', encoding='utf-8') if output else None

    for ocr_path in traverse_ocr(ocr_dir):
        gt_path = find_gt_path(ocr_path, gt_dir)
        if not gt_path:
            print(f"⚠️  no GT for {ocr_path}", file=sys.stderr)
            continue

        ocr_lines = extract_lines(ocr_path)
        gt_lines  = extract_lines(gt_path)

        for o_line, g_line in zip(ocr_lines, gt_lines):
            rec = {
                'filename': os.path.relpath(ocr_path),
                'language': 'fr',
                'dataset_name': dataset,
                'ocr':       {'line': o_line, 'sentence': None},
                'groundtruth': {'line': g_line, 'sentence': None},
            }
            j = json.dumps(rec, ensure_ascii=False)
            if out_f:
                out_f.write(j + '\n')
            else:
                print(j)

    if out_f:
        out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract <line> only into JSONL (flat or nested OCR/GT)'
    )
    parser.add_argument('ocr_dir', help='root folder for OCR .txt files')
    parser.add_argument('gt_dir',  help='folder for GT_*.md/.txt')
    parser.add_argument('-d', '--dataset', default='AN', help='dataset name')
    parser.add_argument('-o', '--output', help='output JSONL file (stdout if omitted)')
    args = parser.parse_args()
    make_jsonl(args.ocr_dir, args.gt_dir, args.dataset, args.output)
