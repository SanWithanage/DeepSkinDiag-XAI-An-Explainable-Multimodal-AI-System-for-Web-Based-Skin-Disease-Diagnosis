#!/usr/bin/env python3
"""
Read and display multiclass labels from labels_26.txt or from a PyTorch checkpoint.
Safe: read-only; does not change any files unless you pass --export.

Usage examples:
  python show_labels.py
  python show_labels.py --labels-file artifacts/labels_26.txt
  python show_labels.py --ckpt artifacts/checkpoints/multiclass/best.pt
  python show_labels.py --json
  python show_labels.py --export labels_export.txt
"""

import os
import json
import argparse
from typing import List, Optional, Any, Dict, Tuple

try:
    import torch  # only needed if loading from checkpoint
except Exception as e:
    torch = None

DEFAULT_LABELS_FILE = "labels_26.txt"
DEFAULT_CKPT = "artifacts/checkpoints/multiclass/best.pt"

LABEL_KEYS = {"labels", "class_names", "classes", "idx_to_class"}


def _read_lines(p: str) -> List[str]:
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _normalize_label_container(v: Any) -> Optional[List[str]]:
    """
    Try to turn common label containers into a flat list[str].
    Supports: list/tuple of str/int, or dict index->name (string or int keys).
    """
    # list/tuple
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if isinstance(x, (str, int)):
                out.append(str(x))
            else:
                return None
        return out if out else None

    # dict (often idx->class_name)
    if isinstance(v, dict):
        items: List[Tuple[int, str]] = []
        # try to interpret keys as integers for stable order
        for k, val in v.items():
            try:
                idx = int(k)
            except Exception:
                # if keys aren't numeric, just collect in insertion order later
                idx = None
            if isinstance(val, (str, int)):
                items.append((idx, str(val)))
            else:
                return None
        # numeric keys first sorted, then non-numeric in original order
        numeric = [(i, s) for i, s in items if i is not None]
        nonnum = [s for i, s in items if i is None]
        numeric.sort(key=lambda z: z[0])
        return [s for _, s in numeric] + nonnum

    return None


def _extract_labels_from_flat_dict(d: Dict[str, Any]) -> Optional[List[str]]:
    """Check only the current dict level for known label keys."""
    for key in LABEL_KEYS:
        if key in d:
            out = _normalize_label_container(d[key])
            if out:
                return out
    return None


def _extract_labels_recursive(obj: Any, depth: int = 0, max_depth: int = 4) -> Optional[List[str]]:
    """
    Recursively search for known label keys in nested dicts (common in checkpoints).
    Stops at the first valid list it finds.
    """
    if depth > max_depth:
        return None

    if isinstance(obj, dict):
        # try at this level
        out = _extract_labels_from_flat_dict(obj)
        if out:
            return out
        # recurse into children
        for v in obj.values():
            res = _extract_labels_recursive(v, depth + 1, max_depth)
            if res:
                return res
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            res = _extract_labels_recursive(v, depth + 1, max_depth)
            if res:
                return res
    return None


def _load_labels_from_checkpoint(ckpt_path: str) -> List[str]:
    if torch is None:
        raise RuntimeError("PyTorch is required to read labels from a checkpoint, but it's not available.")

    if not os.path.isfile(ckpt_path):
        return []

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Best-effort recursive search for labels inside checkpoint dict
    labels = _extract_labels_recursive(ckpt)
    if labels:
        return [str(x) for x in labels if str(x).strip()]

    # Some checkpoints store the actual state_dict in 'state_dict'; labels elsewhere.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Try again without the state_dict
        shallow = {k: v for k, v in ckpt.items() if k != "state_dict"}
        labels = _extract_labels_recursive(shallow)
        if labels:
            return [str(x) for x in labels if str(x).strip()]

    return []


def find_labels(labels_file: str, ckpt_path: str, prefer_file: bool = True) -> Tuple[List[str], str]:
    """
    Return (labels, source). Preference:
      - If prefer_file and labels_file exists -> use it
      - else, try to read from ckpt
      - else, if labels_file exists after all -> use it
      - else, []
    """
    file_labels = _read_lines(labels_file)
    if prefer_file and file_labels:
        return (file_labels, f"file:{labels_file}")

    ckpt_labels = _load_labels_from_checkpoint(ckpt_path)
    if ckpt_labels:
        return (ckpt_labels, f"ckpt:{ckpt_path}")

    # fallback to file if present
    if file_labels:
        return (file_labels, f"file:{labels_file}")

    return ([], "none")


def main():
    ap = argparse.ArgumentParser(description="Display multiclass labels from labels_26.txt or a checkpoint.")
    ap.add_argument("--labels-file", default=DEFAULT_LABELS_FILE, help="Path to labels_26.txt")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to multiclass checkpoint .pt/.pth")
    ap.add_argument("--no-prefer-file", action="store_true", help="Prefer checkpoint over labels file if both exist")
    ap.add_argument("--json", action="store_true", help="Print as JSON array only")
    ap.add_argument("--export", default="", help="Optional path to write labels as a plain text file")
    args = ap.parse_args()

    labels, source = find_labels(
        labels_file=args.labels_file,
        ckpt_path=args.ckpt,
        prefer_file=not args.no_prefer_file
    )

    if not labels:
        print("No labels found. Checked:")
        print(f" - {args.labels_file}")
        print(f" - {args.ckpt}")
        raise SystemExit(1)

    if args.json:
        print(json.dumps(labels, ensure_ascii=False, indent=2))
    else:
        print(f"[SOURCE] {source}")
        print(f"[COUNT]  {len(labels)} labels")
        for i, lbl in enumerate(labels):
            print(f"{i:02d}\t{lbl}")

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            for i, lbl in enumerate(labels):
                f.write(f"{i:02d}\t{lbl}\n")
        print(f"[WRITE]  Exported to {args.export}")


if __name__ == "__main__":
    main()
