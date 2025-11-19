#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

from scripts.utils import setup_reproducibility, shut_down_warnings

shut_down_warnings()
setup_reproducibility(seed=73)


def parse_dirname(dirname):
    """
    Expected format:
        mae_<pct>_<labels>_full
    Returns:
        (pct:int, labels:int) or (None, None) if not matching.
    """
    match = re.match(r"mae_(\d+)_(\d+)_full", dirname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def main():
    train_dir = Path("outputs/train")
    out_dir = Path("assets/weights")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Scanning for full fine-tuned checkpoints in: {train_dir}")

    exported = 0

    for run_dir in train_dir.iterdir():
        if not run_dir.is_dir():
            continue

        dirname = run_dir.name
        pct, labels = parse_dirname(dirname)

        # Skip if not a "full" run
        if pct is None:
            continue

        ckpt_path = run_dir / "checkpoints" / "best.ckpt"

        if not ckpt_path.exists():
            print(f"⚠️ Missing best.ckpt in {dirname}, skipping...")
            continue

        # Output filename format: mae_100_400.ckpt
        out_name = f"mae_{pct:03d}_{labels:03d}.ckpt"
        out_path = out_dir / out_name

        shutil.copy2(ckpt_path, out_path)
        print(f"📦 Copied: {ckpt_path}  →  {out_path}")
        exported += 1

    print(f"\n✅ Done. Exported {exported} checkpoints to: {out_dir}")


if __name__ == "__main__":
    main()
