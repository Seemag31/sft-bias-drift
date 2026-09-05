"""Mirror evaluation outputs from the Modal volume into artifacts/ for local analysis.

Volume layout -> local layout
  eval/bbq/<model>/<run>[__hardened]/   -> artifacts/eval/bbq/<model>/<run>[__guided]/
  eval/domain_ext/<domain>/<model>/<run>/ -> artifacts/eval/capability/<domain>/<model>/<run>/
  eval/task8/<model>/<run>/<mode>/       -> artifacts/eval/deliberation/<model>/<run>/<mode>/
  runs/<run>/<model>/run_config.json     -> artifacts/runs/<run>/<model>/run_config.json

Only predictions.parquet, eval_meta.json, and run_config.json are fetched.

Run: python evaluation/download_results.py
"""
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import VOLUME_NAME  # noqa: E402

LOCAL = ROOT / "artifacts"
MAP = {"eval/bbq": "eval/bbq", "eval/domain_ext": "eval/capability",
       "eval/task8": "eval/deliberation", "runs": "runs"}
KEEP = {"predictions.parquet", "eval_meta.json", "run_config.json"}


def main():
    vol = modal.Volume.from_name(VOLUME_NAME)
    n = 0
    for remote, local in MAP.items():
        for e in vol.listdir(remote, recursive=True):
            name = os.path.basename(e.path)
            if name not in KEEP:
                continue
            rel = e.path[len(remote):].lstrip("/").replace("__hardened", "__guided")
            dest = LOCAL / local / rel
            if dest.exists() and dest.stat().st_size == e.size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in vol.read_file(e.path):
                    f.write(chunk)
            n += 1
            print("fetched", dest.relative_to(ROOT))
    print(f"done: {n} new files under {LOCAL}")


if __name__ == "__main__":
    main()
