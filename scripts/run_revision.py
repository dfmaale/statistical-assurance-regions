"""Run the revision-5 analyses and regenerate revision figures.

Usage (from the repository root):

    python scripts/run_revision.py            # full profile (results/revision, figures/revision)
    python scripts/run_revision.py --smoke    # quick check (results/revision_smoke)
    python scripts/run_revision.py --figures-only

The original full-run outputs in results/full are verified against
results/full/run_manifest.json before they are read; they are never rewritten.
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threadpoolctl import threadpool_limits
from sciguard.revision import RevisionConfig, RevisionRun, REVISION_SECTIONS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_original_full_run():
    manifest = json.loads((ROOT / "results/full/run_manifest.json").read_text())
    if not str(manifest.get("status", "")).startswith("completed"):
        raise RuntimeError("Original full run is not marked completed.")
    for name, digest in manifest["output_sha256"].items():
        if sha256(ROOT / "results/full" / name) != digest:
            raise RuntimeError(f"Original full-run output changed: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--sections", nargs="*", default=None)
    args = parser.parse_args()
    verify_original_full_run()
    config = RevisionConfig.smoke() if args.smoke else RevisionConfig()
    out = ROOT / "results" / ("revision" if not args.smoke else "revision_smoke")
    if not args.figures_only:
        run = RevisionRun(ROOT, config)
        names = args.sections or list(REVISION_SECTIONS)
        with threadpool_limits(1):
            for name in names:
                run.section(name, REVISION_SECTIONS[name])
        run.finish()
    if not args.smoke and not args.no_figures:
        from sciguard.presentation import make_all_figures
        make_all_figures(ROOT)
    print("Revision outputs:", out)


if __name__ == "__main__":
    main()
