"""Re-run the PatchCore reference row via Anomalib.

PatchCore is a *baseline to measure against*, not part of this project's model.
It is here so the baseline row in the results table is reproducible rather than
transcribed.

Needs the optional heavy extra:  pip install -r requirements-baseline.txt

    python scripts/patchcore_baseline.py --data-root data/MVTecAD --category bottle

Caveat, and it is a real one: this script was written against the Anomalib API
but has **not** been executed on this machine (no dataset, no anomalib install).
The numbers in results/history.json came from the original Kaggle run. Treat a
first run of this script as debugging, not as a measurement.
"""

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default="data/MVTecAD",
                   help="folder containing the category folder")
    p.add_argument("--category", default="bottle")
    p.add_argument("--out", default="results/runs/patchcore_baseline.json")
    a = p.parse_args(argv)

    try:
        from anomalib.data import MVTec
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as e:
        raise SystemExit(
            f"anomalib is not installed ({e}).\n"
            "  pip install -r requirements-baseline.txt"
        )

    datamodule = MVTec(root=a.data_root, category=a.category)
    model = Patchcore()
    # Anomalib's rich progress bar can hit a RecursionError inside a notebook
    # kernel; disabling it was the fix in the original run and costs nothing.
    engine = Engine(enable_progress_bar=False)

    engine.fit(model=model, datamodule=datamodule)
    metrics = engine.test(model=model, datamodule=datamodule)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps({"label": "PatchCore (Anomalib)", "raw_metrics": metrics}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, default=str))
    print(f"\nwrote {a.out}")
    print("Note: metric keys differ between Anomalib versions -- map them into "
          "results/history.json by hand rather than trusting an automatic merge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
