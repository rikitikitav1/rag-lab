import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config  # noqa: E402
import use_cases.ingest_quality as ingest_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Coverage report for one source. A dry run cuts in memory and touches "
        "no chunks, but it still appends an entry to the source's history."
    )
    parser.add_argument("source")
    parser.add_argument("--variant", default=config.settings.corpus.variant)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="cut the source in memory instead of reading the indexed rows",
    )
    args = parser.parse_args()

    entry = ingest_quality.analyze(
        args.source, variant=args.variant, mode="dry" if args.dry else "indexed"
    )
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    return 0 if entry["verdict"] != "broken" else 1


if __name__ == "__main__":
    raise SystemExit(main())
