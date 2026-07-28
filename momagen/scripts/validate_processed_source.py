"""CLI wrapper: validate a PROCESSED source demo before syncing it to the generation server.

The importable logic lives in momagen/utils/source_demo_validation.py (momagen/scripts
is not a package, so it cannot be imported as momagen.scripts.validate_processed_source).
See that module's docstring for why this check exists.
"""
import argparse
import sys

from momagen.utils.source_demo_validation import validate_processed_source, sync_advisories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect-versions", default=None,
                    help="comma list, e.g. omnigibson=3.7.1,bddl=3.7.0")
    args = ap.parse_args()

    expected = None
    if args.expect_versions:
        expected = dict(kv.split("=", 1) for kv in args.expect_versions.split(","))

    problems = validate_processed_source(args.path, expected)
    if problems:
        print("INVALID — do not sync:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    for advisory in sync_advisories(args.path):
        print("advisory:", advisory)
    print("VALID — safe to sync:", args.path)


if __name__ == "__main__":
    main()
