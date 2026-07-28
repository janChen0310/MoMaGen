"""CLI wrapper: validate a PROCESSED source demo before syncing it to the generation server.

The importable logic lives in momagen/utils/source_demo_validation.py (momagen/scripts
is not a package, so it cannot be imported as momagen.scripts.validate_processed_source).
See that module's docstring for why this check exists.
"""
import argparse
import sys

from momagen.utils.source_demo_validation import open_or_error, scan_problems, scan_advisories


def main():
    # RawTextHelpFormatter so argparse does not word-wrap the example pin in the
    # middle of a hyphenated component name (behavior-1k-assets) — the example is
    # meant to be copy-pasted verbatim.
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--expect-versions", default=None,
                    help="comma list pinning EVERY component the file declares, e.g.\n"
                         "  omnigibson=3.7.1,bddl=3.7.0,behavior-1k-assets=3.7.2rc1\n"
                         "A partial pin is REJECTED: any declared component left\n"
                         "unpinned would drift silently on the server (an asset-hash\n"
                         "mismatch is only a warning there). The rejection message\n"
                         "for an incomplete pin lists the file's full declared set,\n"
                         "so it can be copied straight back into this flag.")
    args = ap.parse_args()

    expected = None
    if args.expect_versions:
        expected = dict(kv.split("=", 1) for kv in args.expect_versions.split(","))

    # Open once and reuse the handle for both the fatal scan and the advisory scan
    # (validate_processed_source/sync_advisories each open independently, which is
    # fine for library callers but wasteful here since the CLI needs both).
    f, problems = open_or_error(args.path)
    if f is not None:
        with f:
            problems = scan_problems(f, expected)
            if not problems:
                for advisory in scan_advisories(f):
                    print("advisory:", advisory)

    if problems:
        print("INVALID — do not sync:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("VALID — safe to sync:", args.path)


if __name__ == "__main__":
    main()
