"""Convert existing teacher choices to student SFT records."""

import argparse
from .data import export_rows, load_inputs, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = export_rows(load_inputs(args.inputs), read_jsonl(args.teacher))
    write_jsonl(args.output, rows)
    print(f"Exported {len(rows)} training records")


if __name__ == "__main__":
    main()
