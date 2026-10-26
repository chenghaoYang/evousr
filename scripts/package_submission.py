"""Build a source-only paper supplement with per-file hashes; excludes all run data."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("submission/evousr-supplement.zip")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    included = [
        root / name
        for name in ("LICENSE", "pyproject.toml", "MANIFEST.in")
    ]
    allowed = {".py", ".json", ".md", ".txt"}
    for folder in ("src", "tests", "docs", "requirements", "examples", "scripts"):
        included.extend(
            p
            for p in (root / folder).rglob("*")
            if p.is_file()
            and p.suffix in allowed
            and "__pycache__" not in p.parts
            and not any(part.endswith(".egg-info") for part in p.parts)
        )
    included = sorted(included, key=lambda p: p.relative_to(root).as_posix())
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in included
    }
    manifest = (
        json.dumps(dict(schema_version=1, files=hashes), indent=2).encode() + b"\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in included:
            entry = zipfile.ZipInfo(
                "evousr/" + path.relative_to(root).as_posix(),
                date_time=(2026, 10, 26, 0, 0, 0),
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, path.read_bytes())
        entry = zipfile.ZipInfo(
            "evousr/FILES_SHA256.json", date_time=(2026, 10, 26, 0, 0, 0)
        )
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o644 << 16
        archive.writestr(entry, manifest)
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(
        checksum + "  " + args.output.name + "\n"
    )
    print(
        f"{args.output}: {len(included)} source files, {args.output.stat().st_size} bytes"
    )
    print("SHA256:", checksum)


if __name__ == "__main__":
    main()
