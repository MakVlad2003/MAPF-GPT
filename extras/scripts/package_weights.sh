#!/usr/bin/env bash
# Pack canonical LightWeight checkpoints into a zip for colleagues.
# Run from MAPF-GPT repo root:  bash extras/scripts/package_weights.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

VERSION="${1:-v1}"
OUT_DIR="dist/lightweight-checkpoints-${VERSION}"
ZIP="dist/lightweight-checkpoints-${VERSION}.zip"
MANIFEST="extras/weights_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST" >&2
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 - <<'PY' "$MANIFEST" "$OUT_DIR"
import json, shutil, sys
from pathlib import Path

manifest_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(manifest_path.read_text())
missing = []
for item in data["checkpoints"]:
    src = Path(item["source"])
    dst = out_dir / item["pack_as"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        missing.append(str(src))
        continue
    shutil.copy2(src, dst)
    print(f"  + {item['id']:3}  {src}  ->  {dst}")

readme = out_dir / "README.txt"
readme.write_text(
    f"LightWeight MAPF-GPT checkpoints ({data.get('version', '?')})\n"
    f"See {data.get('report', 'extras/REPORT.md')} for metrics and usage.\n"
    f"Eval: python extras/scripts/eval_pogema.py --custom-weights <path> --custom-model-name <name>\n",
    encoding="utf-8",
)
shutil.copy2(manifest_path, out_dir / "weights_manifest.json")
if missing:
    print("\nMISSING (skipped):", *missing, sep="\n  ")
    sys.exit(1)
PY

mkdir -p dist
rm -f "$ZIP"
( cd dist && zip -qr "../$(basename "$ZIP")" "$(basename "$OUT_DIR")" )

echo ""
echo "Created: $ROOT/$ZIP"
du -sh "$ZIP"
