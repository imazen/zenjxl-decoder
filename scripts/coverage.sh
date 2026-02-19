#!/bin/bash
# Code coverage analysis script for jxl-rs
# Usage: ./scripts/coverage.sh [--html] [--open]

set -e

cd "$(dirname "$0")/.."

HTML=false
OPEN=false

for arg in "$@"; do
    case $arg in
        --html) HTML=true ;;
        --open) OPEN=true; HTML=true ;;
        --help|-h)
            echo "Usage: $0 [--html] [--open]"
            echo "  --html  Generate HTML report in coverage_html/"
            echo "  --open  Generate HTML and open in browser"
            exit 0
            ;;
    esac
done

echo "=== Running tests with coverage instrumentation ==="
cargo llvm-cov --no-report --release

echo ""
echo "=== Generating coverage report ==="
cargo llvm-cov report --json --release --output-path=coverage.json

echo ""
echo "=== Coverage Summary ==="
cargo llvm-cov report --release 2>&1 | grep "^TOTAL"

echo ""
echo "=== Per-file Coverage (worst first) ==="
python3 ci/coverage_summary.py 2>/dev/null | head -1
python3 -c "
import json
with open('coverage.json') as f:
    data = json.load(f)

files = []
for entry in data['data'][0]['files']:
    path = entry['filename']
    if '/src/' in path:
        path = path.split('/src/')[1]
    ln = entry['summary']['lines']
    pct = (ln['covered'] / ln['count'] * 100) if ln['count'] > 0 else 100
    files.append((pct, ln['covered'], ln['count'], path))

files.sort()
for pct, cov, total, path in files[:20]:
    mark = '🔴' if pct < 75 else '🟢' if pct >= 90 else '🟡'
    print(f'| {path:50} | {cov:4}/{total:4} ({pct:5.1f}%) {mark} |')
"

if $HTML; then
    echo ""
    echo "=== Generating HTML report ==="
    cargo llvm-cov report --html --release --output-dir=coverage_html
    echo "HTML report: coverage_html/html/index.html"

    if $OPEN; then
        if command -v xdg-open &> /dev/null; then
            xdg-open coverage_html/html/index.html
        elif command -v open &> /dev/null; then
            open coverage_html/html/index.html
        else
            echo "Cannot auto-open browser. Open coverage_html/html/index.html manually."
        fi
    fi
fi

echo ""
echo "=== Done ==="
