set -euo pipefail

pytest -q tests/
python src/digitization/toy_digitize.py "$@"