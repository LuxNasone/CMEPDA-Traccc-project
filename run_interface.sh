set -euo pipefail

pytest -q tests/
python src/python_interface/file_reader.py "$@"