import os
import subprocess
import sys
from pathlib import Path


def test_importing_opt_has_no_side_effects(tmp_path):
    """Importing hybrik.opt must not parse argv or create an exp directory."""
    repo = Path(__file__).resolve().parents[1]
    code = (
        "import hybrik.opt as opt\n"
        "assert 'opt' not in vars(opt)\n"
        "assert 'cfg' not in vars(opt)\n"
        "assert not __import__('pathlib').Path('exp').exists()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    assert result.returncode == 0, result.stderr
