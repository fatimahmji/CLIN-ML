"""One command: test, train, audit, verify frozen reference, then write report."""
import argparse
import subprocess
import sys
from pathlib import Path
from training import main as train
from audits import run_audits
from verify import verify_release
from make_report import make_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to the original Moorfields CSV"
    )
    parser.add_argument(
        "--output", default=str(Path(__file__).parent / "generated"), help="Output directory"
    )
    parser.add_argument(
        "--reference",
        default=str(Path(__file__).parent / "reference"),
        help="Frozen V1 artifacts",
    )
    args = parser.parse_args()
    reference = Path(args.reference)
    required = ["eye_level_dataset.csv", "heldout_predictions.csv"]
    if not all((reference / name).is_file() for name in required):
        parser.error("Frozen reference files are missing. See README.md and pass --reference /path/to/reference.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(Path(__file__).parent),
            "-p",
            "test_labels.py",
        ],
        check=True,
    )
    train(args.input, args.output)
    run_audits(args.input, args.output)
    verify_release(args.output, args.reference, args.input)
    make_report(args.output)
    print(
        "Research release complete; see performance_report.md. Clinical signoff remains pending."
    )
