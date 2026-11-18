"""Fit a residual covariance model from a prepared RGB detection-to-GT NPZ."""

import argparse
from pathlib import Path

import numpy as np

from nauttrack.core import ResidualCalibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--shrinkage", type=float, default=30.0)
    args = parser.parse_args()
    with np.load(args.input_npz) as data:
        required = {"residuals", "boxes", "scores"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"NPZ missing arrays: {sorted(missing)}")
        calibrator = ResidualCalibrator.fit(
            data["residuals"], data["boxes"], data["scores"], args.bins, args.shrinkage
        )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(output)
    print(f"Saved calibrator: {output}")


if __name__ == "__main__":
    main()
