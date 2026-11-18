# NautTrack

**NautTrack** is an RGB-only multi-object tracking research project built on the official Ultralytics YOLO26 detection stack.

The repository starts from the public Ultralytics implementation as an installed, commit-pinned dependency and develops the tracking method in a separate `nauttrack/` package. This keeps the detector baseline reproducible and avoids maintaining a private copy of the entire upstream framework.

## Current baseline

- Detector: Ultralytics YOLO26 (for example, `yolo26n.pt` or your own RGB-trained checkpoint).
- Tracking reference: official Ultralytics ByteTrack, selected explicitly with `tracker="bytetrack.yaml"` for comparisons.
- NautTrack prototype: two-stage association, XYAH constant-velocity Kalman filtering, scale/confidence-aware measurement covariance, and optional covariance calibration from same-frame detection-to-GT residuals.
- Modality: visible RGB images only.

The upstream dependency is pinned to Ultralytics `8.4.215`, source commit [`3f5c9e6207dec8c4c2089300281d9808ce3037de`](https://github.com/ultralytics/ultralytics/commit/3f5c9e6207dec8c4c2089300281d9808ce3037de). YOLO26 model weights are downloaded separately by Ultralytics when first used; large model weights are intentionally not committed to this repository.

## Installation

Create an environment with Python 3.10+ and install PyTorch for your CUDA/CPU setup first, following the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/heu-gj/NautTrack.git
cd NautTrack
pip install -r requirements.txt
pip install -e .
```

## Run NautTrack on an RGB video

```bash
python tools/track_video.py \
  --source path/to/video.mp4 \
  --model yolo26n.pt \
  --config configs/nauttrack.yaml \
  --output runs/nauttrack/video.mp4 \
  --device 0
```

Use `--device cpu` for CPU inference. A custom RGB detector checkpoint can be supplied with `--model path/to/best.pt`.

## Run the official ByteTrack reference

For a detector- and input-matched reference run, use the same YOLO26 weights and inference settings:

```bash
yolo track model=yolo26n.pt source=path/to/video.mp4 \
  tracker=bytetrack.yaml conf=0.1 iou=0.7 imgsz=1280
```

Ultralytics currently offers multiple tracking backends and may change its default tracker. NautTrack experiments should therefore specify `bytetrack.yaml` explicitly when reporting ByteTrack results.

## Calibrating measurement uncertainty

The optional calibration fitter expects an NPZ file with:

- `residuals`: shape `(N, 4)`, using `[dx / det_width, dy / det_height, log(det_width / gt_width), log(det_height / gt_height)]`;
- `boxes`: shape `(N, 4)`, matched RGB detection boxes in `xyxy` format;
- `scores`: shape `(N,)`, detector confidence scores.

Fit a scale/confidence-conditioned covariance model:

```bash
python tools/fit_calibrator.py \
  --input-npz path/to/rgb_detection_residuals.npz \
  --output-json configs/rgb_residual_calibrator.json
```

Then set `tracker.uncertainty_mode: calibrated` and `tracker.calibration_path` in `configs/nauttrack.yaml`. The calibration data must come from eligible training/validation sequences only. Keep official test sequences locked for final evaluation.

## Project layout

```text
nauttrack/       Tracker, Kalman filter, association, uncertainty model, YOLO26 adapter
configs/         RGB-only tracker configuration
tools/           Video runner and residual-calibration fitter
tests/           Unit tests for geometry, Kalman filtering, and track lifecycle
```

## Research status

This is an initial reproducible baseline, not a claim of benchmark superiority. The next steps are to validate the detector-to-GT residual collector on the actual dataset split, integrate empirically fitted uncertainty, and run controlled ablations against official ByteTrack.

## License and attribution

NautTrack is distributed under AGPL-3.0. The detector dependency is [Ultralytics](https://github.com/ultralytics/ultralytics), also distributed under AGPL-3.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [UPSTREAM.md](UPSTREAM.md).
