# Upstream baseline and reproducibility

NautTrack uses the official Ultralytics implementation as its detector and reference-tracking dependency.

- Repository: https://github.com/ultralytics/ultralytics
- Pinned version: 8.4.215
- Pinned source commit: 3f5c9e6207dec8c4c2089300281d9808ce3037de
- Commit URL: https://github.com/ultralytics/ultralytics/commit/3f5c9e6207dec8c4c2089300281d9808ce3037de
- YOLO26 model documentation: https://docs.ultralytics.com/models/yolo26
- Tracking documentation: https://docs.ultralytics.com/modes/track

The upstream project is not copied wholesale into this repository. NautTrack calls the public Ultralytics Python API for YOLO26 inference and implements its own tracker in the nauttrack package. This keeps the upstream detector implementation version-pinned while isolating research changes.

The official ByteTrack implementation is the first tracking reference. Use tracker="bytetrack.yaml" explicitly: Ultralytics supports multiple trackers and its default may change between releases.

Model weights are not stored in Git. Ultralytics downloads yolo26n.pt on first use, or the user can pass a custom RGB-trained checkpoint.

To intentionally upgrade the baseline, change the pinned dependency and this document together, then rerun the complete unit and benchmark suite.
