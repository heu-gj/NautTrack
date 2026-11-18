"""YOLO26 RGB inference and NautTrack video runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from .core import Detection, NautTracker, TrackerConfig


class YOLO26Detector:
    """Adapter over the public Ultralytics YOLO prediction API."""

    def __init__(self, weights, conf, iou, imgsz, max_det, device):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("Install dependencies with pip install -r requirements.txt") from error
        self.model = YOLO(weights)
        self.kwargs = {
            "conf": float(conf), "iou": float(iou), "imgsz": int(imgsz),
            "max_det": int(max_det), "device": device, "verbose": False,
        }

    def predict(self, frame) -> list[Detection]:
        results = self.model.predict(source=frame, **self.kwargs)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        return [Detection(xyxy[i], float(scores[i]), int(classes[i]), i) for i in range(len(xyxy))]


def run_video(source, detector, tracker, output=None, show=False, save_txt=False):
    source_value = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(output) if output else None
    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

    txt_stream = None
    txt_path = None
    if save_txt:
        base = output_path.with_suffix("") if output_path else Path("runs/nauttrack/result")
        txt_path = base.parent / f"{base.name}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_stream = txt_path.open("w", encoding="utf-8")

    frame_id = observations = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            tracks = tracker.update(detector.predict(frame), frame.shape[:2])
            observations += len(tracks)
            rendered = frame.copy()
            for track in tracks:
                x1, y1, x2, y2 = track.box.astype(int).tolist()
                cv2.rectangle(rendered, (x1, y1), (x2, y2), (40, 220, 80), 2)
                cv2.putText(rendered, f"ID {track.track_id} {track.score:.2f}",
                            (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (40, 220, 80), 1, cv2.LINE_AA)
                if txt_stream:
                    txt_stream.write(
                        f"{frame_id},{track.track_id},{x1},{y1},{x2-x1},{y2-y1},{track.score:.6f},{track.cls}\n"
                    )
            if writer:
                writer.write(rendered)
            if show:
                cv2.imshow("NautTrack - YOLO26 RGB", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if txt_stream:
            txt_stream.close()
        if show:
            cv2.destroyAllWindows()
    return {
        "frames": frame_id,
        "track_observations": observations,
        "output_video": str(output_path) if output_path else None,
        "output_txt": str(txt_path) if txt_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Run NautTrack with a YOLO26 RGB detector.")
    parser.add_argument("--source", required=True, help="RGB video path or webcam index.")
    parser.add_argument("--model", default=None, help="YOLO26 weights or custom RGB detector checkpoint.")
    parser.add_argument("--config", default="configs/nauttrack.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    det_cfg, track_cfg = cfg.get("detector", {}), cfg.get("tracker", {})
    weights = args.model or det_cfg.get("model", "yolo26n.pt")
    device = args.device if args.device is not None else str(det_cfg.get("device", "0"))
    imgsz = args.imgsz if args.imgsz is not None else int(det_cfg.get("imgsz", 1280))
    conf = args.conf if args.conf is not None else float(det_cfg.get("conf", 0.1))
    iou = args.iou if args.iou is not None else float(det_cfg.get("iou", 0.7))
    if track_cfg.get("calibration_path") and not Path(track_cfg["calibration_path"]).is_absolute():
        candidate = Path(args.config).resolve().parent / track_cfg["calibration_path"]
        if candidate.exists():
            track_cfg["calibration_path"] = str(candidate)

    detector = YOLO26Detector(weights, conf, iou, imgsz, int(det_cfg.get("max_det", 300)), device)
    tracker = NautTracker(TrackerConfig.from_dict(track_cfg))
    print(run_video(args.source, detector, tracker, args.output, args.show, args.save_txt))


if __name__ == "__main__":
    main()
