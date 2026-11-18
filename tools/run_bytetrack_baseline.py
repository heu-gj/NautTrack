"""Run official Ultralytics ByteTrack with matched YOLO26 detector settings."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Official ByteTrack reference run.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--project", default="runs/bytetrack")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    YOLO(args.model).track(
        source=args.source, tracker="bytetrack.yaml", persist=False, save=True,
        project=args.project, name=args.name, exist_ok=True, device=args.device,
        imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=True,
    )


if __name__ == "__main__":
    main()
