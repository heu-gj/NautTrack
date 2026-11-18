"""NautTrack: RGB-only multi-object tracking on top of YOLO26."""

__version__ = "0.1.0"

from .core import Detection, NautTracker, TrackerConfig

__all__ = ["Detection", "NautTracker", "TrackerConfig", "__version__"]
