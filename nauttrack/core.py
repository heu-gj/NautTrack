"""NautTrack core: geometry, uncertainty, Kalman filtering, and association."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    """Convert xyxy boxes to [center_x, center_y, aspect, height]."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64).reshape(4)
    width, height = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, width / height, height])


def xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    """Convert [center_x, center_y, aspect, height] to xyxy."""
    cx, cy, aspect, height = np.asarray(state, dtype=np.float64).reshape(4)
    height = max(float(height), 1e-3)
    width = max(float(aspect) * height, 1e-3)
    return np.asarray([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], dtype=np.float32)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xyxy boxes."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(br - tl, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)).astype(np.float32)


@dataclass
class Detection:
    """One RGB detector observation."""

    box: np.ndarray
    score: float
    cls: int
    index: int = -1

    def __post_init__(self):
        self.box = np.asarray(self.box, dtype=np.float32).reshape(4)
        self.score = float(self.score)
        self.cls = int(self.cls)


@dataclass
class Track:
    """Track state and lifecycle metadata."""

    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    cls: int
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "tracked"
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])


@dataclass
class TrackerConfig:
    """NautTrack hyperparameters."""

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    min_hits: int = 1
    match_thresh: float = 0.85
    second_match_thresh: float = 0.95
    max_mahalanobis: float = 9.4877
    iou_weight: float = 0.65
    motion_weight: float = 0.30
    confidence_weight: float = 0.05
    uncertainty_mode: str = "heuristic"
    calibration_path: str | None = None
    center_noise_rel: float = 0.05
    aspect_noise: float = 0.05
    height_noise_rel: float = 0.05
    scale_power: float = 0.35
    confidence_noise_weight: float = 1.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrackerConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


def _positive_covariance(covariance: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64).reshape(4, 4)
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


class ResidualCalibrator:
    """Scale/confidence-binned covariance of same-frame detector-to-GT residuals."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.scale_edges = np.asarray(payload["scale_edges"], dtype=np.float64)
        self.score_edges = np.asarray(payload["score_edges"], dtype=np.float64)
        self.global_covariance = _positive_covariance(payload["global_covariance"])
        self.cells = payload.get("cells", {})

    @classmethod
    def load(cls, path: str | Path) -> "ResidualCalibrator":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(json.load(stream))

    @classmethod
    def fit(cls, residuals, boxes, scores, bins: int = 3, shrinkage: float = 30.0):
        """Fit residual covariance cells.

        residuals columns are dx/det_w, dy/det_h, log(det_w/gt_w), log(det_h/gt_h).
        boxes are matched detection boxes in xyxy format.
        """
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1, 4)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if not (len(residuals) == len(boxes) == len(scores)) or len(residuals) < 2:
            raise ValueError("residuals, boxes, and scores must have equal length >= 2")
        if not (np.isfinite(residuals).all() and np.isfinite(boxes).all() and np.isfinite(scores).all()):
            raise ValueError("calibration arrays must be finite")
        bins = max(int(bins), 1)
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-3)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-3)
        scale = np.sqrt(widths * heights)
        scale_edges = np.unique(np.quantile(scale, np.linspace(0, 1, bins + 1)))
        score_edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)))
        if len(scale_edges) < 2:
            scale_edges = np.asarray([scale.min() - 1e-6, scale.max() + 1e-6])
        if len(score_edges) < 2:
            score_edges = np.asarray([scores.min() - 1e-6, scores.max() + 1e-6])
        global_cov = _positive_covariance(np.cov(residuals, rowvar=False))
        si = np.clip(np.searchsorted(scale_edges[1:-1], scale, side="right"), 0, len(scale_edges) - 2)
        ci = np.clip(np.searchsorted(score_edges[1:-1], scores, side="right"), 0, len(score_edges) - 2)
        cells = {}
        for i in range(len(scale_edges) - 1):
            for j in range(len(score_edges) - 1):
                sample = residuals[(si == i) & (ci == j)]
                count = len(sample)
                if count >= 2:
                    local = _positive_covariance(np.cov(sample, rowvar=False))
                    alpha = count / (count + max(float(shrinkage), 0.0))
                    cov = _positive_covariance(alpha * local + (1 - alpha) * global_cov)
                else:
                    cov = global_cov
                cells[f"{i},{j}"] = {"count": count, "covariance": cov.tolist()}
        return cls({
            "schema_version": 1,
            "residual_convention": "dx/det_w,dy/det_h,log(det_w/gt_w),log(det_h/gt_h)",
            "scale_edges": scale_edges.tolist(),
            "score_edges": score_edges.tolist(),
            "global_covariance": global_cov.tolist(),
            "cells": cells,
        })

    def covariance_for(self, box: np.ndarray, score: float) -> np.ndarray:
        box = np.asarray(box, dtype=np.float64).reshape(4)
        scale = np.sqrt(max(box[2] - box[0], 1e-3) * max(box[3] - box[1], 1e-3))
        i = int(np.clip(np.searchsorted(self.scale_edges[1:-1], scale, side="right"), 0, len(self.scale_edges) - 2))
        j = int(np.clip(np.searchsorted(self.score_edges[1:-1], score, side="right"), 0, len(self.score_edges) - 2))
        cell = self.cells.get(f"{i},{j}")
        if not cell or int(cell.get("count", 0)) < 2:
            return self.global_covariance.copy()
        return _positive_covariance(cell["covariance"])

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.payload, stream, indent=2)


class DetectionUncertainty:
    """Map detection scale/confidence to a measurement covariance in XYAH space."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.calibrator = None
        if config.uncertainty_mode == "calibrated":
            if not config.calibration_path:
                raise ValueError("calibrated uncertainty requires tracker.calibration_path")
            self.calibrator = ResidualCalibrator.load(config.calibration_path)

    def measurement_covariance(self, box, score: float, image_shape: tuple[int, int]) -> np.ndarray:
        x1, y1, x2, y2 = np.asarray(box, dtype=np.float64).reshape(4)
        width, height = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
        aspect = width / height
        if self.calibrator is not None:
            residual_cov = self.calibrator.covariance_for(box, score)
            jacobian = np.asarray([
                [width, 0, 0, 0],
                [0, height, 0, 0],
                [0, 0, aspect, -aspect],
                [0, 0, 0, height],
            ], dtype=np.float64)
            covariance = jacobian @ residual_cov @ jacobian.T
        else:
            image_h, image_w = image_shape
            relative_scale = np.sqrt(width * height / max(float(image_h * image_w), 1.0))
            scale = np.clip((0.02 / max(relative_scale, 1e-6)) ** self.config.scale_power, 0.75, 4.0)
            confidence = 1.0 + self.config.confidence_noise_weight * (1.0 - np.clip(score, 0, 1))
            factor = float(scale * confidence)
            sigma = np.asarray([
                self.config.center_noise_rel * width * factor,
                self.config.center_noise_rel * height * factor,
                self.config.aspect_noise * factor,
                self.config.height_noise_rel * height * factor,
            ])
            covariance = np.diag(sigma**2)
        return (covariance + covariance.T) * 0.5 + np.eye(4) * 1e-6


class KalmanFilterXYAH:
    """8-D constant-velocity filter over [cx, cy, aspect, height, vx, vy, va, vh]."""

    def __init__(self, position_weight: float = 0.05, velocity_weight: float = 0.00625):
        self.motion = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.motion[i, i + 4] = 1.0
        self.observation = np.eye(4, 8, dtype=np.float64)
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight

    def initiate(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(4)
        mean = np.r_[measurement, np.zeros(4)]
        h = max(float(measurement[3]), 1.0)
        std = np.asarray([0.1*h, 0.1*h, 0.1, 0.1*h, 0.0625*h, 0.0625*h, 0.01, 0.0625*h])
        return mean, np.diag(std**2)

    def predict(self, mean, covariance, lost: bool = False):
        mean = np.asarray(mean, dtype=np.float64).copy()
        if lost:
            mean[7] = 0.0
        h = max(float(mean[3]), 1.0)
        std = np.asarray([
            self.position_weight*h, self.position_weight*h, 0.05, self.position_weight*h,
            self.velocity_weight*h, self.velocity_weight*h, 0.005, self.velocity_weight*h,
        ])
        q = np.diag(std**2)
        mean = self.motion @ mean
        covariance = self.motion @ covariance @ self.motion.T + q
        return mean, (covariance + covariance.T) * 0.5

    def project(self, mean, covariance, measurement_covariance):
        z_mean = self.observation @ mean
        z_cov = self.observation @ covariance @ self.observation.T + measurement_covariance
        return z_mean, (z_cov + z_cov.T) * 0.5

    def gating_distance(self, mean, covariance, measurement, measurement_covariance):
        z_mean, z_cov = self.project(mean, covariance, measurement_covariance)
        residual = np.asarray(measurement, dtype=np.float64).reshape(4) - z_mean
        try:
            return float(residual.T @ np.linalg.solve(z_cov, residual))
        except np.linalg.LinAlgError:
            return float("inf")

    def update(self, mean, covariance, measurement, measurement_covariance):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(4)
        z_mean, z_cov = self.project(mean, covariance, measurement_covariance)
        cross = covariance @ self.observation.T
        try:
            gain = np.linalg.solve(z_cov, cross.T).T
        except np.linalg.LinAlgError:
            gain = cross @ np.linalg.pinv(z_cov)
        new_mean = mean + gain @ (measurement - z_mean)
        residual_map = np.eye(8) - gain @ self.observation
        new_cov = residual_map @ covariance @ residual_map.T + gain @ measurement_covariance @ gain.T
        return new_mean, (new_cov + new_cov.T) * 0.5


def associate(tracks, detections, kalman_filter, uncertainty, image_shape, config, threshold):
    """Hungarian association with IoU cost and Mahalanobis gating."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    track_boxes = np.asarray([track.box for track in tracks])
    det_boxes = np.asarray([det.box for det in detections])
    ious = iou_matrix(track_boxes, det_boxes)
    costs = np.full((len(tracks), len(detections)), 1e6, dtype=np.float64)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            if track.cls != det.cls:
                continue
            z = xyxy_to_xyah(det.box)
            r = uncertainty.measurement_covariance(det.box, det.score, image_shape)
            d2 = kalman_filter.gating_distance(track.mean, track.covariance, z, r)
            if not np.isfinite(d2) or d2 > config.max_mahalanobis:
                continue
            motion_cost = min(d2 / max(config.max_mahalanobis, 1e-6), 1.0)
            conf_cost = 1.0 - float(np.clip(det.score, 0, 1))
            costs[i, j] = (
                config.iou_weight * (1.0 - float(ious[i, j]))
                + config.motion_weight * motion_cost
                + config.confidence_weight * conf_cost
            )
    rows, cols = linear_sum_assignment(costs)
    matches, used_rows, used_cols = [], set(), set()
    for row, col in zip(rows.tolist(), cols.tolist()):
        if costs[row, col] <= threshold:
            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
    return (
        matches,
        [i for i in range(len(tracks)) if i not in used_rows],
        [j for j in range(len(detections)) if j not in used_cols],
    )


class NautTracker:
    """Two-stage ByteTrack-style tracker with adaptive measurement covariance."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        if self.config.track_low_thresh > self.config.track_high_thresh:
            raise ValueError("track_low_thresh must not exceed track_high_thresh")
        if self.config.uncertainty_mode not in {"heuristic", "calibrated"}:
            raise ValueError("uncertainty_mode must be heuristic or calibrated")
        self.kf = KalmanFilterXYAH()
        self.uncertainty = DetectionUncertainty(self.config)
        self.tracks: list[Track] = []
        self.frame_id = 0
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.frame_id = 0
        self.next_id = 1

    def _start(self, det: Detection) -> Track:
        mean, covariance = self.kf.initiate(xyxy_to_xyah(det.box))
        track = Track(self.next_id, mean, covariance, det.score, det.cls, history=[det.box.copy()])
        self.next_id += 1
        return track

    def _update(self, track: Track, det: Detection, image_shape):
        z = xyxy_to_xyah(det.box)
        r = self.uncertainty.measurement_covariance(det.box, det.score, image_shape)
        track.mean, track.covariance = self.kf.update(track.mean, track.covariance, z, r)
        track.score, track.cls = det.score, det.cls
        track.hits += 1
        track.time_since_update = 0
        track.state = "tracked"
        track.history.append(det.box.copy())

    def update(self, detections: list[Detection], image_shape: tuple[int, int]) -> list[Track]:
        self.frame_id += 1
        image_shape = (int(image_shape[0]), int(image_shape[1]))
        for track in self.tracks:
            if track.state in {"tracked", "lost"}:
                track.mean, track.covariance = self.kf.predict(
                    track.mean, track.covariance, lost=(track.state == "lost")
                )
                track.age += 1
                track.time_since_update += 1

        high = [d for d in detections if d.score >= self.config.track_high_thresh]
        low = [d for d in detections if self.config.track_low_thresh <= d.score < self.config.track_high_thresh]
        candidates = [t for t in self.tracks if t.state in {"tracked", "lost"}]
        matches, unmatched_tracks, unmatched_high = associate(
            candidates, high, self.kf, self.uncertainty, image_shape, self.config, self.config.match_thresh
        )
        for ti, di in matches:
            self._update(candidates[ti], high[di], image_shape)

        remaining_active = [candidates[i] for i in unmatched_tracks if candidates[i].state == "tracked"]
        matches_low, unmatched_active, _ = associate(
            remaining_active, low, self.kf, self.uncertainty, image_shape, self.config, self.config.second_match_thresh
        )
        for ti, di in matches_low:
            self._update(remaining_active[ti], low[di], image_shape)
        matched_active = {ti for ti, _ in matches_low}
        for ti, track in enumerate(remaining_active):
            if ti not in matched_active:
                track.state = "lost"

        for di in unmatched_high:
            if high[di].score >= self.config.new_track_thresh:
                self.tracks.append(self._start(high[di]))

        kept = []
        for track in self.tracks:
            if track.state == "lost" and track.time_since_update > self.config.track_buffer:
                track.state = "removed"
            if track.state != "removed":
                kept.append(track)
        self.tracks = kept
        return [
            t for t in self.tracks
            if t.state == "tracked" and t.time_since_update == 0 and t.hits >= self.config.min_hits
        ]
