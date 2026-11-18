import numpy as np

from nauttrack.core import (
    Detection, DetectionUncertainty, KalmanFilterXYAH, NautTracker,
    ResidualCalibrator, TrackerConfig, iou_matrix, xyah_to_xyxy, xyxy_to_xyah,
)


def test_box_conversion_and_iou():
    box = np.asarray([10.0, 20.0, 50.0, 80.0])
    np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box, atol=1e-5)
    assert np.isclose(iou_matrix(box[None], box[None])[0, 0], 1.0)


def test_kalman_covariance_is_symmetric_psd():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.asarray([100, 80, 1.2, 40]))
    mean, cov = kf.predict(mean, cov)
    mean, cov = kf.update(mean, cov, np.asarray([102, 81, 1.18, 41]), np.diag([4, 4, 0.02, 4]))
    np.testing.assert_allclose(cov, cov.T, atol=1e-8)
    assert np.linalg.eigvalsh(cov).min() > -1e-8


def test_heuristic_measurement_covariance_is_positive():
    model = DetectionUncertainty(TrackerConfig())
    cov = model.measurement_covariance([20, 30, 40, 55], 0.8, (720, 1280))
    assert np.linalg.eigvalsh(cov).min() > 0


def test_calibrator_fit_and_lookup():
    rng = np.random.default_rng(7)
    residuals = rng.normal(0, 0.02, size=(120, 4))
    boxes = np.tile(np.asarray([[10, 10, 30, 30]], dtype=float), (120, 1))
    boxes[60:, 2:] += 20
    scores = rng.uniform(0.2, 0.99, size=120)
    model = ResidualCalibrator.fit(residuals, boxes, scores, bins=2)
    assert np.linalg.eigvalsh(model.covariance_for(boxes[0], float(scores[0]))).min() > 0


def test_track_id_persists_and_low_score_recovers_track():
    cfg = TrackerConfig(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.5)
    tracker = NautTracker(cfg)
    first = tracker.update([Detection([10, 10, 30, 40], 0.9, 0)], (100, 100))
    second = tracker.update([Detection([11, 10, 31, 40], 0.3, 0)], (100, 100))
    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_lost_track_expires_after_buffer():
    tracker = NautTracker(TrackerConfig(track_buffer=1))
    tracker.update([Detection([10, 10, 30, 40], 0.9, 0)], (100, 100))
    tracker.update([], (100, 100))
    tracker.update([], (100, 100))
    tracker.update([], (100, 100))
    assert tracker.tracks == []
