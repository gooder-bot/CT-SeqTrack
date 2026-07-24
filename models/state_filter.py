"""Fixed continuous-discrete state filtering and trajectory-tube utilities.

This is the deliberately conservative M4-0/M4-1 implementation: a persistent
constant-velocity/constant-yaw-rate state with fixed positive covariances.  It
does not learn Q/R and never reads current-frame ground truth after tracker
initialization.
"""

import copy
import math

import numpy as np
from pyquaternion import Quaternion


def wrap_angle(angle):
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def box_yaw(box):
    return float(box.orientation.radians * box.orientation.axis[-1])


def _symmetric_psd(matrix, jitter=1e-9):
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, float(jitter))
    return (eigenvectors * eigenvalues) @ eigenvectors.T


class FixedContinuousDiscreteFilter:
    """Persistent ``[p, v, yaw, yaw_rate]`` filter with fixed Q/R."""

    state_dim = 8
    measurement_dim = 4

    def __init__(
            self,
            acceleration_variance=2.0,
            yaw_acceleration_variance=0.5,
            measurement_position_variance=0.25,
            measurement_yaw_variance=0.09,
            initial_position_variance=0.25,
            initial_velocity_variance=4.0,
            initial_yaw_variance=0.09,
            initial_yaw_rate_variance=1.0,
            mahalanobis_gate=0.0,
            max_delta_t=5.0,
            covariance_jitter=1e-8):
        self.acceleration_variance = float(acceleration_variance)
        self.yaw_acceleration_variance = float(yaw_acceleration_variance)
        self.measurement_position_variance = float(
            measurement_position_variance)
        self.measurement_yaw_variance = float(measurement_yaw_variance)
        self.initial_variances = np.asarray([
            initial_position_variance,
            initial_position_variance,
            initial_position_variance,
            initial_velocity_variance,
            initial_velocity_variance,
            initial_velocity_variance,
            initial_yaw_variance,
            initial_yaw_rate_variance,
        ], dtype=np.float64)
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.max_delta_t = float(max_delta_t)
        self.covariance_jitter = float(covariance_jitter)
        if np.any(self.initial_variances <= 0):
            raise ValueError("initial filter variances must be positive")
        if min(
                self.acceleration_variance,
                self.yaw_acceleration_variance,
                self.measurement_position_variance,
                self.measurement_yaw_variance,
                self.max_delta_t,
                self.covariance_jitter) <= 0:
            raise ValueError("filter variances, max_delta_t and jitter must be positive")
        self.reset()

    def reset(self):
        self.mean = None
        self.covariance = None
        self.timestamp = None
        self.last_posterior_mean = None
        self.last_prediction = None

    @property
    def initialized(self):
        return self.mean is not None

    def initialize(self, position, yaw, timestamp, velocity=None, yaw_rate=0.0):
        position = np.asarray(position, dtype=np.float64).reshape(3)
        velocity = (
            np.zeros(3, dtype=np.float64)
            if velocity is None else np.asarray(velocity, dtype=np.float64).reshape(3)
        )
        values = np.concatenate(
            (position, velocity, [wrap_angle(yaw), float(yaw_rate)]))
        if not np.all(np.isfinite(values)) or not np.isfinite(timestamp):
            raise ValueError("cannot initialize M4 filter from non-finite state")
        self.mean = values
        self.covariance = np.diag(self.initial_variances)
        self.timestamp = float(timestamp)
        self.last_posterior_mean = self.mean.copy()
        self.last_prediction = None
        return self.snapshot(valid=True, delta_t=0.0)

    def _transition(self, delta_t):
        delta_t = float(delta_t)
        transition = np.eye(self.state_dim, dtype=np.float64)
        for axis in range(3):
            transition[axis, axis + 3] = delta_t
        transition[6, 7] = delta_t
        return transition

    def _process_covariance(self, delta_t):
        delta_t = float(delta_t)
        covariance = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        dt2 = delta_t ** 2
        dt3 = delta_t ** 3
        dt4 = delta_t ** 4
        for position_index, velocity_index in ((0, 3), (1, 4), (2, 5)):
            variance = self.acceleration_variance
            covariance[position_index, position_index] = variance * dt4 / 4.0
            covariance[position_index, velocity_index] = variance * dt3 / 2.0
            covariance[velocity_index, position_index] = variance * dt3 / 2.0
            covariance[velocity_index, velocity_index] = variance * dt2
        variance = self.yaw_acceleration_variance
        covariance[6, 6] = variance * dt4 / 4.0
        covariance[6, 7] = variance * dt3 / 2.0
        covariance[7, 6] = variance * dt3 / 2.0
        covariance[7, 7] = variance * dt2
        covariance += np.eye(self.state_dim) * self.covariance_jitter
        return covariance

    def predict(self, timestamp):
        if not self.initialized:
            raise RuntimeError("M4 filter must be initialized before predict")
        timestamp = float(timestamp)
        delta_t = timestamp - self.timestamp
        if (not np.isfinite(delta_t)) or delta_t <= 0 or delta_t > self.max_delta_t:
            self.last_prediction = self.snapshot(valid=False, delta_t=delta_t)
            return self.last_prediction

        self.last_posterior_mean = self.mean.copy()
        transition = self._transition(delta_t)
        self.mean = transition @ self.mean
        self.mean[6] = wrap_angle(self.mean[6])
        self.covariance = _symmetric_psd(
            transition @ self.covariance @ transition.T
            + self._process_covariance(delta_t),
            self.covariance_jitter,
        )
        self.timestamp = timestamp
        self.last_prediction = self.snapshot(valid=True, delta_t=delta_t)
        return self.last_prediction

    def _measurement_matrices(self):
        observation = np.zeros(
            (self.measurement_dim, self.state_dim), dtype=np.float64)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 2] = 1.0
        observation[3, 6] = 1.0
        measurement_covariance = np.diag([
            self.measurement_position_variance,
            self.measurement_position_variance,
            self.measurement_position_variance,
            self.measurement_yaw_variance,
        ])
        return observation, measurement_covariance

    def update(self, position, yaw):
        if not self.initialized:
            raise RuntimeError("M4 filter must be initialized before update")
        measurement = np.concatenate((
            np.asarray(position, dtype=np.float64).reshape(3),
            [wrap_angle(yaw)],
        ))
        if not np.all(np.isfinite(measurement)):
            return {"accepted": False, "reason": "nonfinite_measurement"}

        observation, measurement_covariance = self._measurement_matrices()
        innovation = measurement - observation @ self.mean
        innovation[3] = wrap_angle(innovation[3])
        innovation_covariance = _symmetric_psd(
            observation @ self.covariance @ observation.T
            + measurement_covariance,
            self.covariance_jitter,
        )
        solved_innovation = np.linalg.solve(
            innovation_covariance, innovation)
        mahalanobis = float(innovation @ solved_innovation)
        if self.mahalanobis_gate > 0 and mahalanobis > self.mahalanobis_gate:
            return {
                "accepted": False,
                "reason": "mahalanobis_gate",
                "mahalanobis": mahalanobis,
                "innovation": innovation.copy(),
            }

        gain = np.linalg.solve(
            innovation_covariance,
            observation @ self.covariance,
        ).T
        self.mean = self.mean + gain @ innovation
        self.mean[6] = wrap_angle(self.mean[6])
        identity = np.eye(self.state_dim, dtype=np.float64)
        residual_map = identity - gain @ observation
        self.covariance = _symmetric_psd(
            residual_map @ self.covariance @ residual_map.T
            + gain @ measurement_covariance @ gain.T,
            self.covariance_jitter,
        )
        return {
            "accepted": True,
            "reason": "updated",
            "mahalanobis": mahalanobis,
            "innovation": innovation.copy(),
        }

    def observe_direct(self, position, yaw, timestamp, velocity_momentum=0.5):
        """Update a tube-only state directly from the raw tracker output."""
        position = np.asarray(position, dtype=np.float64).reshape(3)
        yaw = wrap_angle(yaw)
        timestamp = float(timestamp)
        if not self.initialized:
            return self.initialize(position, yaw, timestamp)

        delta_t = timestamp - self.timestamp
        reference_mean = self.mean
        if (abs(delta_t) <= 1e-9
                and self.last_prediction is not None
                and self.last_prediction.get("valid", False)
                and self.last_posterior_mean is not None):
            # ``predict`` has already advanced the state to this timestamp.
            # Estimate the observed velocity from the preceding posterior,
            # rather than treating the zero residual time as a reset.
            delta_t = float(self.last_prediction["delta_t"])
            reference_mean = self.last_posterior_mean
        if (not np.isfinite(delta_t)) or delta_t <= 0 or delta_t > self.max_delta_t:
            return self.initialize(position, yaw, timestamp)
        momentum = float(velocity_momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("velocity_momentum must be in [0, 1)")
        observed_velocity = (position - reference_mean[:3]) / delta_t
        observed_yaw_rate = wrap_angle(yaw - reference_mean[6]) / delta_t
        velocity = (
            momentum * reference_mean[3:6]
            + (1.0 - momentum) * observed_velocity)
        yaw_rate = (
            momentum * reference_mean[7]
            + (1.0 - momentum) * observed_yaw_rate)
        return self.initialize(
            position, yaw, timestamp, velocity=velocity, yaw_rate=yaw_rate)

    def snapshot(self, valid=True, delta_t=0.0):
        return {
            "valid": bool(valid),
            "delta_t": float(delta_t),
            "mean": None if self.mean is None else self.mean.copy(),
            "covariance": (
                None if self.covariance is None else self.covariance.copy()),
            "start_mean": (
                None if self.last_posterior_mean is None
                else self.last_posterior_mean.copy()),
            "timestamp": self.timestamp,
        }

    def box_from_state(self, template_box):
        if not self.initialized:
            raise RuntimeError("M4 filter has no state")
        result = copy.deepcopy(template_box)
        result.center = self.mean[:3].copy()
        result.orientation = Quaternion(
            axis=[0, 0, 1], radians=wrap_angle(self.mean[6]))
        result.velocity = self.mean[3:6].copy()
        return result


def build_trajectory_tube_box(
        template_box,
        prediction,
        base_length=4.0,
        base_width=2.0,
        sigma_parallel_scale=2.0,
        sigma_perpendicular_scale=2.0,
        max_length=20.0,
        max_width=8.0,
        min_speed=0.2):
    """Construct a bounded tube between the last posterior and current prior."""
    if not prediction or not prediction.get("valid", False):
        return None
    positive_values = (
        base_length, base_width, max_length, max_width)
    if any((not np.isfinite(value)) or float(value) <= 0
           for value in positive_values):
        raise ValueError("tube base/max dimensions must be finite and positive")
    if float(max_length) < float(base_length):
        raise ValueError("tube max_length must be at least base_length")
    if float(max_width) < float(base_width):
        raise ValueError("tube max_width must be at least base_width")
    if min(
            float(sigma_parallel_scale),
            float(sigma_perpendicular_scale),
            float(min_speed)) < 0:
        raise ValueError("tube sigma scales and min_speed must be non-negative")
    mean = np.asarray(prediction["mean"], dtype=np.float64)
    covariance = np.asarray(prediction["covariance"], dtype=np.float64)
    start_mean = np.asarray(prediction["start_mean"], dtype=np.float64)
    start = start_mean[:3]
    end = mean[:3]
    planar_displacement = end[:2] - start[:2]
    displacement_norm = float(np.linalg.norm(planar_displacement))
    speed = float(np.linalg.norm(mean[3:5]))

    if displacement_norm >= float(min_speed) * max(
            float(prediction.get("delta_t", 0.0)), 1e-6):
        direction = planar_displacement / max(displacement_norm, 1e-9)
        yaw = math.atan2(direction[1], direction[0])
    elif speed >= float(min_speed):
        direction = mean[3:5] / speed
        yaw = math.atan2(direction[1], direction[0])
    else:
        yaw = wrap_angle(mean[6])
        direction = np.asarray([math.cos(yaw), math.sin(yaw)])
    perpendicular = np.asarray([-direction[1], direction[0]])

    covariance_xy = _symmetric_psd(covariance[:2, :2])
    sigma_parallel = math.sqrt(max(
        float(direction @ covariance_xy @ direction), 0.0))
    sigma_perpendicular = math.sqrt(max(
        float(perpendicular @ covariance_xy @ perpendicular), 0.0))
    length = min(
        float(max_length),
        max(float(base_length),
            float(base_length) + displacement_norm
            + float(sigma_parallel_scale) * sigma_parallel),
    )
    width = min(
        float(max_width),
        max(float(base_width),
            float(base_width)
            + float(sigma_perpendicular_scale) * sigma_perpendicular),
    )

    tube = copy.deepcopy(template_box)
    tube.center = 0.5 * (start + end)
    tube.orientation = Quaternion(axis=[0, 0, 1], radians=wrap_angle(yaw))
    tube.wlh = np.asarray(tube.wlh, dtype=np.float64).copy()
    tube.wlh[0] = width
    tube.wlh[1] = length
    return tube


def union_point_clouds(primary, secondary, tolerance=1e-6):
    """Union two point clouds already expressed in the same coordinate frame."""
    if secondary is None or secondary.nbr_points() == 0:
        return primary
    if primary.nbr_points() == 0:
        return secondary
    tolerance = float(tolerance)
    if tolerance <= 0:
        raise ValueError("point-union tolerance must be positive")
    points = np.concatenate((primary.points, secondary.points), axis=1)
    quantized = np.rint(points.T / tolerance).astype(np.int64)
    _, unique_indices = np.unique(quantized, axis=0, return_index=True)
    points = points[:, np.sort(unique_indices)]
    return primary.__class__(points)


def point_inside_oriented_crop(box, point, scale=1.0, offset=0.0):
    """Geometry primitive used by evaluation-only crop diagnostics."""
    if float(scale) <= 0 or float(offset) < 0:
        raise ValueError("crop scale must be positive and offset non-negative")
    point = np.asarray(point, dtype=np.float64).reshape(3)
    local_point = box.rotation_matrix.T @ (
        point - np.asarray(box.center, dtype=np.float64))
    half_extent = (
        np.asarray(box.wlh, dtype=np.float64) * float(scale) / 2.0
        + float(offset)
    )
    return bool(np.all(np.abs(local_point) < half_extent))
