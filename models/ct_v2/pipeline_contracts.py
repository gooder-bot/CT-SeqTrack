"""Typed contracts for the paper-facing CT-SeqTrack B0--B3 pipeline.

The tracker still exposes a flat dictionary to the legacy evaluator.  These
containers define ownership inside the model and make it explicit which
values may cross a module boundary.  Tensor validation is intentionally
lightweight so the contracts can also be used by unit tests and exporters.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ObservationOutput:
    box: Tensor
    statistics: Tensor
    current_features: Optional[Tensor] = None
    history_features: Optional[Tensor] = None

    def detached(self):
        return ObservationOutput(
            box=self.box.detach(),
            statistics=self.statistics.detach(),
            current_features=(None if self.current_features is None else
                              self.current_features.detach()),
            history_features=(None if self.history_features is None else
                              self.history_features.detach()),
        )


@dataclass(frozen=True)
class MotionPriorOutput:
    center_xy: Tensor
    direction_xy: Tensor
    log_sigma: Tensor
    valid: Tensor
    source: Tensor

    def detached(self):
        return MotionPriorOutput(*(
            value.detach() for value in (
                self.center_xy, self.direction_xy, self.log_sigma,
                self.valid, self.source)))


@dataclass(frozen=True)
class EvidenceOutput:
    raw_box: Tensor
    structural_available: Tensor
    presence_logit: Tensor
    targetness: Tensor
    evidence_summary: Tensor
    point_diagnostics: Dict[str, Tensor]


@dataclass(frozen=True)
class DecisionOutput:
    final_box: Tensor
    help_logit: Tensor
    harm_logit: Tensor
    expected_center_gain: Tensor
    expected_iou_gain: Tensor
    applied: Tensor
    bounded_residual: Tensor

