"""Single-responsibility B0--B3 components exposed by CTSEQTRACK."""

from torch import nn

from models.ct_v2.pipeline_contracts import ObservationOutput, B1Input, AcquisitionRecord
from models.ct_v2.motion import OrderedPhysicalMotionEncoder
from models.ct_v2.evidence_memory import (
    B2EvidenceAcquirer,
    B3SelectiveUpdater,
)


class B0Observation(nn.Module):
    """Detach the observation contract consumed by B1--B3."""

    def forward(
            self, box, statistics, current_features=None,
            history_features=None):
        return ObservationOutput(
            box=box,
            statistics=statistics,
            current_features=current_features,
            history_features=history_features,
        ).detached()


class B1PhysicalTimePrior(OrderedPhysicalMotionEncoder):
    """Paper-facing name for the physical-time prior implementation."""

    def forward_input(self, inputs):
        if not isinstance(inputs, B1Input):
            raise TypeError('B1 forward_input requires the shared B1Input contract')
        return self(**inputs.encoder_kwargs())


__all__ = [
    "B0Observation", "B1PhysicalTimePrior", "B2EvidenceAcquirer",
    "B3SelectiveUpdater",
    "B1Input", "AcquisitionRecord",
]

