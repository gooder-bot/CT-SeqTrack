"""Dependency-free full-gated CfC cell for the B1 temporal backend."""

import torch
from torch import nn


class LeCunTanh(nn.Module):
    """LeCun's scaled tanh used by the reference CfC implementation."""

    def forward(self, value):
        return 1.7159 * torch.tanh(0.666 * value)


class FullGatedCfCCell(nn.Module):
    """One full-gated closed-form continuous-time recurrent update.

    This implements the trainable CfC form from Hasani et al. (2022), not the
    direct-solution, no-gate, or mixed-memory variants. ``elapsed_time`` is an
    explicit input so callers retain ownership of padding and time units.
    """

    def __init__(self, input_size, hidden_size, backbone_units=105):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.backbone_units = int(backbone_units)
        if min(self.input_size, self.hidden_size, self.backbone_units) <= 0:
            raise ValueError("CfC dimensions must be positive")

        self.backbone = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.backbone_units),
            LeCunTanh(),
        )
        self.first_state = nn.Linear(self.backbone_units, self.hidden_size)
        self.second_state = nn.Linear(self.backbone_units, self.hidden_size)
        self.time_a = nn.Linear(self.backbone_units, self.hidden_size)
        self.time_b = nn.Linear(self.backbone_units, self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() == 2:
                nn.init.xavier_uniform_(parameter)

    def _elapsed_column(self, elapsed_time, batch_size, reference):
        if not torch.is_tensor(elapsed_time):
            elapsed_time = torch.as_tensor(
                elapsed_time, device=reference.device, dtype=reference.dtype
            )
        elapsed_time = elapsed_time.to(
            device=reference.device, dtype=reference.dtype
        )
        if elapsed_time.numel() == 1:
            elapsed_time = elapsed_time.reshape(1, 1).expand(batch_size, 1)
        elif elapsed_time.numel() == batch_size:
            elapsed_time = elapsed_time.reshape(batch_size, 1)
        else:
            raise ValueError("CfC elapsed_time must be scalar or one value per batch")
        if not bool(torch.isfinite(elapsed_time).all()):
            raise ValueError("CfC elapsed_time must be finite")
        if bool((elapsed_time < 0).any()):
            raise ValueError("CfC elapsed_time must be non-negative")
        return elapsed_time

    def forward(self, input_value, hidden_state, elapsed_time):
        if input_value.dim() != 2 or input_value.shape[1] != self.input_size:
            raise ValueError(
                f"CfC input must have shape [B,{self.input_size}]"
            )
        if hidden_state.dim() != 2 or hidden_state.shape != (
            input_value.shape[0],
            self.hidden_size,
        ):
            raise ValueError(
                f"CfC hidden state must have shape [B,{self.hidden_size}]"
            )
        elapsed = self._elapsed_column(
            elapsed_time, input_value.shape[0], input_value
        )
        shared = self.backbone(torch.cat((input_value, hidden_state), dim=1))
        first = torch.tanh(self.first_state(shared))
        second = torch.tanh(self.second_state(shared))
        gate = torch.sigmoid(self.time_a(shared) * elapsed + self.time_b(shared))
        return first * (1.0 - gate) + second * gate


__all__ = ["FullGatedCfCCell", "LeCunTanh"]
