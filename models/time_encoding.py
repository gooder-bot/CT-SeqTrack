import math

import torch
from torch import nn


class TimeEncoding(nn.Module):
    def __init__(
        self,
        mode="raw",
        scale=0.5,
        clip=4.0,
        fourier_bands=4,
        hidden_dim=16,
        output_scale=0.1,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.scale = float(scale) if scale is not None else 0.5
        self.clip = float(clip) if clip is not None else 4.0
        self.fourier_bands = int(fourier_bands)
        self.hidden_dim = int(hidden_dim)
        self.output_scale = float(output_scale) if output_scale is not None else 0.1

        if self.scale <= 0:
            raise ValueError("time encoding scale must be positive.")
        if self.clip <= 0:
            raise ValueError("time encoding clip must be positive.")
        if self.output_scale <= 0:
            raise ValueError("time encoding output_scale must be positive.")

        if self.mode in ("none", "identity"):
            self.mode = "raw"
        if self.mode in ("scaled", "raw_scaled", "normalized_raw"):
            self.mode = "scaled_raw"

        if self.mode in ("raw", "scaled_raw"):
            return

        if self.mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            )
            self._zero_biases()
            return

        if self.mode == "fourier":
            if self.fourier_bands <= 0:
                raise ValueError("time_fourier_bands must be positive for fourier mode.")
            frequencies = torch.pow(2.0, torch.arange(self.fourier_bands, dtype=torch.float32))
            self.register_buffer("frequencies", frequencies, persistent=False)
            self.proj = nn.Linear(self.fourier_bands * 2, 1)
            nn.init.zeros_(self.proj.bias)
            return

        raise ValueError(f"Unsupported time_encoding mode: {mode}")

    def _zero_biases(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _normalize(self, time_values):
        tau = torch.sign(time_values) * torch.log1p(torch.abs(time_values) / self.scale)
        return torch.clamp(tau, min=-self.clip, max=self.clip)

    def forward(self, time_values):
        if self.mode == "raw":
            return time_values
        if self.mode == "scaled_raw":
            return time_values * (self.output_scale / self.scale)

        tau = self._normalize(time_values)

        if self.mode == "mlp":
            return self.net(tau)

        frequencies = self.frequencies.to(device=tau.device, dtype=tau.dtype)
        phases = tau * frequencies.view(*([1] * (tau.dim() - 1)), -1) * math.pi
        features = torch.cat((torch.sin(phases), torch.cos(phases) - 1.0), dim=-1)
        return self.proj(features)
