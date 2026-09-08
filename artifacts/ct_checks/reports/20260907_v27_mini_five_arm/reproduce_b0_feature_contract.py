"""Read-only reproduction of B0 frame packing and ignored temporal masks."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from models.attn.Models import Seq2SeqFormer


def main():
    torch.set_num_threads(1)
    batch, channels, frames, points = 1, 14, 4, 1024
    x = torch.arange(batch * channels * frames * points).reshape(
        batch, channels, frames * points)
    current = x.reshape(batch * frames, -1, points)
    expected = x.reshape(batch, channels, frames, points).permute(
        0, 2, 1, 3).reshape(batch * frames, channels, points)
    altered_xyz = x.clone()
    altered_xyz[:, :3, -points:] += 123
    altered_actual = altered_xyz.reshape(batch * frames, -1, points)
    altered_expected = altered_xyz.reshape(
        batch, channels, frames, points).permute(0, 2, 1, 3).reshape(
            batch * frames, channels, points)
    packing = {
        "source_shape": list(x.shape),
        "packed_shape": list(current.shape),
        "same_element_fraction": float((current == expected).float().mean()),
        "changing_only_current_xyz_actual_last_pseudo_frame_max_difference": int(
            (altered_actual[-1] - current[-1]).abs().max()),
        "changing_only_current_xyz_correct_last_frame_max_difference": int(
            (altered_expected[-1] - expected[-1]).abs().max()),
        "actual_channels_and_frames_per_pseudo_frame": [
            [[int(value // (frames * points)),
              int((value // points) % frames)] for value in row[:, 0]]
            for row in current
        ],
        "expected_channels_and_frames_per_frame": [
            [[int(value // (frames * points)),
              int((value // points) % frames)] for value in row[:, 0]]
            for row in expected
        ],
    }
    torch.manual_seed(42)
    model = Seq2SeqFormer(d_word_vec=64, d_model=64, d_inner=512,
                         n_layers=3, n_head=4, d_k=64, d_v=64).eval()
    source = torch.randn(1, 512, 128)
    corners = torch.randn(1, 32, 4)
    valid = torch.tensor([[1., 0., 0.]])
    changed = source.clone()
    changed[:, 128:384] = torch.randn_like(changed[:, 128:384]) * 8 + 3
    with torch.no_grad():
        output = model(corners, source, valid)
        all_valid = model(corners, source, torch.ones_like(valid))
        altered_invalid = model(corners, changed, valid)
    masking = {
        "uses_actual_production_transformer": True,
        "mode": "eval; random initialized synthetic input; no checkpoint gain claim",
        "history_valid_mask": valid.tolist(),
        "same_input_changed_mask_max_output_difference": float(
            (output-all_valid).abs().max()),
        "changed_only_invalid_history_features_current_box_max_difference": float(
            (output[:, -1]-altered_invalid[:, -1]).abs().max()),
        "current_box_original": output[:, -1].tolist(),
        "current_box_altered_invalid_history": altered_invalid[:, -1].tolist(),
    }
    result = {"packing": packing, "masking": masking}
    destination = Path(__file__).with_name("b0_feature_contract_reproduction.json")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
