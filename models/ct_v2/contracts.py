"""Small CT-v2 data contracts shared by the model and unit tests."""

import torch


def resolve_observation_delta_t(
        input_dict, reference, use_ct_v2, default_time_step):
    """Return selected, real and effective query gaps as batch tensors."""
    batch_size = reference.shape[0]
    real_delta_t = input_dict.get(
        "current_delta_t_real", input_dict.get("current_delta_t"))
    if real_delta_t is None:
        real_delta_t = reference.new_full(
            (batch_size,), float(default_time_step))
    elif not torch.is_tensor(real_delta_t):
        real_delta_t = torch.as_tensor(
            real_delta_t, device=reference.device, dtype=reference.dtype)
    else:
        real_delta_t = real_delta_t.to(
            device=reference.device, dtype=reference.dtype)
    if real_delta_t.numel() == 1 and batch_size > 1:
        real_delta_t = real_delta_t.repeat(batch_size)
    real_delta_t = real_delta_t.reshape(batch_size)

    effective_delta_t = input_dict.get(
        "current_delta_t_effective", real_delta_t)
    if not torch.is_tensor(effective_delta_t):
        effective_delta_t = torch.as_tensor(
            effective_delta_t,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        effective_delta_t = effective_delta_t.to(
            device=reference.device, dtype=reference.dtype)
    if effective_delta_t.numel() == 1 and batch_size > 1:
        effective_delta_t = effective_delta_t.repeat(batch_size)
    effective_delta_t = effective_delta_t.reshape(batch_size)
    selected_delta_t = effective_delta_t if use_ct_v2 else real_delta_t
    return selected_delta_t, real_delta_t, effective_delta_t


def build_search_usable_mask(input_dict, obs_aux, reference):
    """Return a mask aligned with the point regularizer's >2 rule."""
    usable_search = input_dict.get("search_has_usable_points")
    if usable_search is None:
        usable_search = obs_aux["obs_num_points_search"] > 2
    if not torch.is_tensor(usable_search):
        usable_search = torch.as_tensor(
            usable_search, device=reference.device, dtype=reference.dtype)
    usable_search = usable_search.to(
        device=reference.device, dtype=reference.dtype)
    if usable_search.numel() == 1 and reference.numel() > 1:
        usable_search = usable_search.repeat(reference.numel())
    return (usable_search.reshape_as(reference) > 0).to(reference.dtype)
