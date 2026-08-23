"""Deterministic observation/mechanism training-stream composition."""

from utils.training_isolation import (
    capture_global_rng_state,
    restore_global_rng_state,
)


class DualStreamLoader:
    """Interleave exactly one mechanism pass into each B0 observation epoch."""

    def __init__(self, observation_loader, mechanism_loader=None, *,
                 schema="ct_seqtrack.dual_stream.v1",
                 isolate_mechanism_rng=False):
        self.observation_loader = observation_loader
        self.mechanism_loader = mechanism_loader
        self.schema = str(schema)
        self.isolate_mechanism_rng = bool(isolate_mechanism_rng)
        self.observation_steps = len(observation_loader)
        self.mechanism_steps = (
            len(mechanism_loader) if mechanism_loader is not None else 0)
        if self.observation_steps <= 0:
            raise ValueError("observation loader must be non-empty")
        if mechanism_loader is not None and self.mechanism_steps <= 0:
            raise ValueError("mechanism loader must be non-empty")
        if self.mechanism_steps > self.observation_steps:
            raise ValueError(
                "mechanism stream may not exceed the observation stream")

    def __len__(self):
        return self.observation_steps

    @staticmethod
    def _scheduled(step, observation_steps, mechanism_steps):
        before = step * mechanism_steps // observation_steps
        after = (step + 1) * mechanism_steps // observation_steps
        return after > before

    def set_epoch(self, epoch):
        for loader in (self.observation_loader, self.mechanism_loader):
            if loader is None:
                continue
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(int(epoch))
            batch_sampler = getattr(loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(int(epoch))
            sampler = getattr(loader, "sampler", None)
            if (sampler is not batch_sampler
                    and hasattr(sampler, "set_epoch")):
                sampler.set_epoch(int(epoch))

    def __iter__(self):
        mechanism_iterator = None
        emitted = 0
        for step, observation_batch in enumerate(self.observation_loader):
            mechanism_batch = None
            if self._scheduled(
                    step, self.observation_steps, self.mechanism_steps):
                if mechanism_iterator is None:
                    rng_state = (
                        capture_global_rng_state()
                        if self.isolate_mechanism_rng else None)
                    try:
                        mechanism_iterator = iter(self.mechanism_loader)
                    finally:
                        if rng_state is not None:
                            restore_global_rng_state(rng_state)
                try:
                    mechanism_batch = next(mechanism_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "mechanism loader ended before its registered pass") from exc
                emitted += 1
            yield {
                "ct_stream_schema": self.schema,
                "observation": observation_batch,
                "mechanism": mechanism_batch,
            }
        if emitted != self.mechanism_steps:
            raise RuntimeError(
                "dual-stream schedule did not consume exactly one mechanism pass")
        if mechanism_iterator is not None:
            try:
                next(mechanism_iterator)
            except StopIteration:
                return
            raise RuntimeError(
                "mechanism loader contains unregistered extra batches")
