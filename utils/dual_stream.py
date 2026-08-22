"""Deterministic observation/mechanism training-stream composition."""


class DualStreamLoader:
    """Interleave exactly one mechanism pass into each B0 observation epoch."""

    def __init__(self, observation_loader, mechanism_loader):
        self.observation_loader = observation_loader
        self.mechanism_loader = mechanism_loader
        self.observation_steps = len(observation_loader)
        self.mechanism_steps = len(mechanism_loader)
        if self.observation_steps <= 0 or self.mechanism_steps <= 0:
            raise ValueError("dual-stream loaders must both be non-empty")
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
            batch_sampler = getattr(loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(int(epoch))
            sampler = getattr(loader, "sampler", None)
            if (sampler is not batch_sampler
                    and hasattr(sampler, "set_epoch")):
                sampler.set_epoch(int(epoch))

    def __iter__(self):
        mechanism_iterator = iter(self.mechanism_loader)
        emitted = 0
        for step, observation_batch in enumerate(self.observation_loader):
            mechanism_batch = None
            if self._scheduled(
                    step, self.observation_steps, self.mechanism_steps):
                try:
                    mechanism_batch = next(mechanism_iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "mechanism loader ended before its registered pass") from exc
                emitted += 1
            yield {
                "ct_stream_schema": "ct_seqtrack.dual_stream.v1",
                "observation": observation_batch,
                "mechanism": mechanism_batch,
            }
        if emitted != self.mechanism_steps:
            raise RuntimeError(
                "dual-stream schedule did not consume exactly one mechanism pass")
        try:
            next(mechanism_iterator)
        except StopIteration:
            return
        raise RuntimeError("mechanism loader contains unregistered extra batches")
