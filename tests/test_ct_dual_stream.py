from utils.dual_stream import DualStreamLoader


class _EpochAwareLoader:
    def __init__(self, values):
        self.values = list(values)
        self.batch_sampler = self
        self.epochs = []

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def test_dual_stream_consumes_one_evenly_spaced_mechanism_pass():
    observation = _EpochAwareLoader(range(10))
    mechanism = _EpochAwareLoader(["m0", "m1", "m2"])
    loader = DualStreamLoader(observation, mechanism)

    rows = list(loader)

    assert len(loader) == 10
    assert [row["observation"] for row in rows] == list(range(10))
    assert [index for index, row in enumerate(rows)
            if row["mechanism"] is not None] == [3, 6, 9]
    assert [row["mechanism"] for row in rows
            if row["mechanism"] is not None] == ["m0", "m1", "m2"]
    assert all(row["ct_stream_schema"] == "ct_seqtrack.dual_stream.v1"
               for row in rows)


def test_dual_stream_forwards_epoch_to_both_samplers():
    observation = _EpochAwareLoader(range(4))
    mechanism = _EpochAwareLoader(range(2))
    loader = DualStreamLoader(observation, mechanism)

    loader.set_epoch(7)

    assert observation.epochs == [7]
    assert mechanism.epochs == [7]


def test_dual_stream_rejects_more_mechanism_than_observation_steps():
    observation = _EpochAwareLoader(range(2))
    mechanism = _EpochAwareLoader(range(3))

    try:
        DualStreamLoader(observation, mechanism)
    except ValueError as exc:
        assert "may not exceed" in str(exc)
    else:
        raise AssertionError("invalid dual-stream geometry was accepted")
