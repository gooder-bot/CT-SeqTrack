"""Small Lightning callbacks for formal scratch-run durability."""

from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback


class DataLoaderGeneratorState(Callback):
    """Checkpoint explicit DataLoader generators at epoch boundaries.

    SeqTrack-strict uses an ordinary shuffled DataLoader, whose private
    ``torch.Generator`` is not part of the model checkpoint.  Persisting it
    keeps an allowed same-run resume on the same sample stream instead of
    silently replaying the epoch-0 shuffle.
    """

    SCHEMA = "ct_seqtrack.dataloader_generators.v1"

    def __init__(self, **generators):
        super().__init__()
        self.generators = {
            str(name): generator for name, generator in generators.items()
            if generator is not None
        }
        if not self.generators:
            raise ValueError("at least one DataLoader generator is required")
        if any(not isinstance(value, torch.Generator)
               for value in self.generators.values()):
            raise TypeError("DataLoader generator state requires torch.Generator")

    @property
    def state_key(self):
        return "ct_seqtrack.DataLoaderGeneratorState.v1"

    def state_dict(self):
        return {
            "schema": self.SCHEMA,
            "states": {
                name: generator.get_state()
                for name, generator in self.generators.items()
            },
        }

    def load_state_dict(self, state_dict):
        if (not isinstance(state_dict, dict)
                or state_dict.get("schema") != self.SCHEMA):
            raise ValueError("formal resume lacks DataLoader generator state")
        states = state_dict.get("states")
        if not isinstance(states, dict) or set(states) != set(self.generators):
            raise ValueError("formal resume DataLoader generator set mismatch")
        for name, generator in self.generators.items():
            state = states[name]
            if not torch.is_tensor(state):
                raise TypeError("DataLoader generator state must be a tensor")
            generator.set_state(state.detach().cpu())


class FinalWindowCheckpoint(Callback):
    """Save the final N epoch-boundary checkpoints without best-epoch bias."""

    def __init__(self, keep=3, directory_name="formal_checkpoints"):
        super().__init__()
        self.keep = int(keep)
        self.directory_name = str(directory_name)
        if self.keep <= 0:
            raise ValueError("final checkpoint window must be positive")
        if not self.directory_name:
            raise ValueError("final checkpoint directory name is required")

    @property
    def state_key(self):
        return (
            "ct_seqtrack.FinalWindowCheckpoint."
            f"keep={self.keep}.dir={self.directory_name}")

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        completed_epoch = int(trainer.current_epoch) + 1
        max_epochs = int(trainer.max_epochs)
        if completed_epoch <= max_epochs - self.keep:
            return
        directory = Path(trainer.default_root_dir) / self.directory_name
        directory.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            str(directory / f"epoch={completed_epoch:03d}.ckpt"))
