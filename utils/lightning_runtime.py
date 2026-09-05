"""Small Lightning callbacks for formal scratch-run durability."""

from pathlib import Path
from functools import wraps

import torch
from pytorch_lightning.callbacks import Callback, Checkpoint

from utils.training_isolation import (
    capture_global_rng_state, restore_global_rng_state,
)


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
        self._validation_setup_complete = False
        self._validation_loop = None

    def setup(self, trainer, pl_module, stage):
        del pl_module
        if stage != "fit":
            return
        loop = trainer.fit_loop.epoch_loop.val_loop
        if self._validation_loop is loop:
            return
        self._validation_loop = loop
        original_setup_data = loop.setup_data

        @wraps(original_setup_data)
        def setup_data_with_resume_rng():
            # PL 2.0.2 may first initialize validation several epochs after
            # resume. Its iterability probe consumes a worker base seed;
            # on_validation_start is already too late (workers were started
            # by evaluation.reset). Isolate only a repeated setup, before
            # the real validation iterator is created. Do not eagerly build
            # validation data or rewind any actual validation sampling.
            uninitialized = loop._combined_loader is None
            repeated_setup = uninitialized and self._validation_setup_complete
            generator = self.generators.get("validation")
            generator_state = (
                generator.get_state() if repeated_setup and generator is not None
                else None)
            global_state = capture_global_rng_state() if repeated_setup else None
            try:
                return original_setup_data()
            finally:
                if repeated_setup:
                    if generator_state is not None:
                        generator.set_state(generator_state)
                    restore_global_rng_state(global_state)
                if loop._combined_loader is not None:
                    self._validation_setup_complete = True
                    # A checkpoint before its first validation has no probe
                    # to replay. If first-ever setup occurs before resumed
                    # epoch-start, preserve its legitimate seed consumption.
                    pending = getattr(self, "_pending_resume_states", None)
                    if (uninitialized and not repeated_setup
                            and pending is not None and generator is not None):
                        pending["validation"] = generator.get_state()

        loop.setup_data = setup_data_with_resume_rng

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
            "validation_setup_complete": self._validation_setup_complete,
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
        # Lightning 2.0.x tests iterability by calling iter(dataloader) during
        # setup, after callback restore. That consumes a DataLoader base seed
        # (and may prefetch worker batches). Restore again after setup and
        # before the first resumed epoch creates its actual iterators.
        self._pending_resume_states = {
            name: state.detach().cpu().clone() for name, state in states.items()}
        self._validation_setup_complete = bool(
            state_dict.get("validation_setup_complete", False))

    def on_train_epoch_start(self, trainer, pl_module):
        del trainer, pl_module
        pending = getattr(self, '_pending_resume_states', None)
        if pending is not None:
            for name, state in pending.items():
                self.generators[name].set_state(state)
            self._pending_resume_states = None


class FinalWindowCheckpoint(Checkpoint):
    """Save final epochs after the module's epoch-end transaction.

    Lightning 2.0.x runs ordinary Callback.on_train_epoch_end before the
    module hook, but Checkpoint callbacks afterwards.  The marker base is
    necessary: the module owns manual scheduler updates and epoch audits.
    """

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
