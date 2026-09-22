"""完整 epoch 边界的定期及最后窗口 checkpoint。"""

from pathlib import Path
from pytorch_lightning.callbacks import Checkpoint


class FinalWindowCheckpoint(Checkpoint):
    """Save final epochs after the module's epoch-end transaction.

    Lightning 2.0.x runs ordinary Callback.on_train_epoch_end before the
    module hook, but Checkpoint callbacks afterwards.  The marker base is
    necessary: the module owns manual scheduler updates and epoch audits.
    """

    def __init__(self, keep=3, directory_name="formal_checkpoints", every_n_epochs=0):
        super().__init__()
        self.keep = int(keep)
        self.directory_name = str(directory_name)
        self.every_n_epochs = int(every_n_epochs)
        if self.every_n_epochs < 0:
            raise ValueError("checkpoint interval must be nonnegative")
        if self.keep <= 0:
            raise ValueError("final checkpoint window must be positive")
        if not self.directory_name:
            raise ValueError("final checkpoint directory name is required")

    @property
    def state_key(self):
        key = (
            "ct_seqtrack.FinalWindowCheckpoint."
            f"keep={self.keep}.dir={self.directory_name}")
        return key + (f".every={self.every_n_epochs}" if self.every_n_epochs else "")

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        completed_epoch = int(trainer.current_epoch) + 1
        max_epochs = int(trainer.max_epochs)
        periodic = self.every_n_epochs > 0 and completed_epoch % self.every_n_epochs == 0
        if completed_epoch <= max_epochs - self.keep and not periodic:
            return
        directory = Path(trainer.default_root_dir) / self.directory_name
        directory.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            str(directory / f"epoch={completed_epoch:03d}.ckpt"))
