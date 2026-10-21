"""v37 Lightning host；复用 v35 的 forward/backward/Adam/commit 事务。"""
import json
from pathlib import Path
import uuid

from models.ctseqtrackv31 import CTSEQTRACKV31, pl
from models.ct_v37.config import normalize_config, as_v35_config, config_identity
from models.ct_v37.data import BatchBuilder, build_loaders
from models.ct_v37.runtime import TrackingEvaluation, resume_payload, validate_resume_payload
from models.ct_v37.identity import model_schema, runtime_key, training_audit_schema


class CTSEQTRACKV37(CTSEQTRACKV31):
    def __init__(self, config=None, *, tracker=None, loaders=None):
        config = normalize_config(config)
        if tracker is None:
            from models.ct_v37.model import JointTracker
            tracker = JointTracker(config)
        super().__init__(as_v35_config(config, arm=config.v37_arm), tracker=tracker, loaders=loaders)
        self.config = config
        self.run_id = uuid.uuid4().hex
        self._loader_factory = build_loaders
        self.train_builder = BatchBuilder(config)
        self.evaluation_builder = BatchBuilder(config)
        self.evaluation = TrackingEvaluation(config)
        from models.ct_v37.training_diagnostics import TrainingDiagnostics
        from utils.b2_centroid_recipes import detach_mass_weights
        self.training_diagnostics = TrainingDiagnostics(config.seed,
            optimized=config.execution_profile != 'legacy',
            centroid_semantics=detach_mass_weights(config)) if config.v37_diagnostics else None
        if pl is not None:
            self.save_hyperparameters(dict(config=dict(config)))

    def on_train_epoch_start(self):
        if self._resume_sampler is not None:
            current = self._loaders['train'].batch_sampler.state_dict()
            if current.get('initial_seed_policy') != self._resume_sampler.get('initial_seed_policy'):
                raise ValueError('v37 resume initial seed policy mismatch')
        super().on_train_epoch_start()

    def _evaluation_start(self, role):
        self.evaluation_builder.reset()
        self.evaluation = TrackingEvaluation(self.config)
        if role in self._loaders:
            self.evaluation.add_singleton_tracks(self._loaders[role].dataset)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.config.log_dir:
            directory = Path(self.config.log_dir) / 'validation'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'epoch={int(self.current_epoch) + 1:03d}.json'
            content = json.dumps(self.evaluation_results, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
            if path.exists():
                if path.read_text(encoding='utf-8') != content:
                    raise FileExistsError('existing v37 validation differs: ' + str(path))
            else:
                with path.open('x', encoding='utf-8') as stream:
                    stream.write(content)

    def _write_epoch_audit(self, sampler):
        if not self.config.log_dir:
            return
        audit = dict(schema=training_audit_schema(self.config),
            model_schema=model_schema(self.config), config_sha256=config_identity(self.config),
            run_id=self.run_id, completed_epoch=self._completed_epoch,
            epoch_complete=self._epoch_complete, rows=self._epoch_rows,
            optimizer_steps=self._epoch_steps, sampler=sampler.state_dict())
        directory = Path(self.config.log_dir) / 'training_audits'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'epoch={self._completed_epoch:03d}.json'
        content = json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        if path.exists():
            if path.read_text(encoding='utf-8') != content:
                raise FileExistsError('existing v37 epoch audit differs: ' + str(path))
        else:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(content)

    def on_save_checkpoint(self, checkpoint):
        loader = self._loaders.get('train')
        checkpoint[runtime_key(self.config)] = resume_payload(self.config,
            completed_epoch=self._completed_epoch, complete=self._epoch_complete,
            rows=self._epoch_rows, steps=self._epoch_steps,
            sampler=loader.batch_sampler.state_dict() if loader is not None else None,
            run_id=self.run_id)

    def on_load_checkpoint(self, checkpoint):
        payload = validate_resume_payload(checkpoint.get(runtime_key(self.config)), self.config,
                                          training=not self.config.test)
        if not self.config.test and self.config.log_dir:
            manifest = Path(self.config.log_dir) / 'run_manifest.json'
            if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')).get('run_id') != payload['run_id']:
                raise ValueError('v37 resume requires the original run_id')
        self.run_id = payload['run_id']
        self._completed_epoch = int(payload['completed_epoch'])
        if not self.config.test:
            self._pending_rng = payload['rng']
            self._resume_sampler = payload['sampler']
