"""Paper-facing CT-SeqTrack entry point.

``SEQTRACK3D`` remains the B0 implementation.  This facade fixes the formal
variant and isolation contracts before constructing it, so paper experiments
cannot silently re-enable a historical branch through inherited YAML flags.
"""

import subprocess
from pathlib import Path

from models.seqtrack3d import SEQTRACK3D
from models.ct_variant import (
    configure_ct_variant,
    get_config,
)
from utils.action_calibration import (
    action_calibration_config_identity,
    load_action_calibration,
    sha256_file,
    sha256_json,
    validate_action_calibration,
)


class CTSEQTRACK(SEQTRACK3D):
    """Composition root for B0 observation, B1 prior, B2 evidence and B3."""

    def __init__(self, config, **kwargs):
        config = configure_ct_variant(config)
        super().__init__(config, **kwargs)
        artifact_path = get_config(config, "ct_action_calibration_path")
        if self.ct_enable_b3 and artifact_path:
            checkpoint_path = get_config(config, "checkpoint")
            manifest_sha = get_config(
                config, "ct_calibration_tracklet_manifest_sha256")
            dev_manifest_sha = get_config(
                config, "ct_dev_tracklet_manifest_sha256")
            try:
                if (not checkpoint_path or not manifest_sha
                        or not dev_manifest_sha):
                    raise ValueError(
                        "action calibration requires checkpoint and "
                        "calibration plus dev tracklet manifest SHAs")
                artifact = load_action_calibration(artifact_path)
                if bool(artifact.get("compatibility_self_check", False)):
                    raise ValueError(
                        "formal selective evaluation forbids calibration "
                        "self-check artifacts")
                config_sha = sha256_json(
                    action_calibration_config_identity(config))
                code_version = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[1],
                    text=True).strip()
                validate_action_calibration(
                    artifact, sha256_file(checkpoint_path), config_sha,
                    manifest_sha, dev_manifest_sha,
                    code_version=code_version)
                self.ct_joint_router.install_calibration(
                    artifact["thresholds"]["presence"],
                    artifact["thresholds"]["action"])
                self._ct_action_calibration = artifact
            except (OSError, ValueError, RuntimeError) as error:
                # v26 deployment contract: any artifact mismatch installs no
                # thresholds.  B3 remains uncalibrated and returns B0 exactly.
                self._ct_action_calibration_error = str(error)
