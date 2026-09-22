"""Dataset-agnostic cadence and dynamics-time protocol support.

Datasets using this mixin must provide:

* ``tracklet_anno_list`` and ``tracklet_len_list``;
* ``_tracklet_identity(source_idx, tracklet)``;
* ``_anno_frame_token(anno)``;
* ``_anno_timestamp(anno)``.

The mixin intentionally owns protocol selection and manifest validation, but
does not know how a dataset stores point clouds or boxes.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from datasets.protocol_utils import normalize_dynamics_time_mode
from datasets.protocol_utils import (
    canonical_sha256,
    file_sha256,
    payload_with_content_sha256,
    verify_content_sha256,
)


class TemporalProtocolMixin:
    """Reusable within-track variable-rate and effective-time controls."""

    _SUPPORTED_VIRTUAL_RATE_MODES = {
        "none",
        "manifest",
        "gap_pattern",
        "periodic_drop",
        "burst_drop",
        "random_drop",
        "stride",
    }

    def _configure_temporal_protocol(
            self, kwargs, *, dataset_name, version, default_delta_t):
        self.protocol_dataset_name = str(dataset_name)
        self.version = str(version)
        self.virtual_rate_mode = self._normalize_virtual_rate_mode(
            kwargs.get("virtual_rate_mode", "none"))
        if self.virtual_rate_mode not in self._SUPPORTED_VIRTUAL_RATE_MODES:
            supported = ", ".join(sorted(self._SUPPORTED_VIRTUAL_RATE_MODES))
            raise ValueError(
                f"Unsupported virtual_rate_mode={self.virtual_rate_mode!r}; "
                f"expected one of: {supported}")

        self.virtual_rate_gap_pattern = self._parse_int_list(
            kwargs.get("virtual_rate_gap_pattern", [1, 1, 2, 4]),
            default=[1, 1, 2, 4])
        self.virtual_rate_stride = int(kwargs.get("virtual_rate_stride", 2))
        self.virtual_rate_drop_every = int(
            kwargs.get("virtual_rate_drop_every", 5))
        self.virtual_rate_drop_prob = float(
            kwargs.get("virtual_rate_drop_prob", 0.0))
        self.virtual_rate_seed = int(kwargs.get("virtual_rate_seed", 42))
        self.virtual_rate_max_gap = int(
            kwargs.get("virtual_rate_max_gap", 5))
        self.virtual_rate_keep_first = self._parse_bool(
            kwargs.get("virtual_rate_keep_first", True))
        self.virtual_rate_keep_last = self._parse_bool(
            kwargs.get("virtual_rate_keep_last", True))
        self.virtual_rate_min_tracklet_len = int(
            kwargs.get("virtual_rate_min_tracklet_len", 0))
        self.virtual_rate_manifest = str(
            kwargs.get("virtual_rate_manifest", "") or "")
        self.virtual_rate_manifest_strict = self._parse_bool(
            kwargs.get("virtual_rate_manifest_strict", True))
        self.virtual_rate_manifest_allow_create = self._parse_bool(
            kwargs.get("virtual_rate_manifest_allow_create", False))
        self.virtual_rate_manifest_require_commit_match = self._parse_bool(
            kwargs.get("virtual_rate_manifest_require_commit_match", False))
        self.protocol_role = str(kwargs.get("protocol_role", "eval"))
        if self.virtual_rate_mode != 'none' or self.virtual_rate_manifest:
            raise ValueError('v31 uses ct_frame_stride; nondefault virtual-rate controls are unsupported')
        self.virtual_rate_manifest_content_sha256 = ""
        self.virtual_rate_manifest_file_sha256 = ""
        if self.virtual_rate_mode == "none" and self.virtual_rate_manifest:
            self.virtual_rate_mode = "manifest"
        if self.virtual_rate_mode == "manifest" and not self.virtual_rate_manifest:
            raise ValueError(
                "virtual_rate_mode=manifest requires virtual_rate_manifest")

        self.virtual_rate_burst_keep_lengths = self._parse_int_list(
            kwargs.get("virtual_rate_burst_keep_lengths", [3, 2, 3]),
            default=[3, 2, 3])
        self.virtual_rate_burst_skip_lengths = self._parse_int_list(
            kwargs.get("virtual_rate_burst_skip_lengths", [2, 3, 3]),
            default=[2, 3, 3])
        self._validate_virtual_rate_parameters()

        self.dynamics_time_mode = normalize_dynamics_time_mode(
            kwargs.get("dynamics_time_mode", "true"))
        self.dynamics_fixed_delta_t = float(
            kwargs.get("dynamics_fixed_delta_t", default_delta_t))
        if not np.isfinite(self.dynamics_fixed_delta_t):
            raise ValueError("dynamics_fixed_delta_t must be finite")
        if self.dynamics_fixed_delta_t <= 0:
            raise ValueError("dynamics_fixed_delta_t must be positive")
        self.dynamics_time_manifest = str(
            kwargs.get("dynamics_time_manifest", "") or "")
        self.dynamics_time_manifest_strict = self._parse_bool(
            kwargs.get("dynamics_time_manifest_strict", True))
        self.dynamics_time_manifest_require_commit_match = self._parse_bool(
            kwargs.get("dynamics_time_manifest_require_commit_match", False))
        self.dynamics_time_manifest_content_sha256 = ""
        self.dynamics_time_manifest_file_sha256 = ""
        self._shuffled_effective_timestamps = {}

    def _initialize_temporal_protocol(self):
        if not self.tracklet_anno_list:
            raise RuntimeError(
                f"No {self.protocol_dataset_name} tracklets remain before "
                "applying the temporal protocol")
        self.virtual_rate_meta = []
        self.virtual_rate_summary = self._build_virtual_rate_summary(
            original_lengths=self.tracklet_len_list,
            filtered_lengths=self.tracklet_len_list)
        self._apply_virtual_rate()
        self._prepare_dynamics_time()

    @staticmethod
    def _normalize_virtual_rate_mode(mode):
        mode = str(mode or "none").strip().lower().replace("-", "_")
        aliases = {
            "": "none",
            "off": "none",
            "false": "none",
            "no": "none",
            "gap": "gap_pattern",
            "gap_pattern_manifest": "gap_pattern",
            "periodic": "periodic_drop",
            "periodicdrop": "periodic_drop",
            "random": "random_drop",
            "random_drop_manifest": "random_drop",
            "randomdrop": "random_drop",
            "burst": "burst_drop",
            "burstdrop": "burst_drop",
            "interval": "stride",
            "fixed_interval": "stride",
        }
        return aliases.get(mode, mode)

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, str):
            return value.strip().lower() not in (
                "0", "false", "no", "off", "")
        return bool(value)

    @staticmethod
    def _parse_int_list(value, default=None):
        if value is None:
            return list(default or [])
        if isinstance(value, str):
            cleaned = value.replace("[", "").replace("]", "").replace(",", " ")
            values = [item for item in cleaned.split() if item]
        else:
            values = list(value)
        parsed = [int(item) for item in values]
        return parsed if parsed else list(default or [])

    def _validate_virtual_rate_parameters(self):
        if any(gap <= 0 for gap in self.virtual_rate_gap_pattern):
            raise ValueError("virtual_rate_gap_pattern values must be positive")
        if self.virtual_rate_stride <= 0:
            raise ValueError("virtual_rate_stride must be positive")
        if self.virtual_rate_drop_every < 2:
            raise ValueError("virtual_rate_drop_every must be at least 2")
        if not 0.0 <= self.virtual_rate_drop_prob <= 0.95:
            raise ValueError("virtual_rate_drop_prob must be in [0, 0.95]")
        if self.virtual_rate_max_gap <= 0:
            raise ValueError("virtual_rate_max_gap must be positive")
        if self.virtual_rate_min_tracklet_len < 0:
            raise ValueError("virtual_rate_min_tracklet_len must be non-negative")
        if (not self.virtual_rate_burst_keep_lengths
                or any(x <= 0 for x in self.virtual_rate_burst_keep_lengths)):
            raise ValueError(
                "virtual_rate_burst_keep_lengths must contain positive values")
        if (not self.virtual_rate_burst_skip_lengths
                or any(x <= 0 for x in self.virtual_rate_burst_skip_lengths)):
            raise ValueError(
                "virtual_rate_burst_skip_lengths must contain positive values")



    def _virtual_rate_cache_tag(self):
        return ''

    @staticmethod
    def _git_state():
        root = Path(__file__).resolve().parents[1]
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                text=True,
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=str(root),
                text=True,
            ).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = "unknown", True
        return {"commit": commit, "dirty_tracked": dirty}

    def _manifest_header(self):
        return {
            "dataset": self.protocol_dataset_name,
            "version": self.version,
            "split": self.split,
            "category_name": self.category_name,
            "protocol_role": self.protocol_role,
        }

    def _virtual_rate_protocol(self):
        return {
            "mode": self.virtual_rate_mode,
            "gap_pattern": list(self.virtual_rate_gap_pattern),
            "stride": self.virtual_rate_stride,
            "drop_every": self.virtual_rate_drop_every,
            "drop_prob": self.virtual_rate_drop_prob,
            "seed": self.virtual_rate_seed,
            "max_gap": self.virtual_rate_max_gap,
            "keep_first": self.virtual_rate_keep_first,
            "keep_last": self.virtual_rate_keep_last,
            "min_tracklet_len": self.virtual_rate_min_tracklet_len,
            "burst_keep_lengths": list(
                self.virtual_rate_burst_keep_lengths),
            "burst_skip_lengths": list(
                self.virtual_rate_burst_skip_lengths),
        }

    def _validate_manifest_header(
            self, manifest, label, *, require_commit_match):
        verify_content_sha256(manifest, label)
        for key, value in self._manifest_header().items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"{label} {key} mismatch: expected={value!r}, "
                    f"found={manifest.get(key)!r}")
        if require_commit_match:
            current = self._git_state()["commit"]
            recorded = manifest.get("code", {}).get("commit")
            if recorded != current:
                raise ValueError(
                    f"{label} commit mismatch: expected current {current}, "
                    f"found {recorded}")

    def _build_virtual_rate_summary(
            self, original_lengths, filtered_lengths):
        original_lengths = list(original_lengths)
        filtered_lengths = list(filtered_lengths)
        original_frames = int(sum(original_lengths))
        filtered_frames = int(sum(filtered_lengths))
        dropped = original_frames - filtered_frames
        ratio = float(dropped / original_frames) if original_frames else 0.0
        return {
            "mode": self.virtual_rate_mode,
            "tracklets_before": len(original_lengths),
            "tracklets_after": len(filtered_lengths),
            "frames_before": original_frames,
            "frames_after": filtered_frames,
            "dropped_frame_ratio": ratio,
            "min_tracklet_len": (
                int(min(filtered_lengths)) if filtered_lengths else 0),
            "mean_tracklet_len": (
                float(np.mean(filtered_lengths)) if filtered_lengths else 0.0),
            "cache_tag": self._virtual_rate_cache_tag() or "none",
        }











    def _apply_virtual_rate(self):
        original_lengths = list(self.tracklet_len_list)
        new_tracklets = []
        new_lengths = []
        active_meta = []
        manifest_meta = []

        for source_idx, tracklet in enumerate(self.tracklet_anno_list):
            identity = self._tracklet_identity(source_idx, tracklet)
            keep_indices = list(range(len(tracklet)))
            included = len(keep_indices) >= 1

            record = {
                **identity,
                "source_tracklet": source_idx,
                "original_len": len(tracklet),
                "kept_len": len(keep_indices),
                "included": bool(included),
                "keep_indices": keep_indices,
            }
            manifest_meta.append(record)
            if not included:
                continue
            new_tracklets.append(
                [tracklet[idx] for idx in keep_indices])
            new_lengths.append(len(keep_indices))
            active_meta.append(record)

        if not new_tracklets:
            raise RuntimeError(
                f"virtual_rate_mode={self.virtual_rate_mode} removed all "
                "tracklets. Lower virtual_rate_min_tracklet_len or use a "
                "milder protocol.")
        self.tracklet_anno_list = new_tracklets
        self.tracklet_len_list = new_lengths
        self.virtual_rate_meta = active_meta
        self.virtual_rate_summary = self._build_virtual_rate_summary(
            original_lengths=original_lengths,
            filtered_lengths=new_lengths)
        selection = [
            {
                "tracklet_key": entry["tracklet_key"],
                "included": entry["included"],
                "keep_indices": entry["keep_indices"],
            }
            for entry in manifest_meta
        ]
        self.virtual_rate_selection_sha256 = canonical_sha256(selection)

    def get_tracklet_key(self, tracklet_id):
        return str(
            self.virtual_rate_meta[int(tracklet_id)]["tracklet_key"])

    def get_endpoint_key(self, tracklet_id, frame_id):
        tracklet_id = int(tracklet_id)
        frame_id = int(frame_id)
        anno = self.tracklet_anno_list[tracklet_id][frame_id]
        return (
            f"{self.get_tracklet_key(tracklet_id)}/frame/"
            f"{self._anno_frame_token(anno)}")

    def _endpoint_records(self):
        records = []
        for tracklet_id, annos in enumerate(self.tracklet_anno_list):
            previous_timestamp = None
            for frame_index, anno in enumerate(annos):
                timestamp = float(self._anno_timestamp(anno))
                if not np.isfinite(timestamp):
                    raise ValueError(
                        "Non-finite physical timestamp at "
                        f"{self.get_endpoint_key(tracklet_id, frame_index)}")
                incoming = None
                if previous_timestamp is not None:
                    incoming = timestamp - previous_timestamp
                    if not np.isfinite(incoming) or incoming <= 0:
                        raise ValueError(
                            "Non-positive physical timestamp gap at "
                            f"{self.get_endpoint_key(tracklet_id, frame_index)}")
                records.append({
                    "endpoint_key": self.get_endpoint_key(
                        tracklet_id, frame_index),
                    "tracklet_key": self.get_tracklet_key(tracklet_id),
                    "frame_token": self._anno_frame_token(anno),
                    "frame_index": frame_index,
                    "timestamp_real": timestamp,
                    "real_incoming_delta_t": incoming,
                })
                previous_timestamp = timestamp
        return records

    @staticmethod
    def _deranged_permutation(size, seed):
        size = int(size)
        if size <= 1:
            return np.arange(size, dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        base = np.arange(size, dtype=np.int64)
        for _ in range(256):
            permutation = rng.permutation(size)
            if np.all(permutation != base):
                return permutation
        return np.roll(base, 1)

    def build_dynamics_time_manifest(self, output_path, seed=42):
        records = self._endpoint_records()
        transitions = [
            record for record in records if record["frame_index"] > 0]
        permutation = self._deranged_permutation(
            len(transitions), seed)

        effective_by_endpoint = {}
        mapping_rows = []
        for target_index, source_index in enumerate(
                permutation.tolist()):
            target = transitions[target_index]
            source = transitions[source_index]
            gap = float(source["real_incoming_delta_t"])
            effective_by_endpoint[target["endpoint_key"]] = {
                "source_endpoint_key": source["endpoint_key"],
                "effective_incoming_delta_t": gap,
            }
            mapping_rows.append({
                "target_endpoint_key": target["endpoint_key"],
                "source_endpoint_key": source["endpoint_key"],
            })

        entries = []
        cumulative_by_tracklet = {}
        for record in records:
            tracklet_key = record["tracklet_key"]
            if record["frame_index"] == 0:
                cumulative_by_tracklet[tracklet_key] = 0.0
                mapped = {
                    "source_endpoint_key": None,
                    "effective_incoming_delta_t": None,
                }
            else:
                mapped = effective_by_endpoint[record["endpoint_key"]]
                cumulative_by_tracklet[tracklet_key] += float(
                    mapped["effective_incoming_delta_t"])
            entries.append({
                **record,
                **mapped,
                "effective_timestamp": cumulative_by_tracklet[
                    tracklet_key],
            })

        payload = {
            "schema": "ct_seqtrack.dynamics_time_manifest",
            "schema_version": 1,
            **self._manifest_header(),
            "hist_num": int(self.hist_num),
            "mode": "shuffled",
            "seed": int(seed),
            "endpoint_count": len(entries),
            "transition_count": len(transitions),
            "virtual_rate_selection_sha256":
                self.virtual_rate_selection_sha256,
            "virtual_rate_manifest_content_sha256": (
                self.virtual_rate_manifest_content_sha256 or None),
            "permutation_sha256": canonical_sha256(mapping_rows),
            "code": self._git_state(),
            "entries": entries,
        }
        payload = payload_with_content_sha256(payload)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return {
            "path": str(output),
            "content_sha256": payload["content_sha256"],
            "file_sha256": file_sha256(output),
            "permutation_sha256": payload["permutation_sha256"],
            "endpoint_count": len(entries),
            "transition_count": len(transitions),
        }

    def _load_dynamics_time_manifest(self):
        path = self.dynamics_time_manifest
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                "dynamics_time_mode=shuffled requires an existing offline "
                f"dynamics_time_manifest; not found: {path!r}")
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get(
                "schema") != "ct_seqtrack.dynamics_time_manifest":
            raise ValueError("Unsupported dynamics-time manifest schema")
        self._validate_manifest_header(
            manifest,
            "dynamics-time manifest",
            require_commit_match=(
                self.dynamics_time_manifest_require_commit_match),
        )
        expected = {
            "hist_num": int(self.hist_num),
            "mode": "shuffled",
            "virtual_rate_selection_sha256":
                self.virtual_rate_selection_sha256,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"dynamics-time manifest {key} mismatch: "
                    f"expected={value!r}, found={manifest.get(key)!r}")

        entries = manifest.get("entries", [])
        by_endpoint = {}
        mapping_rows = []
        source_keys = []
        for entry in entries:
            key = str(entry.get("endpoint_key", ""))
            if not key or key in by_endpoint:
                raise ValueError(
                    "dynamics-time manifest contains a missing or duplicate "
                    "endpoint_key")
            by_endpoint[key] = entry
            if int(entry.get("frame_index", -1)) > 0:
                source_key = str(entry.get("source_endpoint_key", ""))
                source_keys.append(source_key)
                mapping_rows.append({
                    "target_endpoint_key": key,
                    "source_endpoint_key": source_key,
                })

        expected_records = self._endpoint_records()
        expected_by_key = {
            record["endpoint_key"]: record
            for record in expected_records
        }
        expected_keys = set(expected_by_key)
        if set(by_endpoint) != expected_keys:
            missing = sorted(expected_keys - set(by_endpoint))[:3]
            extra = sorted(set(by_endpoint) - expected_keys)[:3]
            raise ValueError(
                "dynamics-time manifest endpoint set mismatch: "
                f"missing={missing}, extra={extra}")
        transition_keys = {
            record["endpoint_key"] for record in expected_records
            if record["frame_index"] > 0
        }
        if (set(source_keys) != transition_keys
                or len(source_keys) != len(set(source_keys))):
            raise ValueError(
                "dynamics-time manifest is not a one-to-one split "
                "permutation")
        actual_mapping_sha = canonical_sha256(mapping_rows)
        if actual_mapping_sha != manifest.get("permutation_sha256"):
            raise ValueError(
                "dynamics-time manifest permutation_sha256 mismatch")

        cumulative_by_tracklet = {}
        for record in expected_records:
            key = record["endpoint_key"]
            entry = by_endpoint[key]
            tracklet_key = record["tracklet_key"]
            if record["frame_index"] == 0:
                expected_timestamp = 0.0
                cumulative_by_tracklet[tracklet_key] = 0.0
                if entry.get("source_endpoint_key") is not None:
                    raise ValueError(
                        "First-frame dynamics-time entry must not have a "
                        "source endpoint")
            else:
                source_key = str(entry["source_endpoint_key"])
                expected_gap = float(
                    expected_by_key[source_key]["real_incoming_delta_t"])
                found_gap = float(
                    entry["effective_incoming_delta_t"])
                if not np.isclose(
                        found_gap, expected_gap, rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"dynamics-time mapped gap mismatch at {key}")
                cumulative_by_tracklet[tracklet_key] += expected_gap
                expected_timestamp = cumulative_by_tracklet[tracklet_key]
            if not np.isclose(
                    float(entry["effective_timestamp"]),
                    expected_timestamp,
                    rtol=0.0,
                    atol=1e-9):
                raise ValueError(
                    f"dynamics-time cumulative timestamp mismatch at {key}")

        self.dynamics_time_manifest_content_sha256 = manifest[
            "content_sha256"]
        self.dynamics_time_manifest_file_sha256 = file_sha256(path)
        self.dynamics_time_permutation_sha256 = actual_mapping_sha
        self._shuffled_effective_timestamps = {
            key: float(entry["effective_timestamp"])
            for key, entry in by_endpoint.items()
        }

    def _prepare_dynamics_time(self):
        self.dynamics_time_permutation_sha256 = ""
        if self.dynamics_time_mode == "shuffled":
            self._load_dynamics_time_manifest()
        elif self.dynamics_time_manifest:
            raise ValueError(
                "dynamics_time_manifest is only valid when "
                "dynamics_time_mode=shuffled; clear it for true/fixed "
                "controls.")
        self.dynamics_time_summary = {
            "mode": self.dynamics_time_mode,
            "fixed_delta_t": self.dynamics_fixed_delta_t,
            "manifest": self.dynamics_time_manifest,
            "manifest_content_sha256":
                self.dynamics_time_manifest_content_sha256,
            "manifest_file_sha256":
                self.dynamics_time_manifest_file_sha256,
            "permutation_sha256":
                self.dynamics_time_permutation_sha256,
        }

    def _enrich_frames_with_effective_time(
            self, seq_id, frame_ids, frames):
        enriched = []
        for frame, frame_id in zip(frames, frame_ids):
            frame = dict(frame)
            endpoint_key = self.get_endpoint_key(seq_id, frame_id)
            if self.dynamics_time_mode == "true":
                effective_timestamp = frame.get("timestamp")
            elif self.dynamics_time_mode == "fixed":
                effective_timestamp = (
                    float(frame_id) * self.dynamics_fixed_delta_t)
            else:
                effective_timestamp = self._shuffled_effective_timestamps[
                    endpoint_key]
            if effective_timestamp is None or not np.isfinite(
                    float(effective_timestamp)):
                raise ValueError(
                    f"Invalid effective timestamp for {endpoint_key}")
            frame["_ct_endpoint_key"] = endpoint_key
            frame["_ct_dynamics_time_mode"] = self.dynamics_time_mode
            frame["_ct_effective_timestamp"] = float(effective_timestamp)
            enriched.append(frame)
        return enriched
