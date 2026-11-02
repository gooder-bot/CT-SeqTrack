"""只用标准库的输入清单回归用例；本轮仅写入与静态 AST 检查，未执行。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("audit_kitti_inputs",
    Path(__file__).resolve().parents[1] / "tools" / "audit_kitti_inputs.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


CALIB = "Tr_velo_cam: 1 0 0 0 0 1 0 0 0 0 1 0\nR_rect: 1 0 0 0 1 0 0 0 1\n"


def label(frame, *, track=0, category="Car", height=1.5, length=4.0):
    return f"{frame} {track} {category} 0 0 0 0 0 50 50 {height} 1.6 {length} 0 0 10 0\n"


class KittiInputAuditTests(unittest.TestCase):
    def make_dataset(self, root):
        for folder in ("label_02", "calib", "velodyne"):
            (root / folder).mkdir()
        for index in range(21):
            scene = f"{index:04d}"
            (root / "label_02" / (scene + ".txt")).write_text("", encoding="utf-8")
            (root / "calib" / (scene + ".txt")).write_text(CALIB, encoding="utf-8")
            (root / "velodyne" / scene).mkdir()

    def test_roles_nominal_exposure_real_gaps_and_size_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            labels = "".join(label(frame, length=4.0 if i == 0 else 4.1)
                             for i, frame in enumerate((0, 1, 3, 4, 5)))
            (root / "label_02" / "0000.txt").write_text(labels, encoding="utf-8")
            for frame in (0, 1, 3, 4, 5):
                (root / "velodyne" / "0000" / f"{frame:06d}.bin").write_bytes(bytes(16))
            result = AUDIT.inspect_dataset(root, "Car")
            self.assertEqual(result["blocking_findings"], [])
            self.assertEqual(result["proposed_roles"]["dev"]["scenes"], ["0017", "0018"])
            self.assertEqual(result["proposed_roles"]["train"]["nominal_original_seqtrack_rows"], 20)
            self.assertEqual(result["proposed_roles"]["train"]["nominal_ct_rows"], 16)
            self.assertAlmostEqual(result["scenes"][0]["frame_gaps_seconds"]["max"], 0.2)
            self.assertAlmostEqual(result["scenes"][0]["mtp_h4_dt_seconds"]["max"], 0.5)
            self.assertEqual(result["scenes"][0]["tracks_with_size_change_over_1mm"], 1)

    def test_missing_and_malformed_clouds_are_blockers_without_dropping_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            (root / "label_02" / "0019.txt").write_text(label(0) + label(1), encoding="utf-8")
            (root / "velodyne" / "0019" / "000001.bin").write_bytes(bytes(15))
            result = AUDIT.inspect_dataset(root, "Car")
            self.assertEqual(result["proposed_roles"]["test"]["object_frames"], 2)
            self.assertIn("velodyne/0019/000000.bin: missing", result["blocking_findings"])
            self.assertIn("velodyne/0019/000001.bin: malformed byte count", result["blocking_findings"])

    def test_invalid_labels_rejected(self):
        for text in (label(0, height=0), label(0, height="nan"), label(0) + label(0), "0 0 Car\n"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_dataset(root)
                (root / "label_02" / "0000.txt").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    AUDIT.inspect_dataset(root, "Car")

    def test_missing_calibration_and_scene_directory_are_blockers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            (root / "calib" / "0020.txt").unlink()
            (root / "velodyne" / "0018").rmdir()
            result = AUDIT.inspect_dataset(root, "Car")
            self.assertIn("0020: calibration missing", result["blocking_findings"])
            self.assertIn("velodyne/0018: missing_directory", result["blocking_findings"])

    def test_empty_cloud_is_valid_observation_and_zero_endpoint_role_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            (root / "label_02" / "0019.txt").write_text(label(0), encoding="utf-8")
            (root / "velodyne" / "0019" / "000000.bin").write_bytes(b"")
            result = AUDIT.inspect_dataset(root, "Car")
            self.assertEqual(result["blocking_findings"], [])
            self.assertEqual(result["empty_files"], ["velodyne/0019/000000.bin"])
            self.assertEqual(result["proposed_roles"]["test"]["object_frames"], 1)
            self.assertFalse(result["input_readiness"]["ready"])
            self.assertEqual(result["role_readiness"]["train"]["reason"], "no_objects")
            self.assertEqual(result["role_readiness"]["test"]["reason"], "no_predictable_endpoints")

    def test_determinant_one_shear_is_not_valid_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calib.txt"
            path.write_text("Tr_velo_cam: 1 0.1 0 0 0 1 0 0 0 0 1 0\nR_rect: 1 0 0 0 1 0 0 0 1\n",
                            encoding="utf-8")
            result = AUDIT.inspect_calibration(path)
            self.assertEqual(result["status"], "invalid_rotation_geometry")
            self.assertAlmostEqual(result["rotation_determinant"], 1.0)
            self.assertGreater(result["rotation_orthogonality_error"], 0.002)

    def test_output_must_stay_under_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                AUDIT.main(["--path", temporary, "--output-dir", str(Path(temporary) / "outside")])


if __name__ == "__main__":
    unittest.main()
