"""Check Windows-compatible process workers produce the same CSVs as serial runs."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class ParallelTests(unittest.TestCase):
    def test_serial_and_parallel_outputs_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            for i in range(2):
                folder = inputs / "test" / f"record_{i}"
                folder.mkdir(parents=True)
                y = np.linspace(0, 10, 49)
                points = np.array([np.column_stack((np.sin(y+t/10), y)) for t in range(18)])
                points[4:6] = np.nan
                table = np.zeros(18, dtype=[("timestamp", "f8"), ("worm_index", "f8"), ("length", "f8")])
                table["timestamp"] = np.arange(18)
                table["worm_index"] = 1
                table["length"] = np.arange(18)**2
                with h5py.File(folder / "worm_features.hdf5", "w") as f:
                    f["coordinates/skeletons"] = points
                    f["features_timeseries"] = table
                    f["features_timeseries"].attrs["fps"] = 25
            for script in ("openworm_skeleton_angles.py", "openworm_manual_features.py"):
                destinations = []
                for workers in (1, 2):
                    output = root / f"{script}_{workers}"
                    result = subprocess.run([sys.executable, str(ROOT/"openworm"/script),
                        "--input-dir", str(inputs), "--output-dir", str(output), "--workers", str(workers)],
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn("TOTAL ETA", result.stderr)
                    destinations.append(output)
                for i in range(2):
                    relative = Path("test")/f"record_{i}.csv"
                    self.assertEqual((destinations[0]/relative).read_bytes(), (destinations[1]/relative).read_bytes())


if __name__ == "__main__":
    unittest.main()
