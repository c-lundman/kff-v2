from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSmokeLayout(unittest.TestCase):
    def test_expected_files_exist(self) -> None:
        required = [
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "README.md",
            REPO_ROOT / "scripts" / "generate_perfect_dataset.py",
            REPO_ROOT / "scripts" / "generate_lossy_datasets.py",
            REPO_ROOT / "data" / "synthetic" / "perfect" / "day_2026-01-15" / "events.csv",
            REPO_ROOT / "data" / "synthetic" / "perfect" / "day_2026-01-15" / "minute_flows.csv",
            REPO_ROOT / "data" / "synthetic" / "lossy" / "day_2026-01-15" / "mild_noise" / "summary.json",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing required artifact: {path}")


if __name__ == "__main__":
    unittest.main()

