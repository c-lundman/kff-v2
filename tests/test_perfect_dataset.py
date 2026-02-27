import csv
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MINUTE_PATH = REPO_ROOT / "data" / "synthetic" / "perfect" / "day_2026-01-15" / "minute_flows.csv"


class TestPerfectDataset(unittest.TestCase):
    def test_occupancy_never_negative(self) -> None:
        with MINUTE_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                occ = int(row["occupancy_end"])
                self.assertGreaterEqual(
                    occ,
                    0,
                    f"Perfect scenario should stay physical, got occupancy {occ}",
                )

    def test_occupancy_balance_identity(self) -> None:
        prev_occ = 0
        with MINUTE_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                in_count = int(row["in_count"])
                out_count = int(row["out_count"])
                occ = int(row["occupancy_end"])
                self.assertEqual(
                    occ,
                    prev_occ + in_count - out_count,
                    "Minute occupancy does not match stock-flow balance identity",
                )
                prev_occ = occ


if __name__ == "__main__":
    unittest.main()

