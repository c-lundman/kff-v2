import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOSSY_ROOT = REPO_ROOT / "data" / "synthetic" / "lossy" / "day_2026-01-15"


class TestLossyVariants(unittest.TestCase):
    def test_expected_variants_exist(self) -> None:
        expected = {
            "mild_noise",
            "asymmetric_inflow_loss",
            "spurious_outflow",
            "mixed_heavy_noise",
        }
        actual = {p.name for p in LOSSY_ROOT.iterdir() if p.is_dir()}
        self.assertEqual(actual, expected)

    def test_at_least_one_variant_has_unphysical_naive_occupancy(self) -> None:
        has_negative = False
        for variant_dir in LOSSY_ROOT.iterdir():
            if not variant_dir.is_dir():
                continue
            summary_path = variant_dir / "summary.json"
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if int(summary["naive_occupancy"]["negative_minutes"]) > 0:
                has_negative = True
                break
        self.assertTrue(
            has_negative,
            "At least one lossy scenario should produce unphysical naive occupancy.",
        )


if __name__ == "__main__":
    unittest.main()

