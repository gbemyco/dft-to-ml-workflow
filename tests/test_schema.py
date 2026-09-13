import unittest

import pandas as pd

from dftml.schema import modelling_frame, validate_frame


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"sample_id": "a", "material_family": "A", "adsorbate": "H", "functional": "PBE", "band_gap_ev": 2.0, "work_function_ev": 5.0, "charge_transfer_e": 0.1, "binding_energy_ev": -0.4, "coordination_number": 2, "converged": True, "target_adsorption_ev": -0.2},
            {"sample_id": "b", "material_family": "B", "adsorbate": "O", "functional": "PBE", "band_gap_ev": 1.0, "work_function_ev": 4.5, "charge_transfer_e": 0.2, "binding_energy_ev": -1.0, "coordination_number": 1, "converged": False, "target_adsorption_ev": -0.8},
        ])

    def test_valid_frame(self):
        self.assertEqual(validate_frame(self.frame), [])

    def test_feature_engineering_and_filter(self):
        clean = modelling_frame(self.frame)
        self.assertEqual(len(clean), 1)
        self.assertAlmostEqual(clean.loc[0, "electronic_alignment_ev"], 4.0)
        self.assertAlmostEqual(clean.loc[0, "binding_per_coord_ev"], -0.2)


if __name__ == "__main__":
    unittest.main()
