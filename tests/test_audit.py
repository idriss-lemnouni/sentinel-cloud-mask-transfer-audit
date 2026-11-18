import unittest

import numpy as np

from audit import compare_screen, index_change, scene_mask, score, vegetation_indices
from fetch_cloudsen12 import source_split


class CloudAuditTest(unittest.TestCase):
    def test_indices_use_surface_reflectance(self):
        values = vegetation_indices(np.array([0.2, 0]), np.array([0.6, 0]), np.array([0.3, 0]))
        self.assertAlmostEqual(float(values["NDVI"][0]), 0.5)
        self.assertAlmostEqual(float(values["NDMI"][0]), 1 / 3)
        self.assertAlmostEqual(float(values["SAVI"][0]), 0.6 / 1.3)
        self.assertTrue(np.isnan(values["NDVI"][1]))

    def test_sen2cor_codes_for_cloud_and_shadow(self):
        self.assertEqual(np.flatnonzero(scene_mask(np.arange(12))).tolist(), [3, 8, 9, 10])

    def test_score_excludes_invalid_and_uses_balanced_binary_recall(self):
        labels = np.array([0, 0, 1, 2, 3, 99])
        prediction = np.array([False, True, True, True, False, False])
        valid = labels != 99
        counts = compare_screen(labels, prediction, valid)
        self.assertEqual(counts["cloud_shadow"], {"pixels": 1, "correct": 0})
        self.assertAlmostEqual(score(counts), (0.5 + 2 / 3) / 2)

    def test_index_shift_keeps_denominators_and_empty_mask_explicit(self):
        input_values = np.array([0.8, 0.6, 0.2])
        truth = np.array([0, 0, 2])
        valid = np.array([True, True, True])
        counts = index_change(input_values, truth, np.array([True, False, False]), valid)
        self.assertEqual(counts["reference_clear_pixels"], 2)
        self.assertEqual(counts["retained_obscured_pixels"], 1)
        self.assertAlmostEqual(counts["mean_shift"], -0.3)
        self.assertIsNone(index_change(input_values, truth, valid, valid)["mean_shift"])

    def test_validation_split_alias(self):
        self.assertEqual(source_split({"tortilla:data_split": "val"}), "validation")
        self.assertEqual(source_split({"tortilla:data_split": "test"}), "test")
