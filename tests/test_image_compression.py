import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_compression import ImageWorkspace, preview_image
from pca_tools import color_image_pca, color_image_svd


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(12).integers(0, 256, (18, 12, 3), dtype=np.uint8)

    def test_reuses_decomposition_and_bounds_results_as_rank_changes(self):
        workspace = ImageWorkspace(self.image)
        with patch('image_compression.np.linalg.svd', wraps=np.linalg.svd) as svd:
            for k in [1, 2, 4, 8, 2]:
                result = workspace.reconstruct('PCA', k)
                self.assertEqual(len(workspace.results), 1)
                self.assertEqual(result['reconstructed_image'].shape, self.image.shape)
            self.assertEqual(svd.call_count, 3)
            self.assertNotIn('SVD', workspace.factors)
            self.assertIs(workspace.reconstruct('PCA', 2), result)
            workspace.reconstruct('SVD', 2)
            workspace.reconstruct('SVD', 3)
            self.assertEqual(svd.call_count, 6)
            self.assertEqual(len(workspace.results), 2)
        self.assertTrue(all(factor.u.dtype == np.float32 for factor in workspace.factors['PCA']))
        self.assertNotIn('channels', result)

    def test_matches_previous_methods_with_at_most_byte_rounding_difference(self):
        for image in [self.image, self.image.transpose(1, 0, 2).copy()]:
            workspace = ImageWorkspace(image)
            for method, old in [('PCA', color_image_pca), ('SVD', color_image_svd)]:
                result = workspace.reconstruct(method, 4)
                previous = old(image, 4)
                error = np.abs(result['reconstructed_image'].astype(int) - previous['reconstructed_image'].astype(int))
                self.assertLessEqual(error.max(), 1)
                self.assertAlmostEqual(result['retained_variance'], previous['retained_variance'], places=5)
                expected_mse = np.mean((image.astype(float) - result['reconstructed_image']) ** 2)
                self.assertAlmostEqual(result['mse'], expected_mse)
                self.assertAlmostEqual(result['storage_ratio'], previous['storage_ratio'])

    def test_full_rank_reconstructs_pixels_and_constant_image_is_safe(self):
        for image in [self.image, np.full_like(self.image, 127)]:
            for method in ['PCA', 'SVD']:
                result = ImageWorkspace(image).reconstruct(method, min(image.shape[:2]))
                np.testing.assert_array_equal(result['reconstructed_image'], image)
                self.assertEqual(result['mse'], 0)
                self.assertTrue(np.isinf(result['psnr']))
        self.assertEqual(ImageWorkspace(np.full_like(self.image, 127)).reconstruct('PCA', 1)['retained_variance'], 0)

    def test_preview_preserves_frame_and_never_upscales(self):
        for width, height in [(120, 80), (80, 120), (80, 80)]:
            image = Image.new('RGB', (width, height), (10, 20, 30))
            preview = preview_image(image, 60)
            self.assertEqual(max(preview.shape[:2]), 60)
            self.assertAlmostEqual(preview.shape[1] / preview.shape[0], width / height)
            np.testing.assert_array_equal(preview[0, 0], [10, 20, 30])
            self.assertEqual(preview_image(image, None).shape, (height, width, 3))
            self.assertEqual(preview_image(image, 720).shape, (height, width, 3))
            self.assertEqual(image.size, (width, height))

    def test_invalid_inputs(self):
        for image in [self.image.astype(float), self.image[:1], self.image[:, :, 0]]:
            with self.assertRaises(ValueError):
                ImageWorkspace(image)
        workspace = ImageWorkspace(self.image)
        for method, k in [('bad', 1), ('PCA', 0), ('SVD', 13), ('PCA', 1.5)]:
            with self.assertRaises(ValueError):
                workspace.reconstruct(method, k)
