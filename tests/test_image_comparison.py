import unittest
from unittest.mock import patch
from base64 import b64decode
from io import BytesIO

import numpy as np
from PIL import Image

from image_comparison import aligned_pair, amplify_difference, comparison_html, difference_map, png_url


class ComparisonTests(unittest.TestCase):
    def test_portrait_landscape_and_square_keep_exact_grid(self):
        for shape in [(19, 7, 3), (7, 19, 3), (8, 8, 3)]:
            original = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
            reconstructed = original.copy()
            left, right = aligned_pair(original, reconstructed)
            self.assertIs(left, original)
            np.testing.assert_array_equal(left, right)
            decoded = np.asarray(Image.open(BytesIO(b64decode(png_url(left).split(',')[1]))))
            np.testing.assert_array_equal(decoded, original)

    def test_rejects_mismatched_dimensions_and_invalid_pixels(self):
        source = np.zeros((4, 8, 3), dtype=np.uint8)
        for invalid in [source.transpose(1, 0, 2), source.astype(float), source[:, :, 0], source[:0]]:
            with self.assertRaises(ValueError):
                aligned_pair(source, invalid)

    def test_absolute_error_is_symmetric_without_uint8_wrap(self):
        left = np.array([[[0, 255, 20], [100, 40, 10]]], dtype=np.uint8)
        right = np.array([[[255, 0, 30], [100, 30, 40]]], dtype=np.uint8)
        expected = [[[255, 255, 10], [0, 10, 30]]]
        np.testing.assert_array_equal(difference_map(left, right), expected)
        np.testing.assert_array_equal(difference_map(right, left), expected)
        np.testing.assert_array_equal(difference_map(left, left), np.zeros_like(left))

    def test_amplification_clips_after_multiplication(self):
        left = np.array([[[1, 64, 255]]], dtype=np.uint8)
        right = np.zeros_like(left)
        np.testing.assert_array_equal(difference_map(left, right, 4), [[[4, 255, 255]]])
        np.testing.assert_array_equal(difference_map(left, right, 1.5), [[[1, 96, 255]]])
        for factor in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                difference_map(left, right, factor)

    def test_amplifies_existing_map_without_overflow_or_mutation(self):
        difference = np.array([[[0, 1, 64]]], dtype=np.uint8)
        np.testing.assert_array_equal(amplify_difference(difference, 4), [[[0, 4, 255]]])
        np.testing.assert_array_equal(amplify_difference(difference, 1e300), [[[0, 255, 255]]])
        np.testing.assert_array_equal(difference, [[[0, 1, 64]]])
        with self.assertRaises(ValueError):
            amplify_difference(difference, float('nan'))

    def test_inspector_reuses_encodings_and_calculates_error_once(self):
        source = np.zeros((3, 4, 3), dtype=np.uint8)
        with patch('image_comparison.difference_map', wraps=difference_map) as difference:
            with patch('image_comparison.png_url', wraps=png_url) as encode:
                html = comparison_html(source, source, 'PCA', 1, 'test',
                                       original_url='data:original', reconstructed_url='data:reconstructed')
        self.assertEqual(difference.call_count, 1)
        self.assertEqual(encode.call_count, 2)  # Only the two error maps.
        self.assertEqual(html.count('data:original'), 1)
        self.assertEqual(html.count('data:reconstructed'), 1)

    def test_html_has_pixel_dimensions_and_escaped_labels(self):
        source = np.zeros((19, 7, 3), dtype=np.uint8)
        html = comparison_html(source, source, '<PCA>', 2, 'test"key')
        self.assertIn('7 × 19 px', html)
        self.assertIn('&lt;PCA&gt; · k=2', html)
        self.assertIn('test&quot;key', html)
        self.assertNotIn('@@', html)


if __name__ == '__main__':
    unittest.main()
