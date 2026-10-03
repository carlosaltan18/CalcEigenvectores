"""Rerun regression: expensive work is lazy, reused, and replaced per upload."""
from io import BytesIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest


def upload(shape):
    data = BytesIO()
    Image.fromarray(np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)).save(data, format='PNG')
    return data


class ImageAppTests(TestCase):
    def test_rank_method_resolution_and_upload_changes(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        with patch('streamlit.file_uploader', return_value=upload((18, 12, 3))) as uploader:
            with patch('image_compression.np.linalg.svd', wraps=np.linalg.svd) as svd:
                app.sidebar.radio[0].set_value('Compresor de imágenes').run()
                self.assertFalse(app.exception)
                self.assertEqual(svd.call_count, 3)
                self.assertEqual(len(app.get('iframe')), 1)
                app.slider[0].set_value(5).run()
                self.assertFalse(app.exception)
                self.assertEqual(svd.call_count, 3)
                # Method widget is the main-area radio, before the sidebar radio.
                method = next(widget for widget in app.radio if widget.label == 'Método')
                method.set_value('SVD rango k').run()
                self.assertEqual(svd.call_count, 6)
                next(widget for widget in app.radio if widget.label == 'Método').set_value('PCA vs. SVD').run()
                self.assertFalse(app.exception)
                self.assertEqual(svd.call_count, 6)
                app.selectbox[1].set_value('PCA vs. SVD').run()
                self.assertEqual(len(app.get('iframe')), 1)
                self.assertEqual(svd.call_count, 6)
                app.selectbox[0].set_value('Resolución original').run()
                self.assertFalse(app.exception)
                self.assertEqual(svd.call_count, 12)
                uploader.return_value = upload((3, 4, 3))
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.slider[0].value, 3)
                entry = app.session_state['image_workspace']
                self.assertEqual(entry['workspace'].image.shape, (3, 4, 3))
                self.assertLessEqual(len(entry['workspace'].results), 2)
                uploader.return_value = None
                app.run()
                self.assertNotIn('image_workspace', app.session_state)
