import unittest

import numpy as np

from eigen_solver import EigenSolver
from pca_tools import fit_pca, image_pca


class EigenSolverTests(unittest.TestCase):
    def test_handles_complex_eigenvalues(self):
        solver = EigenSolver([[0, -1], [1, 0]])
        eigenvalues, eigenvectors, _ = solver.solve()

        self.assertEqual(eigenvectors.shape, (2, 2))
        self.assertTrue(np.allclose(np.sort(eigenvalues.imag), [-1, 1]))
        self.assertTrue(all(item["verified"] for item in solver.verify(eigenvalues, eigenvectors)))

    def test_reports_defective_matrix(self):
        solver = EigenSolver([[2, 1], [0, 2]])
        eigenvalues, eigenvectors, _ = solver.solve()
        analysis = solver.analysis()

        self.assertEqual(eigenvalues.shape, (1,))
        self.assertEqual(eigenvectors.shape, (2, 1))
        self.assertFalse(analysis["is_diagonalizable"])
        self.assertEqual(list(analysis["algebraic_multiplicities"].values()), [2])

    def test_returns_full_basis_for_repeated_diagonal_eigenvalue(self):
        solver = EigenSolver([[3, 0], [0, 3]])
        eigenvalues, eigenvectors, _ = solver.solve()

        self.assertEqual(eigenvalues.shape, (2,))
        self.assertEqual(eigenvectors.shape, (2, 2))
        self.assertTrue(solver.analysis()["is_diagonalizable"])


class PcaToolsTests(unittest.TestCase):
    def test_full_component_reconstruction_is_exact(self):
        data = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 7.0], [6.0, 8.0]])
        result = fit_pca(data, n_components=2, standardize=True)

        np.testing.assert_allclose(result["reconstructed"], data, atol=1e-10)
        self.assertAlmostEqual(float(result["cumulative_ratio"][-1]), 1.0)

    def test_constant_image_has_safe_variance_ratio(self):
        result = image_pca(np.full((5, 5), 127, dtype=np.uint8), n_components=1)

        self.assertEqual(result["mse"], 0.0)
        self.assertEqual(float(result["cumulative_ratio"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
