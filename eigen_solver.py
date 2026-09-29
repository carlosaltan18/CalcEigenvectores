"""Cálculo explicable de eigenvalores, eigenvectores y propiedades espectrales."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sympy import Matrix, det, factor, latex, nsimplify, roots, simplify, symbols


class EigenSolver:
    """Resuelve ``det(A - λI) = 0`` y conserva la información simbólica útil."""

    def __init__(self, matrix: List[List[float]]) -> None:
        self.A_numeric = np.asarray(matrix, dtype=float)
        if self.A_numeric.ndim != 2 or self.A_numeric.shape[0] != self.A_numeric.shape[1]:
            raise ValueError("La matriz debe ser cuadrada.")
        self.A_sympy = Matrix([[nsimplify(value) for value in row] for row in matrix])
        self.n = self.A_numeric.shape[0]
        self.steps: List[Dict[str, Any]] = []
        self._root_multiplicities: Dict[Any, int] = {}
        self._geometric_multiplicities: Dict[Any, int] = {}

    @staticmethod
    def _to_complex(value: Any) -> complex:
        return complex(value.evalf())

    @staticmethod
    def _display_scalar(value: complex, precision: int = 5) -> str:
        real = 0.0 if abs(value.real) < 1e-10 else value.real
        imag = 0.0 if abs(value.imag) < 1e-10 else value.imag
        if imag == 0:
            return f"{real:.{precision}f}"
        sign = "+" if imag >= 0 else "−"
        return f"{real:.{precision}f} {sign} {abs(imag):.{precision}f}i"

    def solve(self) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Devuelve una base de cada espacio propio y los pasos de la solución."""
        self.steps = []
        self._root_multiplicities = {}
        self._geometric_multiplicities = {}
        lam = symbols("lambda")
        identity = Matrix.eye(self.n)

        self.steps.append(
            {
                "step": 1,
                "title": "Construir la matriz identidad",
                "description": f"Se construye I de tamaño {self.n} × {self.n}.",
                "latex": f"I_{{{self.n}}} = {latex(identity)}",
            }
        )

        characteristic_matrix = self.A_sympy - lam * identity
        self.steps.append(
            {
                "step": 2,
                "title": "Formar A − λI",
                "description": "Se resta λ en la diagonal de la matriz A.",
                "matrix_display": True,
                "matrix": characteristic_matrix,
            }
        )

        characteristic_polynomial = factor(simplify(det(characteristic_matrix)))
        self.steps.append(
            {
                "step": 3,
                "title": "Obtener el polinomio característico",
                "description": "Se calcula det(A − λI) y se factoriza cuando es posible.",
                "latex": f"\\det(A - \\lambda I) = {latex(characteristic_polynomial)}",
            }
        )

        root_map = roots(characteristic_polynomial, lam)
        if not root_map:
            raise ValueError("No fue posible resolver el polinomio característico.")
        self._root_multiplicities = dict(root_map)
        root_description = ", ".join(
            f"{latex(root)} (m={multiplicity})" for root, multiplicity in root_map.items()
        )
        self.steps.append(
            {
                "step": 4,
                "title": "Resolver det(A − λI) = 0",
                "description": "Los valores m indican la multiplicidad algebraica de cada raíz.",
                "latex": f"\\lambda: {root_description}",
            }
        )

        pairs: List[Tuple[complex, np.ndarray]] = []
        for index, (eigenvalue, algebraic_multiplicity) in enumerate(root_map.items(), start=1):
            eigenspace = (self.A_sympy - eigenvalue * identity).nullspace()
            geometric_multiplicity = len(eigenspace)
            self._geometric_multiplicities[eigenvalue] = geometric_multiplicity
            eigenvalue_numeric = self._to_complex(eigenvalue)

            if not eigenspace:
                self.steps.append(
                    {
                        "step": f"5 ({index})",
                        "title": f"Espacio propio para λ = {self._display_scalar(eigenvalue_numeric)}",
                        "description": "No se encontró una base no trivial para el espacio propio.",
                        "found": False,
                    }
                )
                continue

            for vector_index, eigenvector in enumerate(eigenspace, start=1):
                numeric_vector = np.asarray(
                    [self._to_complex(component) for component in eigenvector], dtype=complex
                )
                norm = np.linalg.norm(numeric_vector)
                normalized_vector = numeric_vector / norm if norm > 1e-12 else numeric_vector
                pairs.append((eigenvalue_numeric, normalized_vector))

                self.steps.append(
                    {
                        "step": f"5 ({index}.{vector_index})",
                        "title": f"Base del espacio propio de λ = {self._display_scalar(eigenvalue_numeric)}",
                        "description": (
                            f"Multiplicidad algebraica: {algebraic_multiplicity}; "
                            f"multiplicidad geométrica: {geometric_multiplicity}."
                        ),
                        "latex": f"v_{{{index},{vector_index}}} = {latex(eigenvector)}",
                        "found": True,
                    }
                )

        pairs.sort(key=lambda pair: (round(pair[0].real, 12), round(pair[0].imag, 12)))
        eigenvalues = np.asarray([pair[0] for pair in pairs], dtype=complex)
        eigenvectors = (
            np.column_stack([pair[1] for pair in pairs]).astype(complex)
            if pairs
            else np.empty((self.n, 0), dtype=complex)
        )

        self.steps.append(
            {
                "step": 6,
                "title": "Interpretar los resultados",
                "description": (
                    f"Se obtuvieron {len(pairs)} vector(es) en las bases de los espacios propios. "
                    "Las raíces repetidas se muestran con sus multiplicidades."
                ),
                "found": bool(pairs),
            }
        )
        return eigenvalues, eigenvectors, self.steps

    def analysis(self) -> Dict[str, Any]:
        """Resume propiedades relevantes de A para la interfaz educativa."""
        if not self._root_multiplicities:
            self.solve()

        determinant = simplify(self.A_sympy.det())
        trace = simplify(self.A_sympy.trace())
        rank = int(self.A_sympy.rank())
        geometric_total = sum(self._geometric_multiplicities.values())
        is_diagonalizable = geometric_total == self.n
        is_symmetric = bool(self.A_sympy == self.A_sympy.T)
        has_complex = any(abs(self._to_complex(root).imag) > 1e-10 for root in self._root_multiplicities)

        return {
            "determinant": determinant,
            "trace": trace,
            "rank": rank,
            "is_invertible": determinant != 0,
            "is_symmetric": is_symmetric,
            "is_diagonalizable": is_diagonalizable,
            "has_complex_eigenvalues": has_complex,
            "algebraic_multiplicities": self._root_multiplicities,
            "geometric_multiplicities": self._geometric_multiplicities,
        }

    def verify(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> List[Dict[str, Any]]:
        """Verifica numéricamente la igualdad ``A v = λ v`` para cada vector."""
        verifications: List[Dict[str, Any]] = []
        for eigenvalue, vector in zip(eigenvalues, eigenvectors.T):
            transformed = self.A_numeric.astype(complex) @ vector
            scaled = eigenvalue * vector
            residual = float(np.linalg.norm(transformed - scaled))
            verifications.append(
                {
                    "eigenvalue": eigenvalue,
                    "eigenvector": vector,
                    "Av": transformed,
                    "lambda_v": scaled,
                    "residual": residual,
                    "verified": residual < 1e-8,
                }
            )
        return verifications
