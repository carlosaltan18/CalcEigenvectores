import numpy as np
import plotly.graph_objects as go
from sympy import symbols, det, Matrix, solve, simplify, latex, nsimplify
from typing import Tuple, List, Dict


class EigenSolver:
    """
    Implements the Eigenvalues_and_Eigenvectors algorithm using the
    characteristic polynomial det(A - λI) = 0.
    """

    def __init__(self, matrix: List[List[float]]):
        self.A_numeric = np.array(matrix, dtype=float)
        self.A_sympy = Matrix([[nsimplify(x) for x in row] for row in matrix])
        self.n = len(matrix)
        self.steps: List[Dict] = []

    def solve(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Main solver following the algorithm:
        1. Build identity matrix I
        2. Compute B = A - λI
        3. Compute det(B)
        4. Solve det(A - λI) = 0
        5. For each λ, solve (A - λI)v = 0
        """
        self.steps = []
        lam = symbols('lambda')

        # Step 1: Build identity matrix
        I = Matrix.eye(self.n)
        self.steps.append({
            "step": 1,
            "title": "Construir la matriz identidad I",
            "description": f"Se construye I de tamaño {self.n}×{self.n}",
            "formula": f"I = {self._matrix_to_str(np.eye(self.n))}",
            "latex": f"I_{{{self.n}}} = \\text{{eye}}({self.n})"
        })

        # Step 2: Compute B = A - λI
        B = self.A_sympy - lam * I
        self.steps.append({
            "step": 2,
            "title": "Calcular B = A − λI",
            "description": "Se resta λ veces la identidad a la matriz A",
            "formula": f"B = A - λ·I",
            "matrix_display": True,
            "matrix": B
        })

        # Step 3: Compute det(B)
        char_poly = det(B)
        char_poly_simplified = simplify(char_poly)
        self.steps.append({
            "step": 3,
            "title": "Calcular det(A − λI)",
            "description": "Se calcula el polinomio característico",
            "formula": f"det(A − λI) = {char_poly_simplified}",
            "latex": f"\\det(A - \\lambda I) = {latex(char_poly_simplified)}"
        })

        # Step 4: Solve det(A - λI) = 0
        eigenvalues_sympy = solve(char_poly_simplified, lam)
        self.steps.append({
            "step": 4,
            "title": "Resolver det(A − λI) = 0",
            "description": "Se iguala el polinomio a cero y se resuelve",
            "formula": f"Eigenvalores encontrados: {[str(simplify(ev)) for ev in eigenvalues_sympy]}",
            "latex": f"\\lambda = {', '.join([latex(simplify(ev)) for ev in eigenvalues_sympy])}"
        })

        # Step 5: Get eigenvalues as floats
        eigenvalues_float = [complex(ev) for ev in eigenvalues_sympy]

        # Steps 6-15: For each eigenvalue, find eigenvector
        all_eigenvalues = []
        all_eigenvectors = []

        pairs = []

        for i, (ev_sympy, ev_float) in enumerate(zip(eigenvalues_sympy, eigenvalues_float)):
            ev_real = float(ev_float.real)

            # Step 7: C = A - λI
            C = self.A_sympy - ev_sympy * I
            C_simplified = C.applyfunc(simplify)

            # Step 8: Solve C·v = 0 (null space)
            try:
                null_space = C_simplified.nullspace()
                if null_space:
                    # Step 10-11: Non-trivial solution exists
                    evec_sympy = null_space[0]
                    evec_float = np.array([float(simplify(x)) for x in evec_sympy])

                    # Normalize
                    norm = np.linalg.norm(evec_float)
                    if norm > 1e-10:
                        evec_normalized = evec_float / norm
                    else:
                        evec_normalized = evec_float

                    self.steps.append({
                        "step": f"6-11 (λ_{i+1})",
                        "title": f"Eigenvector para λ = {ev_real:.4f}",
                        "description": f"Se calcula C = A − ({ev_real:.4f})·I y se resuelve C·v = 0",
                        "formula": f"v_{i+1} = {np.array2string(evec_normalized, precision=4)}",
                        "latex": f"v_{{{i+1}}} = {latex(evec_sympy)}",
                        "found": True
                    })
                    pairs.append((ev_real, evec_normalized))
                    
                else:
                    # Step 12-13: No non-trivial solution
                    self.steps.append({
                        "step": f"6-13 (λ_{i+1})",
                        "title": f"λ = {ev_real:.4f} — sin eigenvector no trivial",
                        "description": "No se encontró solución no trivial. Se continúa con el siguiente λ.",
                        "formula": "",
                        "found": False
                    })
            except Exception as e:
                self.steps.append({
                    "step": f"Error (λ_{i+1})",
                    "title": f"Error procesando λ = {ev_real:.4f}",
                    "description": str(e),
                    "formula": "",
                    "found": False
                })
        pairs.sort(key=lambda x: x[0])

        all_eigenvalues = [p[0] for p in pairs]
        all_eigenvectors = [p[1] for p in pairs]

        self.steps.append({
            "step": 16,
            "title": "Retornar eigenvalores λ y eigenvectores v",
            "description": f"Se encontraron {len(all_eigenvalues)} pares (λ, v)",
            "formula": f"λ = {[round(ev, 4) for ev in all_eigenvalues]}",
            "latex": ""
        })

        eigenvalues_arr = np.array(all_eigenvalues)
        eigenvectors_arr = np.column_stack(all_eigenvectors) if all_eigenvectors else np.array([])

        return eigenvalues_arr, eigenvectors_arr, self.steps

    def _matrix_to_str(self, mat: np.ndarray) -> str:
        rows = []
        for row in mat:
            rows.append("[" + "  ".join(f"{x:6.2f}" for x in row) + "]")
        return "\n".join(rows)

    def verify(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> List[Dict]:
        """Verify Av = λv for each pair."""
        verifications = []
        for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            Av = self.A_numeric @ vec
            lambda_v = lam * vec
            residual = np.linalg.norm(Av - lambda_v)
            verifications.append({
                "eigenvalue": lam,
                "eigenvector": vec,
                "Av": Av,
                "lambda_v": lambda_v,
                "residual": residual,
                "verified": residual < 1e-6
            })
        return verifications