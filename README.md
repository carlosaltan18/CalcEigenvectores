# Álgebra Visual

Aplicación interactiva en Streamlit para conectar álgebra lineal, análisis espectral y reducción dimensional con PCA.

## Ejecutar

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Qué incluye

- **Calculadora espectral:** matrices reales de 2×2 a 4×4, eigenvalores reales o complejos, bases de espacios propios, multiplicidades, diagonalización y verificación de `A·v = λ·v`.
- **Laboratorio PCA:** datasets de ejemplo o datos CSV pegados, estandarización, varianza explicada, proyección, matriz de covarianza y cargas de los componentes.
- **Compresor PCA:** reconstrucción de imágenes en escala de grises, control del número de componentes, MSE, PSNR, porcentaje de varianza y descarga PNG.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

Los números que se ingresan en la calculadora son reales; el resultado puede contener eigenvalores y eigenvectores complejos cuando la matriz lo requiere.
