# Álgebra Visual

Aplicación interactiva en Streamlit para conectar álgebra lineal, análisis espectral y reducción dimensional con PCA.

## Ejecutar

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Qué incluye

- **Calculadora espectral:** matrices reales de 2×2 a 4×4, eigenvalores reales o complejos, bases de espacios propios, multiplicidades, diagonalización y verificación de `A·v = λ·v`.
- **Laboratorio PCA:** datasets de ejemplo, datos pegados o archivos CSV, estandarización, varianza explicada, proyección 2D/3D, matriz de covarianza y cargas de los componentes.
- **Compresor RGB:** PCA por canal de color y SVD de rango k, reconstrucción, mapa de diferencia, MSE, PSNR, energía retenida y descarga PNG.

## Comparación de imágenes y rendimiento

El inspector ofrece divisor interactivo, vista lado a lado y diferencias RGB normales o amplificadas ×4. Ambas imágenes usan la misma escala y encuadre, sin estirarse.

- La vista previa predeterminada limita el lado mayor a **720 px**; también hay una opción de **360 px** y **Resolución original**. Nunca se amplían archivos pequeños. La interfaz indica las dimensiones del archivo y las de comparación/descarga.
- Solo se procesa el método seleccionado. **PCA vs. SVD** calcula ambos. Los factores se conservan en la sesión y se reutilizan al cambiar k.
- Se retiene una reconstrucción por método y un único inspector. Cambiar el archivo o la resolución sustituye el espacio de trabajo; quitar el archivo lo libera.
- PCA de imágenes usa SVD sobre cada canal centrado, sin almacenar la matriz de covarianza ni los resultados intermedios del laboratorio PCA. Los factores usan `float32`; los píxeles reconstruidos se redondean y limitan a 0–255.
- La resolución original todavía puede requerir bastante CPU/RAM en la primera descomposición. La descarga tiene la resolución de procesamiento seleccionada.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

Los números que se ingresan en la calculadora son reales; el resultado puede contener eigenvalores y eigenvectores complejos cuando la matriz lo requiere.
