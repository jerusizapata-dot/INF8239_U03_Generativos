# Model Card: Autoencoder y VAE

## Información del proyecto

| Campo | Descripción |
|---|---|
| Asignatura | INF-8239 · Ciencia de Datos II |
| Laboratorio | U03.LAB10 |
| Estudiante | Laudys Jerusi Zapata |
| Autor académico | Edwin Ramón José Nolasco |
| Dataset | Fashion-MNIST |
| Semilla | 42 |
| Entorno | Python 3.12, TensorFlow 2.21.0, CPU en Windows |

## 1. Descripción de los modelos

Este proyecto implementa dos modelos de aprendizaje profundo para estudiar la reconstrucción de imágenes y la generación mediante representaciones latentes.

- **Autoencoder (AE):** comprime una imagen en una representación latente y utiliza un decodificador para reconstruirla.
- **Autoencoder variacional (VAE):** aprende una representación probabilística regularizada mediante la divergencia KL y permite generar muestras a partir del espacio latente.

## 2. Uso previsto

Los modelos están destinados a fines educativos y experimentales:

- Estudiar la compresión y reconstrucción de imágenes.
- Explorar las diferencias entre representaciones latentes deterministas y probabilísticas.
- Generar muestras sintéticas mediante el decodificador del VAE.
- Analizar interpolaciones y costos computacionales.

No se recomienda utilizarlos como sistemas de producción ni extrapolar directamente los resultados a imágenes de mayor resolución.

## 3. Datos y preprocesamiento

| Característica | Descripción |
|---|---|
| Conjunto de datos | Fashion-MNIST |
| Imágenes de entrenamiento | 60,000 |
| Imágenes de prueba | 10,000 |
| Dimensiones | 28 × 28 píxeles |
| Canales | 1, escala de grises |
| Preprocesamiento | Normalización al intervalo [0, 1] |
| Semilla | 42 |

Las imágenes se normalizaron para que los píxeles estuvieran entre 0 y 1. La guía del laboratorio identifica Fashion-MNIST con licencia MIT.

## 4. Arquitectura y espacio latente

### Autoencoder

El AE utiliza un codificador para obtener una representación comprimida y un decodificador para reconstruir las imágenes. Su objetivo principal es reducir el error de reconstrucción.

### Autoencoder variacional

El VAE aprende una representación probabilística del espacio latente. Su pérdida combina el error de reconstrucción y la divergencia KL. El espacio latente de dos dimensiones permite explorar visualmente las coordenadas mediante la aplicación interactiva.

## 5. Resultados experimentales

Resultados de la ejecución registrada con semilla 42.

| Métrica | Autoencoder | VAE |
|---|---:|---:|
| Épocas completadas | 6 | 8 |
| Tiempo de entrenamiento | 5.16 s | 8.67 s |
| Parámetros | 25,888 | 202,516 combinados |
| Pérdida final de entrenamiento | 0.3060 | No comparable en el mismo formato |
| Pérdida de validación del AE | 0.3071 | No registrada aquí |
| Pérdida total del VAE | No aplica en el mismo formato | 273.6144 |
| Pérdida de reconstrucción del VAE | No aplica en el mismo formato | 267.3546 |
| Divergencia KL del VAE | No aplica | 6.2598 |

Los 202,516 parámetros del VAE corresponden a la suma del codificador (100,996) y el decodificador (101,520).

Las pérdidas de ambos modelos no son directamente comparables porque sus objetivos de optimización y escalas son diferentes. En esta ejecución, el AE utilizó menos parámetros y requirió menos tiempo de entrenamiento. Esto no demuestra por sí solo una mejor calidad de reconstrucción o generación.

## 6. Artefactos generados

| Archivo | Propósito |
|---|---|
| `models/autoencoder.keras` | Autoencoder entrenado |
| `models/decoder.keras` | Decodificador del VAE |
| `reports/ae_vae_panel.png` | Panel visual de resultados |
| `reports/interpolation.png` | Interpolación en el espacio latente |
| `reports/training_metrics.json` | Métricas y tiempos |
| `app/streamlit_app.py` | Aplicación interactiva |

## 7. Evaluación de generación y diversidad

El proyecto incluye visualizaciones de reconstrucción, muestras e interpolación, así como una aplicación para explorar cambios en las coordenadas latentes.

La inspección visual permite observar la apariencia de las imágenes y las transiciones entre puntos latentes, pero no demuestra por sí sola diversidad suficiente ni ausencia de memorización.

Para fortalecer la evaluación se recomienda:

- Mostrar 16 ejemplos de prueba seleccionados con una semilla fija.
- Comparar originales, reconstrucciones del AE, reconstrucciones del VAE y muestras nuevas.
- Examinar muestras sin seleccionar únicamente los mejores resultados.
- Comparar las muestras generadas con sus vecinos más cercanos del entrenamiento.
- Añadir métricas cuantitativas de calidad y diversidad cuando corresponda.

Estas verificaciones deben considerarse pendientes hasta que exista evidencia de su ejecución.

## 8. Limitaciones y riesgos

- Fashion-MNIST contiene imágenes pequeñas y monocromáticas.
- El VAE puede producir imágenes borrosas o poco plausibles.
- Una interpolación gradual no garantiza que todas las imágenes intermedias sean realistas.
- Una muestra sintética no garantiza originalidad, privacidad ni ausencia de sesgo.
- La similitud con imágenes de entrenamiento debe analizarse antes de afirmar que no existe memorización.
- Los tiempos dependen del hardware y de las condiciones de ejecución.

## 9. Uso responsable

Estos modelos se presentan con fines educativos. Sus resultados no deben interpretarse como imágenes reales ni como evidencia de que una muestra sea original o privada. Cualquier uso fuera del contexto académico requiere una evaluación adicional de calidad, riesgos y adecuación al propósito.

## 10. Reproducibilidad

Para preparar el entorno, comprobar el runtime, entrenar los modelos y ejecutar las pruebas:

```bash
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
uv run pytest -q
```

La ejecución registrada reportó cuatro pruebas aprobadas. Los resultados y tiempos pueden variar según las versiones del software, el hardware y las condiciones de ejecución.

## 11. Conclusión

El proyecto permitió comparar un AE orientado a reconstrucción con un VAE que incorpora una representación latente probabilística y capacidad de muestreo. El AE presentó menor costo paramétrico y menor tiempo de entrenamiento en la ejecución registrada, mientras que el VAE permitió explorar generación e interpolaciones en el espacio latente.

La selección entre ambos depende del objetivo del experimento. Una evaluación más completa debe considerar reconstrucción, calidad de las muestras, diversidad, posibles similitudes con los datos de entrenamiento y costo computacional.

---

**Repositorio:** [INF8239_U03_Generativos](https://github.com/jerusizapata-dot/INF8239_U03_Generativos)
