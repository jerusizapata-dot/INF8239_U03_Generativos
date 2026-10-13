# INF-8239 · Unidad 03 · Modelos generativos

## LAB10 — Autoencoder y Autoencoder Variacional (VAE)

**Estudiante:** Laudys Jerusi Zapata  
**Autor académico:** Edwin Ramón José Nolasco  
**Programa:** Maestría en Ciencia de Datos e Inteligencia Artificial  
**Proyecto:** Fashion-MNIST · Reconstrucción y generación de imágenes

## 1. Descripción del proyecto

Este laboratorio implementa y compara dos modelos de aprendizaje profundo: el Autoencoder (AE) y el Autoencoder Variacional (VAE), utilizando Fashion-MNIST como conjunto de datos.

El propósito es estudiar la compresión de imágenes, su reconstrucción, la generación de nuevas muestras y la exploración del espacio latente. También se consideran el tiempo de entrenamiento, el número de parámetros y las limitaciones de los modelos.

## 2. Objetivos

- Implementar y entrenar un Autoencoder y un VAE.
- Normalizar las imágenes y preparar los datos de entrenamiento y prueba.
- Comparar los objetivos de reconstrucción de ambos modelos.
- Generar muestras e interpolaciones en el espacio latente.
- Registrar métricas y costos computacionales.
- Documentar las limitaciones, los usos previstos y la responsabilidad académica.

## 3. Tecnologías utilizadas

- Python 3.12
- TensorFlow 2.21.0
- NumPy
- Matplotlib
- Streamlit
- pytest
- uv para la gestión del entorno y las dependencias

La ejecución documentada se realizó en Windows utilizando CPU.

## 4. Dataset

Se utilizó Fashion-MNIST, compuesto por imágenes de prendas de vestir en escala de grises, con dimensiones de 28 × 28 píxeles.

| Característica | Valor |
|---|---:|
| Imágenes de entrenamiento | 60,000 |
| Imágenes de prueba | 10,000 |
| Dimensiones | 28 × 28 |
| Canales | 1 |
| Rango de entrada | [0, 1] |
| Semilla aleatoria | 42 |

Las imágenes se normalizaron para trabajar con valores entre 0 y 1.

## 5. Modelos implementados

### Autoencoder (AE)

El AE aprende una representación comprimida de cada imagen y utiliza un decodificador para reconstruirla. Se entrenó durante 6 épocas.

### Autoencoder Variacional (VAE)

El VAE aprende una representación probabilística del espacio latente. Su función de pérdida combina el error de reconstrucción y la divergencia KL. Se entrenó durante 8 épocas y utiliza un espacio latente de dos dimensiones.

## 6. Resultados experimentales

Los siguientes resultados corresponden a la ejecución con semilla 42.

| Métrica | Autoencoder | VAE |
|---|---:|---:|
| Épocas completadas | 6 | 8 |
| Tiempo de entrenamiento | 5.16 s | 8.67 s |
| Parámetros del modelo | 25,888 | 202,516 combinados |
| Pérdida final de entrenamiento | 0.3060 | — |
| Pérdida de validación del AE | 0.3071 | — |
| Pérdida total del VAE | — | 273.6144 |
| Pérdida de reconstrucción del VAE | — | 267.3546 |
| Divergencia KL del VAE | — | 6.2598 |

Las pérdidas de ambos modelos no son directamente comparables, porque corresponden a objetivos de optimización diferentes. Los tiempos dependen del hardware y de las condiciones de ejecución.

El AE presentó un menor costo paramétrico y un menor tiempo de entrenamiento en esta ejecución. El VAE permite explorar un espacio latente probabilístico y generar muestras mediante muestreo. La elección depende del objetivo del experimento y no de una única métrica.

## 7. Visualizaciones y artefactos generados

- `reports/ae_vae_panel.png`: imágenes originales, reconstrucciones y muestras generadas.
- `reports/interpolation.png`: interpolación entre puntos del espacio latente.
- `reports/training_metrics.json`: métricas y tiempos de entrenamiento.
- `models/autoencoder.keras`: modelo Autoencoder entrenado.
- `models/decoder.keras`: decodificador del VAE.
- `app/streamlit_app.py`: aplicación de demostración con Streamlit.

## 8. Instalación y ejecución

Se requiere Python 3.12 y uv.

### Preparar el entorno

```bash
uv python install 3.12
uv sync --extra cpu
```

### Comprobar el entorno

```bash
uv run python scripts/check_runtime.py
```

### Entrenar los modelos

```bash
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
```

El entrenamiento genera los modelos, las visualizaciones y el archivo de métricas.

### Ejecutar la aplicación

```bash
uv run streamlit run app/streamlit_app.py
```

### Ejecutar las pruebas

```bash
uv run pytest -q
```

## 9. Estructura del proyecto

```text
INF8239_U03_Generativos_Proyecto_Base/
├── app/
│   └── streamlit_app.py
├── models/
│   ├── autoencoder.keras
│   └── decoder.keras
├── reports/
│   ├── ae_vae_panel.png
│   ├── interpolation.png
│   └── training_metrics.json
├── scripts/
│   ├── check_runtime.py
│   └── train_ae_vae.py
├── src/
│   └── inf8239_u03_gen/
├── tests/
├── AI_USE_DECLARATION.md
├── MODEL_CARD.md
├── pyproject.toml
├── requirements-cloud.txt
├── requirements-colab.txt
└── uv.lock
```

## 10. Pruebas y reproducibilidad

Las pruebas automatizadas se ejecutaron con pytest y el resultado registrado fue **4 pruebas aprobadas**.

Se utilizó la semilla 42 para favorecer la reproducibilidad. No obstante, los tiempos y resultados pueden variar según el hardware, las versiones del software y las condiciones de ejecución.

## 11. Limitaciones

Fashion-MNIST contiene imágenes pequeñas y monocromáticas, por lo que los resultados no deben extrapolarse directamente a imágenes reales de mayor complejidad.

La inspección visual de las muestras no demuestra por sí sola diversidad suficiente ni ausencia de memorización. Tampoco se ha establecido una comparación cuantitativa independiente de calidad perceptual entre ambos modelos.

## 12. Conclusión

El laboratorio permitió implementar y evaluar dos enfoques generativos con objetivos diferentes. El AE mostró menor costo paramétrico y menor tiempo de entrenamiento en la ejecución realizada, mientras que el VAE permitió explorar un espacio latente probabilístico y generar muestras mediante muestreo.

Los resultados destacan la importancia de considerar conjuntamente la reconstrucción, la generación, el costo computacional y las limitaciones de evaluación al seleccionar una arquitectura. Para establecer conclusiones más robustas, se recomienda realizar experimentos con varias semillas y añadir métricas cuantitativas de calidad, diversidad y posibles casos de memorización.

## 13. Documentación adicional

- [Repositorio del proyecto en GitHub](https://github.com/jerusizapata-dot/INF8239_U03_Generativos)
- `MODEL_CARD.md`: características, usos previstos, métricas y limitaciones de los modelos.
- `AI_USE_DECLARATION.md`: declaración del uso de herramientas de inteligencia artificial como apoyo académico.