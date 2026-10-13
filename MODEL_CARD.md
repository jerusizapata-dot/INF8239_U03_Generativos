# \# Model Card · Autoencoder y VAE

# 

# \## 1. Modelos y versiones

# 

# \*\*Proyecto:\*\* INF-8239 · Ciencia de Datos II · Unidad 03 · U03.LAB10  

# \*\*Modelos:\*\* Autoencoder (AE) y Autoencoder Variacional (VAE).  

# \*\*Dataset:\*\* Fashion-MNIST.  

# \*\*Semilla aleatoria:\*\* 42.  

# \*\*Entorno:\*\* Python 3.12, TensorFlow 2.21.0 y ejecución en CPU bajo Windows.

# 

# Se implementaron dos modelos para estudiar la reconstrucción y generación de imágenes. El AE aprende una representación comprimida de las imágenes, mientras que el VAE aprende una distribución latente que permite muestrear nuevas imágenes.

# 

# \## 2. Uso previsto y usuarios

# 

# Los modelos están destinados a fines académicos y experimentales:

# 

# \- Estudiar representaciones latentes de imágenes.

# \- Comparar reconstrucción entre un AE y un VAE.

# \- Explorar generación de imágenes a partir de vectores latentes.

# \- Observar interpolaciones entre puntos del espacio latente.

# \- Analizar el costo computacional y las limitaciones de los modelos generativos.

# 

# Los usuarios previstos son estudiantes, docentes y personas que estudian aprendizaje profundo.

# 

# \## 3. Usos fuera de alcance

# 

# Estos modelos no están diseñados para:

# 

# \- Identificar personas ni inferir atributos personales.

# \- Tomar decisiones médicas, financieras, laborales o legales.

# \- Garantizar imágenes realistas o semánticamente correctas.

# \- Representar toda la variedad de prendas y estilos del mundo real.

# \- Utilizarse como sistemas de producción sin validación adicional.

# 

# \## 4. Dataset, licencia y particiones

# 

# Se utilizó Fashion-MNIST, un conjunto de imágenes de prendas de vestir en escala de grises, con dimensiones de 28 × 28 píxeles.

# 

# \- Entrenamiento: 60,000 imágenes.

# \- Prueba: 10,000 imágenes.

# \- Rango de entrada: valores normalizados entre 0 y 1.

# \- Partición de validación del AE: 10 % de los datos de entrenamiento.

# \- Semilla utilizada: 42.

# 

# La fuente original del dataset es Zalando Research, distribuida bajo licencia MIT. La licencia y las condiciones deben verificarse en la fuente oficial antes de redistribuir el conjunto de datos.

# 

# \## 5. Arquitectura y espacio latente

# 

# \### Autoencoder (AE)

# 

# El AE incluye un codificador que comprime la imagen y un decodificador que reconstruye la entrada. Se entrenó con el optimizador Adam y la función de pérdida binary cross-entropy.

# 

# \- Parámetros entrenables: 25,888.

# \- Épocas completadas: 6.

# \- Se aplicó parada temprana con paciencia de 2 épocas y restauración de los mejores pesos.

# 

# \### Autoencoder Variacional (VAE)

# 

# El VAE utiliza un codificador que produce la media y la log-varianza de una distribución latente, además de un decodificador que genera imágenes desde vectores latentes.

# 

# \- Dimensión latente: 2.

# \- Parámetros del codificador: 100,996.

# \- Parámetros del decodificador: 101,520.

# \- Épocas completadas: 8.

# \- Función de pérdida: reconstrucción más divergencia KL.

# 

# El espacio latente bidimensional facilita la exploración y la interpolación, aunque limita la capacidad de representar variaciones complejas.

# 

# \## 6. Métricas de reconstrucción y generación

# 

# Resultados registrados en `reports/training\_metrics.json` para la ejecución con semilla 42:

# 

# | Métrica | Resultado |

# |---|---:|

# | Pérdida final de entrenamiento del AE | 0.3060 |

# | Pérdida final de validación del AE | 0.3071 |

# | Pérdida total final del VAE | 273.6144 |

# | Pérdida de reconstrucción del VAE | 267.3546 |

# | Divergencia KL del VAE | 6.2598 |

# | Tiempo de entrenamiento del AE | 5.16 s |

# | Tiempo de entrenamiento del VAE | 8.67 s |

# 

# Las pérdidas del AE y del VAE no deben compararse directamente como si estuvieran en la misma escala. La pérdida total del VAE suma los componentes de reconstrucción y KL conforme a la implementación.

# 

# Se generaron los siguientes artefactos:

# 

# \- `reports/ae\_vae\_panel.png`: ejemplos originales, reconstrucciones y muestras generadas.

# \- `reports/interpolation.png`: interpolación entre puntos del espacio latente.

# \- `models/autoencoder.keras`: modelo AE entrenado.

# \- `models/decoder.keras`: decodificador del VAE entrenado.

# 

# Las imágenes se inspeccionan visualmente. No se ha registrado una métrica cuantitativa independiente de calidad perceptual o realismo de las imágenes generadas.

# 

# \## 7. Diversidad y vecinos cercanos

# 

# La generación se realizó muestreando vectores de una distribución normal estándar en el espacio latente de dos dimensiones. También se generó una secuencia de interpolación entre dos puntos latentes.

# 

# Estas visualizaciones permiten explorar la variación entre salidas, pero no demuestran por sí solas diversidad suficiente ni ausencia de memorización. No se ha calculado una evaluación cuantitativa de vecinos cercanos ni una métrica formal de diversidad. Por tanto, no se presentan conclusiones definitivas sobre esos aspectos.

# 

# \## 8. Hardware, tiempo y tamaño

# 

# \- Sistema operativo: Windows.

# \- Ejecución: CPU.

# \- GPU: no utilizada en esta ejecución.

# \- Tiempo AE: aproximadamente 5.16 segundos.

# \- Tiempo VAE: aproximadamente 8.67 segundos.

# \- Tamaño del archivo AE: 338,597 bytes.

# \- Tamaño del archivo del decodificador VAE: 425,632 bytes.

# \- Tamaño conjunto aproximado: 764 KB.

# 

# Los tiempos dependen del equipo, las versiones del software y las condiciones de ejecución; no deben interpretarse como una garantía de rendimiento en otros entornos.

# 

# \## 9. Limitaciones, riesgos y supervisión

# 

# Las imágenes son pequeñas, monocromáticas y pertenecen a diez categorías de prendas. Los resultados no representan la complejidad de imágenes de alta resolución ni de otros dominios.

# 

# La calidad visual de una reconstrucción no equivale necesariamente a fidelidad semántica. El VAE puede producir imágenes borrosas o poco reconocibles, y el espacio latente bidimensional puede limitar la diversidad de las muestras.

# 

# La evaluación realizada es académica. Antes de cualquier uso externo, se recomienda revisar visualmente las salidas, documentar las condiciones de ejecución, calcular métricas adicionales y comprobar posibles casos de memorización.

# 

# \## Conclusión

# 

# El desarrollo de este laboratorio permitió implementar, entrenar y analizar dos enfoques de aprendizaje profundo para la reconstrucción y generación de imágenes: el Autoencoder (AE) y el Autoencoder Variacional (VAE), utilizando Fashion-MNIST como conjunto de datos de referencia. Ambos modelos permiten estudiar cómo una red neuronal aprende representaciones comprimidas de los datos, aunque difieren en la forma en que construyen y utilizan el espacio latente.

# 

# En la ejecución realizada, el Autoencoder alcanzó una pérdida de entrenamiento de \*\*0.3060\*\* y una pérdida de validación de \*\*0.3071\*\*, con un tiempo de entrenamiento aproximado de \*\*5.16 segundos\*\*. La proximidad entre ambas pérdidas sugiere un comportamiento consistente en los datos evaluados, aunque no constituye, por sí sola, una prueba definitiva de generalización. Por su parte, el VAE obtuvo una pérdida total de \*\*273.6144\*\*, compuesta por una pérdida de reconstrucción de \*\*267.3546\*\* y una divergencia KL de \*\*6.2598\*\*, con un tiempo de entrenamiento aproximado de \*\*8.67 segundos\*\*. Estas pérdidas no son directamente comparables, ya que responden a objetivos de optimización diferentes.

# 

# Desde el punto de vista metodológico, el AE prioriza la reconstrucción de las imágenes de entrada, mientras que el VAE incorpora una regularización probabilística que facilita el muestreo y la exploración de un espacio latente continuo. Esta diferencia permite comprender que la evaluación de un modelo generativo no debe limitarse a la pérdida final: también debe considerar la calidad visual de las reconstrucciones, la coherencia de las muestras generadas, la diversidad de las salidas y el costo computacional.

# 

# Las visualizaciones obtenidas permiten examinar imágenes originales, reconstrucciones, muestras generadas e interpolaciones entre puntos latentes. Sin embargo, la inspección visual no es suficiente para demostrar que las muestras presentan diversidad adecuada o que el modelo no memoriza ejemplos del entrenamiento. Además, la dimensión latente de dos variables facilita la interpretación visual, pero puede restringir la representación de variaciones complejas presentes en las prendas.

# 

# En términos de eficiencia, el AE requirió menos tiempo de entrenamiento y cuenta con \*\*25,888 parámetros\*\*, frente a los \*\*202,516 parámetros combinados del codificador y decodificador del VAE\*\*. Esta diferencia refleja un mayor costo paramétrico del VAE en la implementación evaluada, asociado a su objetivo probabilístico y a su arquitectura. No obstante, la cantidad de parámetros y el tiempo de ejecución no determinan por sí solos la calidad de las imágenes generadas.

# 

# En conclusión, el laboratorio evidencia que la selección entre un AE y un VAE depende del objetivo de la aplicación: la reconstrucción eficiente constituye una prioridad para el AE, mientras que la exploración de un espacio latente probabilístico y el muestreo de nuevas imágenes son aspectos centrales del VAE. Para establecer una comparación más rigurosa, sería conveniente realizar varias ejecuciones con distintas semillas, medir cuantitativamente la calidad y diversidad de las imágenes y analizar posibles casos de memorización. De esta manera, las conclusiones podrían sustentarse no solo en las pérdidas y los tiempos registrados, sino también en evidencias reproducibles sobre la capacidad de reconstrucción y generación de cada modelo.

