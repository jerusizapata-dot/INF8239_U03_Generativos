
from __future__ import annotations

import argparse
import io
import json
import os
import random
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder


ROOT = Path(__file__).resolve().parents[1]
SEED = 42

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class VAE(tf.keras.Model):
    """Autoencoder variacional con pérdida de reconstrucción y KL."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(
            name="reconstruction"
        )
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.reconstruction_tracker,
            self.kl_tracker,
        ]

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            mean, log_variance = self.encoder(images, training=True)

            epsilon = tf.random.normal(tf.shape(mean))
            latent = mean + tf.exp(0.5 * log_variance) * epsilon

            reconstruction = self.decoder(latent, training=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        images, reconstruction
                    ),
                    axis=(1, 2),
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + log_variance
                    - tf.square(mean)
                    - tf.exp(log_variance),
                    axis=1,
                )
            )

            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        self.loss_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }


def save_figure_to_png(figure, path: Path):
    """Guarda una figura usando memoria intermedia."""
    buffer = io.BytesIO()

    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=170,
            bbox_inches="tight",
        )
        image_bytes = buffer.getvalue()

        if not image_bytes:
            raise OSError("La imagen PNG generada está vacía.")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)

        print(f"Imagen guardada: {path}")
    finally:
        buffer.close()
        plt.close(figure)


def image_panel(
    path: Path,
    originals,
    ae_reconstructed,
    vae_reconstructed,
    generated,
):
    """Compara imágenes originales, reconstrucciones y generaciones."""
    figure, axes = plt.subplots(4, 12, figsize=(12, 4))

    collections = [
        originals,
        ae_reconstructed,
        vae_reconstructed,
        generated,
    ]

    try:
        for column in range(12):
            for row, collection in enumerate(collections):
                axes[row, column].imshow(
                    collection[column].squeeze(),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[row, column].axis("off")

        row_labels = [
            "Original",
            "Autoencoder",
            "VAE",
            "Generadas",
        ]

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(
                label,
                rotation=0,
                labelpad=45,
                va="center",
            )

        figure.tight_layout()
    except Exception:
        plt.close(figure)
        raise

    save_figure_to_png(figure, path)


def save_interpolation(decoder, path: Path):
    """Genera una interpolación entre dos puntos del espacio latente."""
    start_point = np.array([-2.0, -1.0])
    end_point = np.array([2.0, 1.0])

    grid = np.array([
        (1 - value) * start_point + value * end_point
        for value in np.linspace(0, 1, 12)
    ])

    interpolation = decoder.predict(grid, verbose=0)

    figure, axes = plt.subplots(1, 12, figsize=(12, 1.5))

    try:
        for axis, image in zip(axes, interpolation):
            axis.imshow(
                image.squeeze(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axis.axis("off")

        figure.tight_layout()
    except Exception:
        plt.close(figure)
        raise

    save_figure_to_png(figure, path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Entrena y compara un Autoencoder y un VAE "
            "con Fashion-MNIST."
        )
    )
    parser.add_argument("--epochs-ae", type=int, default=6)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    if args.epochs_ae < 1 or args.epochs_vae < 1:
        parser.error("Las épocas deben ser mayores que cero.")

    if args.limit < 0:
        parser.error("--limit no puede ser negativo.")

    reports = ROOT / "reports"
    models = ROOT / "models"

    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    # Carga y normalización de Fashion-MNIST.
    (x_train, _), (x_test, _) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    if args.limit:
        x_train = x_train[:args.limit]
        x_test = x_test[:max(100, args.limit // 5)]

    validate_images(x_train)
    validate_images(x_test)

    # Entrenamiento del Autoencoder.
    autoencoder = build_autoencoder(tf)
    autoencoder.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )

    start = perf_counter()

    ae_history = autoencoder.fit(
        x_train,
        x_train,
        validation_split=0.1,
        epochs=args.epochs_ae,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    ae_seconds = perf_counter() - start

    # Entrenamiento del Autoencoder Variacional.
    encoder, decoder = build_encoder_decoder(tf, latent_dim=2)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")

    start = perf_counter()

    vae_history = vae.fit(
        x_train,
        epochs=args.epochs_vae,
        batch_size=256,
        verbose=2,
    )

    vae_seconds = perf_counter() - start

    # Reconstrucciones sobre el conjunto de prueba.
    originals = x_test[:12]

    ae_reconstructed = autoencoder.predict(
        originals,
        verbose=0,
    )

    mean, _ = encoder.predict(originals, verbose=0)

    vae_reconstructed = decoder.predict(
        mean,
        verbose=0,
    )

    # Generación reproducible desde el espacio latente.
    rng = np.random.default_rng(SEED)
    latent_samples = rng.normal(size=(12, 2))

    generated = decoder.predict(
        latent_samples,
        verbose=0,
    )

    # Evidencias visuales.
    image_panel(
        reports / "ae_vae_panel.png",
        originals,
        ae_reconstructed,
        vae_reconstructed,
        generated,
    )

    save_interpolation(
        decoder,
        reports / "interpolation.png",
    )

    # Guardado de modelos.
    autoencoder.save(models / "autoencoder.keras")
    decoder.save(models / "decoder.keras")

    # Registro de métricas y costo computacional.
    metrics = {
        "seed": SEED,
        "dataset": "Fashion-MNIST",
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "ae_epochs_requested": args.epochs_ae,
        "vae_epochs_requested": args.epochs_vae,
        "ae_epochs_completed": len(ae_history.epoch),
        "vae_epochs_completed": len(vae_history.epoch),
        "ae_final_train_loss": float(
            ae_history.history["loss"][-1]
        ),
        "ae_final_validation_loss": float(
            ae_history.history["val_loss"][-1]
        ),
        "vae_final_loss": float(
            vae_history.history["loss"][-1]
        ),
        "vae_final_reconstruction_loss": float(
            vae_history.history["reconstruction"][-1]
        ),
        "vae_final_kl_loss": float(
            vae_history.history["kl"][-1]
        ),
        "ae_train_seconds": ae_seconds,
        "vae_train_seconds": vae_seconds,
        "ae_parameters": int(autoencoder.count_params()),
        "encoder_parameters": int(encoder.count_params()),
        "decoder_parameters": int(decoder.count_params()),
    }

    metrics_path = reports / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RESULTADOS DEL EXPERIMENTO ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nPanel: {reports / 'ae_vae_panel.png'}")
    print(f"Interpolación: {reports / 'interpolation.png'}")
    print(f"Métricas: {metrics_path}")
    print(f"Autoencoder: {models / 'autoencoder.keras'}")
    print(f"Decoder VAE: {models / 'decoder.keras'}")


if __name__ == "__main__":
    main()
