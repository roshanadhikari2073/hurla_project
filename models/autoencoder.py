# models/autoencoder.py
# ─────────────────────────────────────────────────────────────────────────────
# Production-grade deep auto-encoder used by the HURLA 2018 pipeline.
# • Works in two modes:
#     1.  Fresh training  → AutoencoderModel(input_dim=77, …)
#     2.  Inference only → AutoencoderModel(model=load_model(...))
# • Exposes `encoder` (bottleneck) for downstream analytics if the named layer
#   "bottleneck" is present.
# • The layer layout is symmetric and narrow (latent_dim=4 by default) which
#   is well suited to network-flow reconstruction.
# • Early-stopping is built-in and the compile() step is done once inside
#   build_model(), so the class can be instantiated safely inside a long-
#   running inference service without recompilation penalties.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Sequence, Optional

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping


class AutoencoderModel:
    """
    Encapsulates a deep, column-schema–aligned auto-encoder.

    Parameters
    ----------
    input_dim : int, optional
        Number of numeric features **after** schema alignment. Required when
        training from scratch.
    model : keras.Model, optional
        Pre-loaded Keras model; skip architecture build and use for inference.
    latent_dim : int, default 4
        Size of the compression (bottleneck) layer.
    hidden_dims : Sequence[int], default (32, 16)
        Encoder hidden-layer widths; decoder mirrors this list in reverse.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        input_dim: Optional[int] = None,
        *,
        model: Optional[Model] = None,
        latent_dim: int = 4,
        hidden_dims: Sequence[int] = (32, 16),
    ):
        if model is not None:
            # Inference-only path — reuse the supplied model artefact
            self.model: Model = model
            self.input_dim: int = self.model.input_shape[1]
        elif input_dim is not None:
            # Training path — build fresh architecture
            self.input_dim = input_dim
            self._latent_dim = latent_dim
            self._hidden_dims = tuple(hidden_dims)
            self.model = self._build_model()
        else:
            raise ValueError("Provide either `input_dim` (train) or `model` (inference).")

        # Try to expose encoder → will fail gracefully if layer missing
        try:
            self.encoder: Optional[Model] = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("bottleneck").output,
                name="encoder",
            )
        except ValueError:
            self.encoder = None  # encoder unavailable (e.g. older artefact)

    # ──────────────────────────────────────────────────────────────────────
    # Architecture
    # ──────────────────────────────────────────────────────────────────────
    def _build_model(self) -> Model:
        """Return a compiled auto-encoder with symmetrical encoder/decoder."""
        inp = Input(shape=(self.input_dim,), name="ae_input")
        x = inp

        # ---------- Encoder ----------
        for idx, width in enumerate(self._hidden_dims):
            x = Dense(
                width,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
                name=f"enc_dense_{idx}",
            )(x)
            x = BatchNormalization(name=f"enc_bn_{idx}")(x)
            x = Dropout(0.10, name=f"enc_do_{idx}")(x)

        bottleneck = Dense(self._latent_dim, activation="relu", name="bottleneck")(x)

        # ---------- Decoder ----------
        x = bottleneck
        for idx, width in enumerate(reversed(self._hidden_dims)):
            x = Dense(
                width,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
                name=f"dec_dense_{idx}",
            )(x)
            x = BatchNormalization(name=f"dec_bn_{idx}")(x)
            x = Dropout(0.10, name=f"dec_do_{idx}")(x)

        out = Dense(self.input_dim, activation="linear", name="ae_output")(x)

        model = Model(inputs=inp, outputs=out, name="deep_autoencoder")
        model.compile(optimizer="adam", loss="mse")
        return model

    # ──────────────────────────────────────────────────────────────────────
    # Training utility
    # ──────────────────────────────────────────────────────────────────────
    def train(
        self,
        X,
        *,
        epochs: int = 25,
        batch_size: int = 1024,
        validation_split: float = 0.10,
        patience: int = 4,
    ) -> None:
        """
        Fit the auto-encoder on `X` (target = input), with early stopping.

        Parameters
        ----------
        X : ndarray
            Scaled feature matrix (shape: n_samples × input_dim).
        epochs : int
            Maximum training epochs.
        batch_size : int
            Mini-batch size.
        validation_split : float
            Fraction of `X` used for validation loss monitoring.
        patience : int
            Early-stopping patience in epochs.
        """
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )
        self.model.fit(
            X,
            X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=2,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Convenience wrappers
    # ──────────────────────────────────────────────────────────────────────
    def encode(self, X):
        """
        Return latent vectors from the bottleneck layer.

        Raises
        ------
        RuntimeError
            If the underlying model lacks a layer named "bottleneck".
        """
        if self.encoder is None:
            raise RuntimeError(
                "Encoder unavailable: this model has no 'bottleneck' layer."
            )
        return self.encoder.predict(X, verbose=0)

    @staticmethod
    def load(path: str) -> "AutoencoderModel":
        """One-liner to reload a saved Keras model."""
        from tensorflow.keras.models import load_model

        return AutoencoderModel(model=load_model(path))