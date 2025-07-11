# autoencoder.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

class AutoencoderModel:
    """
    Encapsulates a deep, configurable autoencoder with a named bottleneck layer.
    Supports instantiation from scratch or loading an existing Keras model.
    """

    def __init__(self, input_dim=None, model=None, latent_dim=4, hidden_dims=(32, 16)):
        """
        Initialize the AutoencoderModel.
        
        Parameters:
        - input_dim (int): dimension of input features (required if model=None)
        - model (keras.Model): preloaded autoencoder model for inference
        - latent_dim (int): size of the bottleneck layer
        - hidden_dims (tuple[int]): sizes of hidden layers before/after bottleneck
        """
        if model:
            # reuse a loaded model
            self.model = model
            self.input_dim = self.model.input_shape[1]
            try:
                self.encoder = Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer("bottleneck").output
                )
            except ValueError:
                self.encoder = None
        elif input_dim:
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.hidden_dims = hidden_dims
            self.model = self.build_model()
            # encoder for downstream tasks
            self.encoder = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("bottleneck").output
            )
        else:
            raise ValueError("Either input_dim or model must be provided.")

    def build_model(self):
        """
        Build and compile a symmetrical deep autoencoder.
        Encoder: Input → Dense(hidden_dims[0]) → BN → ReLU → Dropout → Dense(hidden_dims[1]) → BN → ReLU → Dropout → Bottleneck(latent_dim)
        Decoder: mirror of encoder
        """
        inp = Input(shape=(self.input_dim,), name="ae_input")
        x = inp

        # --- Encoder ---
        for idx, h in enumerate(self.hidden_dims):
            x = Dense(h, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"enc_dense_{idx}")(x)
            x = BatchNormalization(name=f"enc_bn_{idx}")(x)
            x = Dropout(0.1, name=f"enc_do_{idx}")(x)

        bottleneck = Dense(self.latent_dim, activation="relu", name="bottleneck")(x)

        # --- Decoder ---
        x = bottleneck
        for idx, h in enumerate(reversed(self.hidden_dims)):
            x = Dense(h, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name=f"dec_dense_{idx}")(x)
            x = BatchNormalization(name=f"dec_bn_{idx}")(x)
            x = Dropout(0.1, name=f"dec_do_{idx}")(x)

        out = Dense(self.input_dim, activation="sigmoid", name="ae_output")(x)

        autoencoder = Model(inputs=inp, outputs=out, name="deep_autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder

    def train(self, X, epochs=50, batch_size=256):
        """
        Train the autoencoder on X (where target == X).
        Uses early stopping to prevent overfitting.
        """
        early_stop = EarlyStopping(
            monitor="loss", min_delta=1e-6, patience=5, restore_best_weights=True
        )
        self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stop]
        )

    def encode(self, X):
        """
        Produce latent representations via the bottleneck.
        Raises if no encoder was built or the loaded model lacked the named layer.
        """
        if self.encoder is None:
            raise RuntimeError(
                "No encoder available. Ensure 'bottleneck' layer exists or retrain."
            )
        return self.encoder.predict(X)