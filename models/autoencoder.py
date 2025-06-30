from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

class AutoencoderModel:
    def __init__(self, input_dim=None, model=None):
        if model:
            self.model = model
            self.input_dim = self.model.input_shape[1]
            self.encoder = Model(inputs=self.model.input, outputs=self.model.get_layer("bottleneck").output)
        elif input_dim:
            self.input_dim = input_dim
            self.model = self.build_model()
            self.encoder = Model(inputs=self.model.input, outputs=self.model.get_layer("bottleneck").output)
        else:
            raise ValueError("Either input_dim or model must be provided.")

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))

        # Reduce encoder capacity and add L2 regularization
        encoded = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(input_layer)
        encoded = Dropout(0.1)(encoded)
        bottleneck = Dense(8, activation="relu", name="bottleneck")(encoded)

        # Decoder mirrors encoder
        decoded = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(bottleneck)
        decoded = Dropout(0.1)(decoded)
        output_layer = Dense(self.input_dim, activation="sigmoid")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self, X, epochs=50, batch_size=256):
        # Early stopping based on minimal validation loss drop
        early_stop = EarlyStopping(
            monitor='loss',
            min_delta=1e-6,
            patience=5,
            restore_best_weights=True
        )
        self.model.fit(X, X,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       callbacks=[early_stop])

    def encode(self, X):
        return self.encoder.predict(X)