from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

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
        encoded = Dense(64, activation="relu")(input_layer)
        encoded = Dense(32, activation="relu")(encoded)
        bottleneck = Dense(16, activation="relu", name="bottleneck")(encoded)
        decoded = Dense(32, activation="relu")(bottleneck)
        decoded = Dense(64, activation="relu")(decoded)
        output_layer = Dense(self.input_dim, activation="sigmoid")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self, X, epochs=10, batch_size=512):
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True)

    def encode(self, X):
        return self.encoder.predict(X)