import tensorflow as tf
from src.visualization import plot_history


class BaseLineMethod:
    def __init__(self, settings, paths, data_loader):
        self.model = None
        self.history = None
        self.settings = settings
        self.data_loader = data_loader
        self.num_of_labels = settings.num_of_labels
        self.save_path = paths.path_result[0]
        self.create_model()

    def create_model(self):
        input_shape = [self.data_loader.train_df.shape[1]]

        self.model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=input_shape),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dense(units=self.num_of_labels, activation='sigmoid')
        ])

        # Compile models
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['binary_accuracy', tf.keras.metrics.AUC()],
        )

    def train(self):
        self.history = self.model.fit(
            self.data_loader.train_df, self.data_loader.labels_df,
            batch_size=self.settings.batch_size,
            epochs=self.settings.epochs
        )

    def predict(self, test_df):
        return self.model.predict(test_df)

    def save_results(self):
        plot_history(self.history, save_path=self.save_path, save_fig=True)
