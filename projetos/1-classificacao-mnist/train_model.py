import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

val_split = 5000
x_val = x_train[-val_split:]
y_val = y_train[-val_split:]
x_train = x_train[:-val_split]
y_train = y_train[:-val_split]

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

val_acc = max(history.history["val_accuracy"])
print(f"\nValidacao final - acuracia: {val_acc:.4f}")

model.save("model.h5")
print("Modelo salvo como model.h5")

try:
    import tf_keras as tfk
    model.save("model_tmp.keras")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = tfk.models.load_model("model_tmp.keras")
    legacy.save("model.h5")
    import os
    os.remove("model_tmp.keras")
    print("Modelo convertido para formato Keras 2 (compativel com TF 2.12-2.15)")
except ImportError:
    pass
