import tensorflow as tf
import os

model = tf.keras.models.load_model("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

original_size = os.path.getsize("model.h5") / 1024
optimized_size = len(tflite_model) / 1024
print(f"model.h5: {original_size:.1f} KB")
print(f"model.tflite: {optimized_size:.1f} KB")
print(f"Reducao: {(1 - optimized_size / original_size) * 100:.1f}%")
