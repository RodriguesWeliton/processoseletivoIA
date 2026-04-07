import os
import tensorflow as tf


def convert_to_tflite(keras_model_path, tflite_model_path):
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model not found: {keras_model_path}")

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    original_size = os.path.getsize(keras_model_path)
    optimized_size = os.path.getsize(tflite_model_path)
    print(f"Saved TensorFlow Lite model to {tflite_model_path}")
    print(f"Original size: {original_size / 1024:.2f} KB")
    print(f"Optimized size: {optimized_size / 1024:.2f} KB")


def main():
    keras_model_path = "mnist_cnn.h5"
    tflite_model_path = "mnist_cnn.tflite"
    convert_to_tflite(keras_model_path, tflite_model_path)


if __name__ == "__main__":
    main()
