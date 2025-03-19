import tensorflow as tf
generator = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)), tf.keras.layers.Dense(784, activation='sigmoid')])
discriminator = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)), tf.keras.layers.Dense(1, activation='sigmoid')])