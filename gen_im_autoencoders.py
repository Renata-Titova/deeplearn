"""
Реализуем вариационный автокодировщик, способный генерировать изображения цифр, похожие на изображения 
в наборе данных MNIST
Он будет состоять из трех частей:
 1. сети кодировщика, превращающей реальное изображение в среднее и дисперсию в скрытом пространстве;
 2. слоя выбора образца, принимающего среднее значение и дисперсию 
 и использующего их для выбора случайной точки в скрытом пространстве;
 3. сети декодера, превращающей точки из скрытого пространства обратно в изображения
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Tuple

# Сеть кодировщика VAE
latent_dim: int = 2 # Размерность скрытого пространства: двумерная плоскость
# Создаем входной слой 28x28 с 1 каналом
encoder_inputs: keras.Input = keras.Input(shape=(28, 28, 1))
# Создаем сверточный слой (Conv2D) (кол-во выходных фильтров, размер ядра свертки, ф-я активацит, способ обработки границ изображения (с сохранением размера))
x: tf.Tensor = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# Преобразуем выходной тензор в одномерный вектор
x = layers.Flatten()(x)
# Создаем полносвязный слой (Dense)
x = layers.Dense(16, activation="relu")(x)
# Создаем полносвязный слой, который вычисляет среднее значение и логарифм дисперсии
z_mean: tf.Tensor = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var: tf.Tensor = layers.Dense(latent_dim, name="z_log_var")(x)
encoder: keras.Model = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder") # изображение, среднее, дисперсия

encoder.summary()


# Слой выбора точки из скрытого пространства
class Sampler(layers.Layer):
    def call(self, z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
        batch_size: tf.Tensor = tf.shape(z_mean)[0]  # сколько изображений за раз обрабатывается
        z_size: tf.Tensor = tf.shape(z_mean)[1]  # размерность скрытого пространства
        epsilon: tf.Tensor = tf.random.normal(shape=(batch_size, z_size))  # тензор случайных значений из станд. нормального распределения
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
   
latent_inputs: keras.Input = keras.Input(shape=(latent_dim,)) # Входной слой декодера 
#  Произвести столько же коэффициентов из вектора, сколько имеется на уровне слоя Flatten в кодировщике
x: tf.Tensor = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# вектор обратно в 3-мерный тензор как в Flatten
x = layers.Reshape((7, 7, 64))(x)
#  Восстановить слой Conv2D кодировщика
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# Выход будет иметь форму (28, 28, 1) (grayscale)
decoder_outputs: tf.Tensor = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
decoder: keras.Model = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()


# Модель VAE с нестандартным методом train_step()
class VAE(keras.Model):
    def __init__(self, encoder: Model, decoder: Model, **kwargs: Any):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        # Эти метрики используются для слежения за средними значениями потерь в каждой эпохе
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    @property
    def metrics(self) -> List[keras.metrics.Mean]:
        """
        Перечисляем метрики в свойстве metrics, чтобы модель могла сбрасывать их 
        после каждой эпохи (или между вызовами fit()/evaluate())
        """
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]


    def train_step(self, data) -> Dict[str, tf.Tensor]:
        """
        Суммируем потери при реконструкции по пространственным измерениям (оси 1 и 2) и берем их средние знач
        """
        with tf.GradientTape() as tape:
            # Пропускаем входное изобр через кодировщик и получает среднее и логарифм дисперсии скрытого представления
            z_mean, z_log_var = self.encoder(data)
            # Выбирает точку из скрытого пространства
            z = self.sampler(z_mean, z_log_var)
            # Пропускает точку через декодер и получает реконструированное изображение
            reconstruction = decoder(z)
            # Вычисляем потерю реконструкции
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            #  Добавляем член регуляризации (расхождение Кульбака — Лейблера), чтобы скрытое пространство было непрерывным
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        # Вычисляем градиенты общей потери по всем обучаемым параметрам сети     
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Применяем градиенты для изменения обучаемых параметров
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

# Обучение VAE
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits: np.ndarray = np.concatenate([x_train, x_test], axis=0) # Обучение на полном наборе MNIST -> объединили обучающий и контрольный наборы
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# Создаем экземпляр модели
vae: keras.Model = VAE(encoder, decoder)
# не передаем аргумент loss в вызов compile(), потому что вычисление потерь выполняется в train_step()
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
vae.fit(mnist_digits, epochs=30, batch_size=128) # не передаем цели в метод fit(), поскольку их нет в train_step

# Отобразить сетку 30 × 30 цифр (всего 900 цифр)
n: int = 30
digit_size: int = 28
figure: np.ndarray = np.zeros((digit_size * n, digit_size * n))
# Выбрать ячейки, линейно распределенные в двумерной сетке
grid_x: np.ndarray = np.linspace(-1, 1, n)
grid_y: np.ndarray = np.linspace(-1, 1, n)[::-1]

# Выполнить обход ячеек в сетке
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        # Для каждой ячейки выбрать цифру и добавить в изобр
        z_sample: np.ndarray = np.array([[xi, yi]])
        x_decoded: np.ndarray = vae.decoder.predict(z_sample)
        digit: np.ndarray = x_decoded[0].reshape(digit_size, digit_size)
        figure[
            i * digit_size : (i + 1) * digit_size,
            j * digit_size : (j + 1) * digit_size,
        ] = digit

plt.figure(figsize=(15, 15))
start_range = digit_size // 2
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.axis("off")
plt.imshow(figure, cmap="Greys_r")

# Сохраняем изображение
fname = "vae_latent_space.png"
keras.utils.save_img(fname, figure)  # Используем keras.utils.save_img