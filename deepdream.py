"""
DeepDream — техника визуализации, которая усиливает признаки, обнаруженные в нейронной сети, 
путем градиентного подъема на основе предварительно обученной сверточной сети.

1. Используем предварительно обученную сеть для извлечения признаков из разных слоев
2. Взвешиваем вклад каждого слоя, чтобы управлять тем, какие признаки будут усилены
3. Используем градиентный подъем, чтобы изменить входное изображение так, чтобы активации выбранных слоев 
(потери) становились больше, тем самым усиливая признаки, которые сеть считает важными
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# Получение изображения для экспериментов
base_image_path: str = keras.utils.get_file(
    "coast.jpg", origin="https://img-datasets.s3.amazonaws.com/coast.jpg")

plt.axis("off")
plt.imshow(keras.utils.load_img(base_image_path))

# Создание экземпляра предварительно обученной модели Inception V3
"""
Используем предварительно обученную сверточную сеть для создания модели извлечения признаков, 
которая будет возвращать активации различных промежуточных слоев.
Для каждого слоя получим скалярную оценку, взвешивающую вклад слоя в потери, которые мы будем максимизировать в процессе градиентного восхождения
"""
from tensorflow.keras.applications import inception_v3
model = inception_v3.InceptionV3(weights="imagenet", include_top=False) # удаляем классификационный слой, оставляем сверточный

# Определение вклада каждого слоя в потери DeepDream
layer_settings: Dict[str, float] = { # Слои, для которых максимизируем активации и их веса в общей сумме потерь. 
    # Эти параметры можно настраивать и получать новые визуальные эффекты
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}
outputs_dict = dict( # Символические выходы каждого слоя
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)
#  Модель, возвращающая значения активаций для каждого выбранного слоя (в форме словаря)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

def compute_loss(input_image: tf.Tensor) -> tf.Tensor:
    """
    Вычисляет скалярное значение потерь для изображения
    """
    features = feature_extractor(input_image) # Получаем признаки (значения активаций) из предобученной сети
    loss: tf.Tensor = tf.zeros(shape=()) # Заводим тензор для ф-ции потерь
    for name in features.keys():
        coeff: float = layer_settings[name] # Берем веса каждого слоя
        activation: tf.Tensor = features[name] # Берем активации каждого слоя (карты признаков)
        # Обрезаем активаци на 2 пик, **2 для полож-ых значений, находим среднее и умножаем на весовой коэфф. Добавляем к общему тензору
        loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
    return loss



@tf.function # Для ускорения функции gradient_ascent_step

# Настроим процесс градиентного восхождения
def gradient_ascent_step(image: tf.Tensor, learning_rate: float) -> Tuple[tf.Tensor, tf.Tensor]: 
    """
    image: Текущее изображение
    learning_rate: Скорость обучения, которая определяет, насколько сильно мы будем менять изображение на каждом шаге
        returns: 
    loss: Значение функции потерь 
    image: Обновленное изображение 
    """
    with tf.GradientTape() as tape:
        tape.watch(image) 
        loss: tf.Tensor = compute_loss(image) # вычисляем скалярное значение потерь для текущего изображения
    grads: tf.Tensor = tape.gradient(loss, image) # вычисляем градиент loss относительно image (как нужно изменить пиксель, чтобы увеличить активации)
    grads = tf.math.l2_normalize(grads) # нормализуем
    # Получаем градиентное восхождение
    image += learning_rate * grads # для контроля шага изменения изображения
    return loss, image


def gradient_ascent_loop(image: tf.Tensor, iterations: int, learning_rate: float, max_loss: float=None)-> tf.Tensor:
    """
    Начинает с исходного изображения
    Многократно вызывает gradient_ascent_step, изменяя изображение на каждом шаге
    Периодически выводит значение функции потерь
    Останавливается, если достигнуто максимальное количество итераций или максимальное значение потерь
    Возвращает конечное изображение после обработки
    """

    for i in range(iterations):
        loss, image = gradient_ascent_step(image, learning_rate)
        if max_loss is not None and loss > max_loss: # нужно, чтобы предотвратить неограниченный рост потерь
            break
        print(f"... Loss value at step {i}: {loss:.2f}")
    return image

#  Определим список масштабов обработки (=октав) в которых будут обрабатываться изображения
step: float = 20. # Размер шага градиентного восхождения (насколько сильно будет изменяться изображение на каждом шаге при вычислении градиента)
num_octave: int = 3 #  Количество масштабов, на которых выполняется градиентное восхождение
octave_scale: float = 1.4 # Отношения между соседними масштабами
iterations: int = 30 # Число шагов восхождения для каждого масштаба (сколько раз будет обновляться изображение на каждом масштабе)
max_loss: float = 15. # Если величина потерь вырастет больше этого значения, мы должны прервать процесс градиентного восхождения

# Вспомогательные функции для обработки изображений
import numpy as np

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Открывает изображение, 
    изменяет его размер 
    и преобразует в соответствующий массив
    """
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.inception_v3.preprocess_input(img) # предобработка изображения
    return img

def deprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Выполняет обратное преобразование изображения, чтобы его можно было отобразить на экране
    """
    img = img.reshape((img.shape[1], img.shape[2], 3))
    # Отмена нормализации
    img /= 2.0
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype("uint8")
    return img


#  Выполнение градиентного восхождения с изменением изображения

# Загрузка базового изображения
original_img: np.ndarray = preprocess_image(base_image_path)
original_shape: Tuple[int, int] = original_img.shape[1:3]

# список хранит размеры всех октав (масштабов) изображения
successive_shapes: List[Tuple[int, int]] = [original_shape] 

# Вычисление нужной формы изображения для разных октав (уменьшаем на octave_scale)
for i in range(1, num_octave):
    shape: Tuple[int, int] = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

# Переворачиваем список с размерами, чтобы сначала обрабатывать самые маленькие изображения, а потом самые большие
successive_shapes = successive_shapes[::-1]
# Уменьшаем оригинал картинки до самого маленького размера и сохр копией
shrunk_original_img: tf.Tensor = tf.image.resize(original_img, successive_shapes[0]) 
# Делаем копию исходного изображения
img: tf.Tensor = tf.identity(original_img) 

for i, shape in enumerate(successive_shapes):
    print(f"Processing octave {i} with shape {shape}")
    img = tf.image.resize(img, shape) # масштабируем копию исходного изображения до размера текущей октавы
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    ) # считаем градиентное восхождение на текущем масштабе изображения и перезаписываем изборажние

    upscaled_shrunk_original_img: tf.Tensor = tf.image.resize(shrunk_original_img, shape) # масштабируем уменьшенную копию до размера текущей октавы
    same_size_original: tf.Tensor = tf.image.resize(original_img, shape) # масштабируем оригинал до размера текущей октавы
    lost_detail: tf.Tensor = same_size_original - upscaled_shrunk_original_img # находим разницу в деталях
    img += lost_detail # добавляем потерянные детали к картинке
    shrunk_original_img = tf.image.resize(original_img, shape) # обновляем уменьшенную копию после изменений

keras.utils.save_img("dream.png", deprocess_image(img.numpy())) # сохраняем

