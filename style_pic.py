"""
Получение стиля и содержимого изображений
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple, List, Dict


# Скачиваем изоражения
base_image_path: str = keras.utils.get_file(
    "sf.jpg", origin="https://img-datasets.s3.amazonaws.com/sf.jpg")
style_reference_image_path: str = keras.utils.get_file(
    "starry_night.jpg", origin="https://img-datasets.s3.amazonaws.com/starry_night.jpg")

# Загружаем оригинальную картинку, получаем ширину и высоту и сохраняем в переменных
original_width, original_height = keras.utils.load_img(base_image_path).size

# Размеры генерируемого изображения
img_height: int = 400
img_width: int = round(original_width * img_height / original_height)

# Вспомогательные функции
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Открывает изображение, изменяет его размер и преобразует в соответствующий массив
    """
    img = keras.utils.load_img(
        image_path, target_size=(img_height, img_width)) #  меняем размер во время загрузки
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0) # добавляем размерность для батча
    img = keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_image(img: np.ndarray) -> np.ndarray:
    """
     Вспомогательная функция для преобразования массива NumPy в допустимое изображение
    """
    img = img.reshape((img_height, img_width, 3)) # Делаем массив 3-мерным (высота, ширина, каналы)
    # Центрировать относительно нуля путем удаления среднего значения пикселя из ImageNet
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    #  Конвертировать изображения из BGR в RGB
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img

# Используем предварительно обученную сверточную сеть и создадим модель, извлекающую признаки и возвращающую активации промежуточных слоев (трех)
# weights="imagenet" -> загружаем предварительно обученные веса модели
# include_top=False -> удаляем полносвязные слои, оставляем только сверточные, нам классификация не нужна
model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)
# Создаем список [имя слоя, активация], проходимся по всем слоям в модели 
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# Модель, возвращающая значения активаций всех целевых слоев
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

# Определим функцию потерь содержимого, которая позволит гарантировать сходство представлений целевого (исходного) и сгенерированного изображений в верхнем слое сети
def content_loss(base_img: tf.Tensor, combination_img: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.square(combination_img - base_img))

# Функция потерь стиля
def gram_matrix(x: tf.Tensor) -> tf.Tensor:
    """
    Вычисляет матрицу Грама из входной карты признаков x. 
    Матрица Грама описывает корреляцию между разными каналами в карте признаков
    """    
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style_img: tf.Tensor, combination_img: tf.Tensor) -> tf.Tensor:
    """
    Вычисляет скалярное значение, которое показывает, насколько сильно различается стиль у двух изображений
    """
    S = gram_matrix(style_img)
    C = gram_matrix(combination_img)
    channels = 3
    size = img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

#  Функция общей потери вариации
def total_variation_loss(x: tf.Tensor) -> tf.Tensor:
    """
    Принимает на вход тензор x = сгенерированное изображение
    Вычисляет меру изменения в сгенерированном изображении
    Чем больше значение total_variation_loss, тем шумнее изображение, позволяет избежать появления мозаичного эффекта
    """
    a = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] - x[:, 1:, : img_width - 1, :]
    )
    b = tf.square(
        x[:, : img_height - 1, : img_width - 1, :] - x[:, : img_height - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


# Функция общей потери вариации, которая  будет минимизироваться
# Список слоев, участвующих в вычислении потери стиля
style_layer_names: List[str] = [ 
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
content_layer_name: str = "block5_conv2" # Слой, используемый для вычисления потерь содержимого
total_variation_weight: float = 1e-6 # Вес вклада общей потери вариации
style_weight: float = 1e-6 # Вес вклада потери стиля
content_weight: float = 2.5e-8 # Вес вклада потери содержимого чем > оно, тем > сходство сгенерированного и ориг. изобр.

def compute_loss(combination_image: tf.Tensor, base_image: tf.Tensor, style_reference_image: tf.Tensor) -> tf.Tensor:
    """
    combination_image: Генерируемое изображение
    base_image: Изображение содержимого
    style_reference_image: Изображение стиля
        returns:
    loss: общую функцию потерь
    """
    # объединяем изображения в один тезор
    input_tensor: tf.Tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    # Извлекаем признаки из нужных слоев
    features:  Dict[str, tf.Tensor] = feature_extractor(input_tensor)
    loss: tf.Tensor = tf.zeros(shape=()) # Заводим тензор для ф-ии потерь
    # Извлекаем признаки из верхнего слоя (для 3-х изображений) Добавление потери содержимого
    layer_features: tf.Tensor = features[content_layer_name]
    base_image_features: tf.Tensor = layer_features[0, :, :, :]
    combination_features: tf.Tensor = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Добавление потери стиля
    for layer_name in style_layer_names:
        # Получаем признаки для текущего слоя
        layer_features = features[layer_name]
        style_reference_features: tf.Tensor = layer_features[1, :, :, :]
        combination_features: tf.Tensor = layer_features[2, :, :, :]
        # Считаем, на сколько отличается стиль у 2-х изображений
        style_loss_value: tf.Tensor = style_loss(
          style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * style_loss_value

    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

# Настройка процесса градиентного спуска
@tf.function # для ускорения
def compute_loss_and_grads(combination_image: tf.Tensor, base_image: tf.Tensor, style_reference_image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Вычисляет общую функцию потерь и градиенты потерь по отношению к сгенерированному изображению
    """
    with tf.GradientTape() as tape:
        loss: tf.Tensor = compute_loss(combination_image, base_image, style_reference_image)
    grads: tf.Tensor = tape.gradient(loss, combination_image) # показывает направление и величину изменения изображения для минимизации потерь
    return loss, grads

# Создаем оптимизатор градиентного спуска с затуханием скорость обучения 100, а затем будем уменьшать ее на 4 % через каждые 100 шагов
optimizer: keras.optimizers.Optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)
# Предобрабатываем базовое и стилевое изображения, загружая и подготавливая их
base_image: np.ndarray = preprocess_image(base_image_path)
style_reference_image: np.ndarray = preprocess_image(style_reference_image_path)
combination_image: tf.Variable = tf.Variable(preprocess_image(base_image_path)) # сгенерированное изображение, которое будет изменяться в процессе обучения

iterations: int = 4000 # количество итераций оптимизации
# Проходимся по каждой итерации
for i in range(1, iterations + 1):
    # считаем функцию потерь и градиенты на текущей итерации
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    # Обновим комбинированное изображение в направлении уменьшения потери передачи стиля
    # Применяем вычисленные градиенты к сгенерированной картинке с помощью оптимизатора SGD, изменяя ее для уменьшения функции потерь
    optimizer.apply_gradients([(grads, combination_image)])
    # Выводим информацию каждые 100 итераций
    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss:.2f}")
        # Преобразуем и сохраняет изображение
        img = deprocess_image(combination_image.numpy())
        fname = f"combination_image_at_iteration_{i}.png"
        keras.utils.save_img(fname, img)