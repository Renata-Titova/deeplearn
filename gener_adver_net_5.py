import zipfile
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

zip_path = 'D:\\программирование\\deeplearn\\data\\train.zip'
extract_path = 'D:\\программирование\\deeplearn\\celeba_gan'      	
fantasy_zip = zipfile.ZipFile(zip_path)
fantasy_zip.extractall(extract_path)

def load_image(image_path, image_size=(64, 64)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Нормализация сразу здесь
    return image

# Собираем список путей к изображениям
image_paths = tf.io.gfile.glob(os.path.join(extract_path, '*/*.jpg'))

# Создаем tf.data.Dataset
image_size = (64, 64)
batch_size = 32


dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(lambda x: load_image(x, image_size), num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# # Загружаем датасет
# dataset = keras.utils.image_dataset_from_directory(
#     extract_path,  
#     label_mode=None,
#     image_size=(64, 64),
#     batch_size=32)
# print("Датасет успешно загружен.")