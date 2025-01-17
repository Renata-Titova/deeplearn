import numpy as np
import tarfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
#  original_distribution — это одномерный массив NumPy значений вероятностей, сумма которых должна быть равна 1; 
# temperature — это коэффициент, определяющий уровень энтропии выходного распределения
def reweight_distribution(original_distribution, temperature=0.5):
    """
    Низкая температура = высокая предсказуемость, высокая температура = более случайный результат
    """
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    #  Возвращает взвешенную версию оригинального распределения. 
    # Сумма вероятностей в новом распределении может получиться больше 1, поэтому разделим элементы вектора на сумму для нового распр-я
    return distribution / np.sum(distribution)

# Скачиваем данные
# Добавила в папку меньше текстов, чтобы побыстрее проверять
# Проверяем, существует ли папка data, если нет, то создаем ее
if not os.path.exists("data"):
    os.makedirs("data")

# Проверяем, существует ли файл, если нет, то сообщаем об этом
if not os.path.exists("data/aclImdb_v1.tar.gz"):
    print("Error: File 'aclImdb_v1.tar.gz' not found in the 'data' folder.")
else:
    # Распаковка архива
    print("Unpacking 'aclImdb_v1.tar.gz'...")
    with tarfile.open("data/aclImdb_v1.tar.gz", "r:gz") as tar:
        tar.extractall(".")
    print("Archive unpacked successfully!")

# Создание набора данных из текстовых файлов
# Определяем base_path
base_path = "aclImdb/train" 
# text_dataset_from_directory -> функция для создания набора данных TensorFlow из текстовых файлов в определенной директории
# label_mode=None -> не нужны метки классов
# batch_size -> размер пакета для набора данных, по 256 текстовых файла 
# ВОТ ТУТ ОШИБКА ВОЗНИКАЕТ
dataset = keras.utils.text_dataset_from_directory(directory=base_path, label_mode=None, batch_size=256)
# map (объединение) -> применяет lambda функцию к каждому элементу в наборе данных
# lambda принимает на вход один аргумент (текст из файла) и возвращает преобразованный текст
#  "<br />" -> Удаление HTML-тегов
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))


# Подготовка слоя TextVectorization -> Преобразовывает пакет текстовых строк в пакет последовательностей целых чисел 
sequence_length = 100 # максимальная длина последовательности токенов
vocab_size = 15000 # размер словаря
text_vectorization = TextVectorization( 
    max_tokens=vocab_size, # максимальный размер словаря
    output_mode="int", # формат вывода слоя
    output_sequence_length=sequence_length, # длина выходной последовательности
)
text_vectorization.adapt(dataset) # проходит по всем текстам в dataset, создаёт и заполняет словарь + индексы



# Настройка набора данных для языковой модели
def prepare_lm_dataset(text_batch):
    """
    создает смещение 
    """
    vectorized_sequences = text_vectorization(text_batch) # Преобразовывает пакет текстовых строк в пакет последовательностей целых чисел 
    # смещение
    x = vectorized_sequences[:, :-1] # берем все токены в последовательности, кроме последнего входные данные
    y = vectorized_sequences[:, 1:] #  Создаем цели смещением послед-ти на 1, мы берем все токены в последовательности, кроме первого
    return x, y
lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)


# Создаем класс PositionalEmbedding
class PositionalEmbedding(layers.Layer):
    # __init__ -> конструктор класса для создания экземпляра класса
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        # super -> для вызова родительского класса
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        """
        Вычисляет длину входной последовательности
        Создает тензор с номерами позиций
        Преобразует токены в векторные представления
        Преобразует позиции в векторные представления
        Складывает векторные представления токенов и позиций 
        """
        length = tf.shape(inputs)[-1] # Вычисляем фактическую длину входной последовательности
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        """
        Создает маску, которая показывает, какие элементы в последовательности являются валидными, а какие — нет
        """
        return tf.math.not_equal(inputs, 0) # True = ненулевой элемент в inputs,False = ноль

    def get_config(self):
        """
        Возвращает параметры слоя
        """
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

# Создаем класс TransformerDecoder
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim #  Размер входного вектора токенов
        self.dense_dim = dense_dim #  Размер внутреннего полносвязного слоя
        self.num_heads = num_heads #  кол-во голов внимания
        self.attention_1 = layers.MultiHeadAttention( # смотрит на входную последовательность токенов, чтобы понять, как связаны между собой слова в последовательности
          num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
          num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential( # полносвязная сеть
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        # Слои нормализации помогают стабилизировать и ускорить обучение
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        """
        Возвращает параметры слоя для сохранения и загрузки модели
        """
        config = super(TransformerDecoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        """
        Создаёт маску, в которой каждое слово может видеть только предыдущие слова в последовательности
        чтобы игнорировать нули, которые добавлялись в батче
        Механизм, который позволяет слою знать, какие элементы в последовательности являются реальными токенами
        (индексами слов), а какие являются нулями-заполнителями
        """
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        """
        Применяет MultiHeadAttention к входной последовательности 
        Добавляет остаточное соединение (residual connection) и нормализацию
        Применяет MultiHeadAttention ко входу и выходу кодировщика 
        Добавляет остаточное соединение и нормализацию
        Применяет полносвязную сеть
        Добавляет остаточное соединение и нормализацию
        """
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = mask
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


# Модель «последовательность в последовательность» 
# Простая языковая модель на основе архитектуры Transformer 
embed_dim = 256 # размерность векторного представления слов
latent_dim = 2048 # внутренняя размерность, через которую проходят данные в слоях Transformer
num_heads = 2 # параметр для Multi-Head Attention внутри Transformer Decoder

# Создаем входной слой Input
# shape=(None,) -> на вход последовательность любой длины 
# dtype="int64" -> входные данные (индексы слов) - целые числа 64 бита
inputs = keras.Input(shape=(None,), dtype="int64")

# Создаем входной слой для модели
# PositionalEmbedding -> добавляет информацию о положении слова в последовательности к его векторному представлению
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs) # Применяем слой к входным данным inputs

# Преобразуем входные индексы слов x в векторные представления
# добавляем информацию о позиции каждого слова в последовательности с механизмом внимания
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x) # Применяем слой к входным данным x, т.к. яз модель предсказывает следующее слово по предыдущим

# Создаем полносвязный выходной (Dense) слой, обрабатываем последовательность x
# Кол-во нейронов=размер словаря
# softmax -> каждое значение – вероятность, что соответствующий токен является следующим токеном в последовательности
# outputs -> вероятностное распределение по всему словарю, говорящее о том, какое слово скорее всего будет следующим
outputs = layers.Dense(vocab_size, activation="softmax")(x) 
# Принимает на вход: Последовательности индексов слов
# Преобразуем outputs в вероятности для каждого слова в словаре
model = keras.Model(inputs, outputs)

# Компилируем модель 
# sparse_categorical_crossentropy -> для задач классификации, когда выходы – это целые числа (индексы классов)
# rmsprop -> адаптирует темп обучения во время тренировки
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop") 

# Обратный вызов для генерации текста
tokens_index = dict(enumerate(text_vectorization.get_vocabulary())) # ключи — числовые индексы, значения — соответствующие токены

def sample_next(predictions, temperature=1.0):
    """
    Отвечает за выбор следующего токена с учетом вероятностей, предсказанных моделью
    Она принимает на вход вектор вероятностей, предсказанных моделью для каждого токена в словаре, 
    и возвращает индекс выбранного токена. 
    Задача – учесть предсказанные вероятности и добавить элемент случайности, используя температуру
    """
    predictions = np.asarray(predictions).astype("float64") # Вектор вероятностей для каждого возможного токена
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions) # для неотрицательных значений
    predictions = exp_preds / np.sum(exp_preds) # Нормализация вероятностей (сумма всех вероятностей равна 1)
    probas = np.random.multinomial(1, predictions, 1) # Выбирается 1 токен из выборки
    return np.argmax(probas)

# Callback вызывается после каждой эпохи обучения, используется для генерации текста на основе обученной модели 
# Генерирует и выводит текст, начав с prompt, предсказывая по одному токену за раз
class TextGenerator(keras.callbacks.Callback): 
    def __init__(self,
                 prompt, # Вводная фраза
                 generate_length, # Количество токенов, которые нужно сгенерировать
                 model_input_length, # Длина входной последовательности модели
                 temperatures=(1.,), # Отвечает за случайность генерации
                 print_freq=1): # Частота вывода сгенерированного текста
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.print_freq = print_freq
        vectorized_prompt = text_vectorization([prompt])[0].numpy()
        self.prompt_length = np.nonzero(vectorized_prompt == 0)[0][0]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return
        # Генерирует текст для каждой temperature
        for temperature in self.temperatures:
            print("== Generating with temperature", temperature)
            sentence = self.prompt
            # Генерирует текст по одному токену за итерацию
            for i in range(self.generate_length):
                tokenized_sentence = text_vectorization([sentence]) # Векторизует текущий сгенерированный текст
                predictions = self.model(tokenized_sentence) # Получает предсказания модели
                next_token = sample_next(
                    predictions[0, self.prompt_length - 1 + i, :]
                ) # Выбирает следующий токен
                sampled_token = tokens_index[next_token] # Преобразует индекс в сам токен
                sentence += " " + sampled_token # Добавляет выбранный токен к тексту
            print(sentence)

prompt = "This movie" # подсказка для начала генерации
text_gen_callback = TextGenerator(
    prompt,
    generate_length=50,
    model_input_length=sequence_length,
    temperatures=(0.2, 0.5, 0.7, 1., 1.5)) #  Для демонстрации влияния температуры на генерацию текста используем несколько разных температур

# Обучение языковой модели
model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])