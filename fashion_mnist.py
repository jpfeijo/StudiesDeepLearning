import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)


# Normalizes the images 'uint8' -> 'float32'
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE # Fornece imagens do tipo 'uint8' que vão ser normalizadas
)
ds_train = ds_train.cache() # Conforme ajusta os dados na memória, vai armazenando em cache antes de embaralhar para melhor desempenho
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples) # Embaralha os dados
ds_train = ds_train.batch(128) # Divide os dados em lotes de 128 imagens
ds_train = ds_train.prefetch(tf.data.AUTOTUNE) # Boa prática encerrar o modelo por pre busca para desempenho

# Criar um pipeline de avaliação
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Criar o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compilando o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    #tf.keras.metrics.SparseCategoricalAccuracy()
)

# Treinando o modelo
model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test
)