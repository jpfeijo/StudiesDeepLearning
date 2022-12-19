import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

test_images = test_images / 255.0
train_images = train_images / 255.0

model = keras.Sequential([
    # Todas as imagens devem ser inseridas no mesmo tamanho e tipo
    keras.layers.Flatten(input_shape=(28,28)), # Sem neuronios, apenas transforma a imagem que foi inputada em um array
    keras.layers.Dense(128, activation='relu'),
    #keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax') # 10 neuronios, uma para cada tipo de classe -> Softmax ajdua a encontrar a classe que está mais próxima
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_fit = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=100,
    verbose = 1 # 0 = silent, 1 = progress bar, 2 = one line per epoch
)

test_loss, test_acc = model.evaluate(test_images, test_labels)

plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['loss'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

#predictions = model.predict(test_images)