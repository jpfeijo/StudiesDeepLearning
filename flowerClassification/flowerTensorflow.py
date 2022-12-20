import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = pd.read_csv('iris.csv')

xs = base.iloc[:, 0:4].values
ys = base.iloc[:, 4].values

values = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

xs = xs.astype('float32')
ys = np.asarray([values[y] for y in ys], dtype=np.int32)

#for x in xs: print(type(x))
#print(ys)
#for y in ys: print(type(y))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(xs.dtype)

model_fit = model.fit(
    xs,
    ys,
    epochs=50,
    batch_size=1
)

plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['loss'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

save = input('Save model? (y/n): ')
if save == 'y':
    model.save('flowerModel.h5')
    print('Model saved!')
else:
    print('Model not saved!')

predictions = model.predict(xs)
