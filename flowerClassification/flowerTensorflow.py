import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Loading the dataset
base = pd.read_csv('iris.csv')
base_test = pd.read_csv('irisTest.csv')

x_train = base.iloc[:, 0:4].values
y_train = base.iloc[:, 4].values

x_test = base_test.iloc[:, 0:4].values
y_test = base_test.iloc[:, 4].values

values = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

x_train = x_train.astype('float32')
y_train = np.asarray([values[y] for y in y_train], dtype=np.int32)

#for x in x_train: print(type(x))
#for y in y_train: print(type(y))

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(x_train.dtype)

# Training the model
model_fit = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=1,
    verbose=0
)

# Evaluating the model
predictions = model.predict(x_test)
print(predictions)

# Plotting the results
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['loss'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Saving the model
save = input('Save model? (y/n): ')
if save == 'y':
    model.save('flowerModel.h5')
    print('Model saved!')
else:
    print('Model not saved!')

