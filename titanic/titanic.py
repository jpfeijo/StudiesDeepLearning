import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Loading the dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Dropping unnecessary columns
data_train = data_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data_test = data_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Mapping values
sex_values = {
    'male' : 0,
    'female' : 1
}
embarked_values = {
    'S' : 0,
    'C' : 1,
    'Q' : 2
}

# Dealing with missing values
data_train = data_train.fillna({'Embarked' : 'S'})
data_train = data_train.fillna({'Age' : data_train.Age.mean()})

data_test = data_test.fillna({'Age' : data_test.Age.mean()})
data_test = data_test.fillna({'Embarked' : 'S'})
data_test = data_test.fillna({'Fare' : data_test.Fare.mean()})

# Mapping values
data_train['Sex'] = data_train.Sex.map(sex_values)
data_train['Embarked'] = data_train.Embarked.map(embarked_values)

data_test['Sex'] = data_test.Sex.map(sex_values)
data_test['Embarked'] = data_test.Embarked.map(embarked_values)

# Splitting the dataset
x_train = data_train.iloc[:, 1:11].values
y_train = data_train.iloc[:, 0].values

x_test = data_test.iloc[:, 1:11].values

#print(data_train.info())
#print(data_test.isnull().sum())
print(data_train)

# Converting to numpy arrays
y_train = np.asarray(y_train, dtype=np.int32)
x_train = np.asarray(x_train, dtype='float32')

x_test = np.asarray(x_test, dtype='float32')


# Creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training the model
model_fit = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=1,
    verbose=2
)

# Evaluating the model
#predictions = model.predict(x_test)

# Plotting the results
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['loss'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

#print(predictions)
#print(data_train.head())