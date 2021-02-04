import pickle
import numpy as np
import time as time
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
from keras.utils import to_categorical

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

testdict = unpickle('./cifar-10-batches-py/test_batch')
datadict1 = unpickle('./cifar-10-batches-py/data_batch_1')
datadict2 = unpickle('./cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('./cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('./cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('./cifar-10-batches-py/data_batch_5')
labeldict = unpickle('./cifar-10-batches-py/batches.meta')

X1 = datadict1['data']
X2 = datadict2['data']
X3 = datadict3['data']
X4 = datadict4['data']
X5 = datadict5['data']
Y1 = datadict1['labels']
Y2 = datadict2['labels']
Y3 = datadict3['labels']
Y4 = datadict4['labels']
Y5 = datadict5['labels']

X1 = X1.reshape(10000, 3, 32, 32).astype("int")
X2 = X2.reshape(10000, 3, 32, 32).astype("int")
X3 = X3.reshape(10000, 3, 32, 32).astype("int")
X4 = X4.reshape(10000, 3, 32, 32).astype("int")
X5 = X5.reshape(10000, 3, 32, 32).astype("int")

testDataArray = testdict["data"]
testLabelArray = testdict["labels"]

testDataArray = testDataArray.reshape(10000, 3, 32, 32).astype("int")

dataArray = np.concatenate([X1, X2])
dataArray = np.concatenate([dataArray, X3])
dataArray = np.concatenate([dataArray, X4])
dataArray = np.concatenate([dataArray, X5])

printDataArray = dataArray.transpose(0, 2, 3, 1)

labelArray = np.concatenate([Y1, Y2])
labelArray = np.concatenate([labelArray, Y3])
labelArray = np.concatenate([labelArray, Y4])
labelArray = np.concatenate([labelArray, Y5])

dataArray = dataArray.transpose(0, 2, 3, 1)
testDataArray = testDataArray.transpose(0, 2, 3, 1)

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
labelNamesArray = labeldict["label_names"]

testDataArray = np.array(testDataArray)
testDataArray = testDataArray.transpose(0, 3, 1, 2)
testDataArray = testDataArray.reshape(10000, 3072)

testLabelArray = np.array(testLabelArray)
testLabelArray = to_categorical(testLabelArray)
dataArray = np.array(dataArray)
labelArray = np.array(labelArray)
labelArray = to_categorical(labelArray)

dataArray = dataArray.transpose(0, 3, 1, 2)
dataArray = dataArray.reshape(50000, 3072)

print('labels: ', labelArray.shape)

inputs = keras.Input(shape=(3072), name='img')

x = layers.Dense(3072, activation='elu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='elu')(x)
x = layers.Dropout(0.001)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='elu')(x)
x = layers.Dropout(0.01)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='elu')(x)
x = layers.Dropout(0.1)(x)
x = layers.BatchNormalization()(x)

outputs = layers.Dense(10, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='model')
print(model.summary())

model.compile(
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Nadam(0.0005),
    metrics = ['accuracy']
)

history = model.fit(dataArray, labelArray, batch_size = 200, epochs = 100, validation_split = 0.2)

test_scores = model.evaluate(testDataArray, testLabelArray, verbose = 2)
print('Test loss: ', test_scores[0])
print('Test accuracy: ', test_scores[1])
