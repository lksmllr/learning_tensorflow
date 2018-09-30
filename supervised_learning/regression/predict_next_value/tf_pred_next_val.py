# author lksmllr

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

# generate some data 
no_entries = 1000
no_features = 4
step = 2
gen_data = np.zeros(shape=(no_entries, no_features))
gen_labels = np.zeros(shape=(no_entries, 1))
for i in range(0, no_entries):
	count = i
	for j in range(0, no_features):
		gen_data[i][j] = count
		count += step
	gen_labels[i][0] = count

# Shuffle the data set
order = np.argsort(np.random.random(gen_labels.shape[0]))
gen_data = gen_data[order]
gen_labels = gen_labels[order]

# split to train and test
len_data = len(gen_data)
tmp_index = int(len_data*0.7)

train_data = gen_data[:tmp_index]
train_labels = gen_labels[:tmp_index]
test_data = gen_data[tmp_index:]
test_labels = gen_labels[tmp_index:]

# check if everything is right
print(train_data[0])
print(train_labels[0])

# build model
def build_model():
	model = keras.Sequential([
    keras.layers.Dense(2, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    #keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1)
  	])

	#optimizer = tf.train.RMSPropOptimizer(0.001)
	optimizer = tf.train.AdamOptimizer()

	model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
	return model

model = build_model()
model.summary()

# Train the model
# The model is trained for 500 epochs, and record the training and validation accuracy 
# in the history object.

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

# Let's see how did the model performs on the test set:
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("\n")
print("\nTesting set Mean Abs Error: {:7.2f}".format(mae))
print("Test Data is: " + str(test_data[0]))
print("Label is: " + str(test_labels[0]))
predictions = model.predict(test_data)
print("Prediction: " + str(predictions[0]))
print("\n")

def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [1000$]')
	plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
	plt.legend()
	plt.ylim([0, 5])
	plt.show()

plot_history(history)








